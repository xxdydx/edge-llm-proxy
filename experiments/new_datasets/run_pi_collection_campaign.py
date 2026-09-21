"""Permanent Pi Coder collection campaign orchestrator.

Orchestrates the evidence-faithful multi-phase permanent collection campaign for Pi Coder:
1. W3 Pi A/C capture (`run_pi_dataset_ac.py` -> `w3_capture/pi_dataset_ac_v1`)
   - Requires exactly 21 unique preflight tasks crossed with edge/cloud (42 cells).
   - Validates that all 42 cells achieve terminal capture classification before grading.
2. Official SWE-Bench-Fork grading (`grade_pi_dataset_ac.py`)
   - Requires completed cells to have valid official grades recorded.
3. Isolated remote trace synchronization
   - Syncs isolated remote trace roots `/workspace/flowmesh/traces/pi_dataset_ac_v1/cloud`
     and `edge` via `ssh -p 31226 gw@lum.id` into a versioned local campaign trace root.
   - Strictly scrubs secrets without printing tokens/credentials.
4. Isolated Pi Dataset A & C ingestion and export into a versioned root
   - No synthetic Dataset A/C records: ingests only real Pi run metadata plus real
     edgeproxy request/response traces joined by session_id and official grade artifacts.
   - Fails closed on absent/mismatched traces, duplicate sessions, or missing grades.
   - Absolutely no turn-count budget anywhere; 1800-second trajectory budget comes from W3 metadata.
5. W4 prospective Dataset B certification gate check
   - Evaluates the real prospective certification gate; halts safely if blocked.
6. W4 prospective Dataset B collection (`run_pi_dataset_b.py`) ONLY if certified.

Resumable and checkpoints all phase transitions atomically.
Guarantees strict isolation from legacy Claude data: never pools or overwrites
legacy Claude datasets or SQLite stores.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable
import uuid

logger = logging.getLogger("pi_campaign")

REPO_ROOT = Path(__file__).resolve().parents[2]
ND = REPO_ROOT / "experiments/new_datasets"

# Canonical protocol versions for the permanent Pi campaign
PROTOCOL_PI_AC = "pi-dataset-ac-v1"
PROTOCOL_PI_B = "pi-dataset-b-v1"
CAMPAIGN_ID_DEFAULT = "pi-permanent-v1"
COHORT_DEFAULT = "pi-permanent"

# Preflight configuration: exactly 21 unique preflight tasks x 2 backends = 42 cells
SOURCE_FILES_PREFLIGHT = (
    ND / "w2_preflight/smoke_instances.json",
    ND / "w2_preflight/train50_5tasks_instances.json",
    ND / "w2_preflight/gym_batch2_14tasks_instances.json",
)
BACKENDS_PREFLIGHT = ("edge", "cloud")
EXPECTED_TASK_COUNT = 21
EXPECTED_CELL_COUNT = 42

TERMINAL_CAPTURE_CLASSIFICATIONS = {
    "completed",
    "timeout",
    "repeat",
    "process",
    "setup",
    "protocol",
}

# Disallowed legacy protocols that must never be mixed into Pi datasets
LEGACY_CLAUDE_PROTOCOLS = {
    "smoke-v1",
    "train50-batch1-v1",
    "gym-batch2-v1",
    "timeout-recovery-v2",
    "timeout-recovery-v2-edge-8k-no-thinking-repeat4",
    "harness-ab-pi-v1",
}

# Forbidden turn-count / turn-limit budget fields
FORBIDDEN_TURN_BUDGET_KEYS = {
    "max_main_logical_calls",
    "calls_used_so_far",
    "max_turns",
    "turn_limit",
    "turn_count_budget",
    "max_turn_count",
    "turns_remaining",
    "turn_budget",
    "turns_used",
}

# Known sensitive environment variables and patterns to scrub
KNOWN_SECRETS_VARS = {
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "FLOWMESH_PI_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "LUMID_TOKEN",
}
SECRET_PATTERNS = re.compile(
    r"(token|auth|key|secret|password|credential)", re.IGNORECASE
)

LOCAL_CONTEXT_WINDOW_DEFAULT = 262144


class CampaignPhase(str, Enum):
    INITIAL = "INITIAL"
    CAPTURE_AC = "CAPTURE_AC"
    GRADE_AC = "GRADE_AC"
    SYNC_TRACES = "SYNC_TRACES"
    INGEST_EXPORT_AC = "INGEST_EXPORT_AC"
    CERTIFY_B = "CERTIFY_B"
    B_CERTIFICATION_BLOCKED = "B_CERTIFICATION_BLOCKED"
    COLLECT_B = "COLLECT_B"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class PhaseStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class CampaignIsolationError(ValueError):
    """Raised when legacy Claude data or invalid protocol versions are detected."""


class CertificationGateBlockedError(RuntimeError):
    """Raised when Dataset B certification fails and stops continuation."""


class PreflightTaskCountError(RuntimeError):
    """Raised when the preflight task count differs from the required 21 tasks / 42 cells."""


class IncompleteCaptureError(RuntimeError):
    """Raised when any of the 42 cells lacks a terminal capture classification."""


class MissingGradeError(RuntimeError):
    """Raised when an official grade artifact is missing or invalid for a completed cell."""


class GradeMismatchError(RuntimeError):
    """Raised when metadata grade pass/fail status mismatches the grade report."""


class TraceJoinError(RuntimeError):
    """Raised when required edgeproxy traces cannot be joined for a captured session."""


class DuplicateSessionError(RuntimeError):
    """Raised when duplicate session IDs are detected across cells or trajectories."""


class TurnBudgetForbiddenError(ValueError):
    """Raised when any turn-limit or turn-count budget field is encountered."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scrub_secrets(text: str) -> str:
    """Scrub tokens, secrets, credentials, and passwords from strings without leaking secrets."""
    if not isinstance(text, str):
        text = str(text)
    for var in KNOWN_SECRETS_VARS:
        val = os.environ.get(var)
        if val and len(val) >= 4:
            text = text.replace(val, "[REDACTED]")
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", text)
    text = re.sub(r"sk-ant-[A-Za-z0-9_\-]+", "[REDACTED]", text)
    text = re.sub(r"(x-api-key['\":\s=]+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    return text


def verify_no_turn_limit_field(obj: Any, context: str = "record") -> None:
    """Recursively verify that no turn-count or turn-limit field exists in the data."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_TURN_BUDGET_KEYS:
                raise TurnBudgetForbiddenError(
                    f"Turn-limit forbidden: {context} contains turn budget field '{k}'"
                )
            verify_no_turn_limit_field(v, context=f"{context}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            verify_no_turn_limit_field(item, context=f"{context}[{i}]")
    elif is_dataclass(obj) and not isinstance(obj, type):
        verify_no_turn_limit_field(asdict(obj), context=context)


def _atomic_write_json(path: Path, value: Any) -> None:
    """Atomically write JSON with fsync and rename, preventing torn files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with temp_path.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


@dataclass
class CampaignPaths:
    """Explicit versioned output roots and script locations for the campaign."""
    campaign_root: Path
    w3_capture_root: Path
    pi_store_root: Path
    w4_b_root: Path
    checkpoint_file: Path
    w3_capture_script: Path
    w3_grade_script: Path
    w4_b_script: Path
    w4_cert_file: Path
    campaign_trace_root: Path
    remote_trace_root_cloud: str = "/workspace/flowmesh/traces/pi_dataset_ac_v1/cloud"
    remote_trace_root_edge: str = "/workspace/flowmesh/traces/pi_dataset_ac_v1/edge"
    remote_ssh_target: str = "gw@lum.id"
    remote_ssh_port: int = 31226

    @classmethod
    def from_root(cls, campaign_root: Path, base_nd: Path = ND) -> CampaignPaths:
        campaign_root = campaign_root.resolve()
        return cls(
            campaign_root=campaign_root,
            w3_capture_root=base_nd / "w3_capture/pi_dataset_ac_v1",
            pi_store_root=campaign_root / "dataset_ac",
            w4_b_root=campaign_root / "dataset_b",
            checkpoint_file=campaign_root / "campaign_checkpoint.json",
            w3_capture_script=base_nd / "w3_capture/run_pi_dataset_ac.py",
            w3_grade_script=base_nd / "w3_capture/grade_pi_dataset_ac.py",
            w4_b_script=base_nd / "w4_branch/run_pi_dataset_b.py",
            w4_cert_file=base_nd / "w4_branch/checkpoint_certificate.json",
            campaign_trace_root=campaign_root / "traces",
        )


@dataclass
class CampaignConfig:
    campaign_id: str = CAMPAIGN_ID_DEFAULT
    protocol_version_ac: str = PROTOCOL_PI_AC
    protocol_version_b: str = PROTOCOL_PI_B
    cohort: str = COHORT_DEFAULT
    paths: CampaignPaths = field(
        default_factory=lambda: CampaignPaths.from_root(ND / "campaigns/pi_permanent_v1")
    )
    edge_health_url: str = "http://127.0.0.1:18012/health"
    relay_tmux_session: str = "direct-gpu-relay-timeout-v2"
    driver_tmux_session: str = "pi-collection-campaign"


def create_initial_state(config: CampaignConfig) -> dict[str, Any]:
    """Create fresh initial state dictionary."""
    return {
        "schema_version": 1,
        "campaign_id": config.campaign_id,
        "protocol_version_ac": config.protocol_version_ac,
        "protocol_version_b": config.protocol_version_b,
        "cohort": config.cohort,
        "current_phase": CampaignPhase.INITIAL.value,
        "current_status": PhaseStatus.PENDING.value,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "phase_history": [],
        "checkpoints": {
            "capture_ac": {"status": PhaseStatus.PENDING.value},
            "grade_ac": {"status": PhaseStatus.PENDING.value},
            "sync_traces": {"status": PhaseStatus.PENDING.value},
            "ingest_export_ac": {"status": PhaseStatus.PENDING.value},
            "certify_b": {"status": PhaseStatus.PENDING.value},
            "collect_b": {"status": PhaseStatus.PENDING.value},
        },
        "paths": {
            "campaign_root": str(config.paths.campaign_root),
            "w3_capture_root": str(config.paths.w3_capture_root),
            "pi_store_root": str(config.paths.pi_store_root),
            "w4_b_root": str(config.paths.w4_b_root),
            "checkpoint_file": str(config.paths.checkpoint_file),
            "campaign_trace_root": str(config.paths.campaign_trace_root),
        },
    }


def load_or_init_state(config: CampaignConfig) -> dict[str, Any]:
    """Load existing checkpoint or initialize new one atomically."""
    checkpoint_path = config.paths.checkpoint_file
    if checkpoint_path.is_file():
        try:
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            logger.info("Resumed campaign checkpoint from %s at phase %s",
                        checkpoint_path, state.get("current_phase"))
            return state
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupted checkpoint at %s, creating backup: %s", checkpoint_path, exc)
            backup = checkpoint_path.with_name(f"campaign_checkpoint.corrupt.{int(time.time())}.bak")
            checkpoint_path.rename(backup)

    state = create_initial_state(config)
    _atomic_write_json(checkpoint_path, state)
    return state


def checkpoint_phase_transition(
    config: CampaignConfig,
    state: dict[str, Any],
    new_phase: CampaignPhase,
    status: PhaseStatus,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Atomically record a phase transition in campaign state."""
    old_phase = state.get("current_phase", CampaignPhase.INITIAL.value)
    now = _utc_now()
    history_entry = {
        "from_phase": old_phase,
        "to_phase": new_phase.value,
        "status": status.value,
        "transitioned_at_utc": now,
        "summary": summary or {},
        "error": error,
    }
    state["phase_history"].append(history_entry)
    state["current_phase"] = new_phase.value
    state["current_status"] = status.value
    state["updated_at_utc"] = now

    phase_key = new_phase.value.lower()
    if "checkpoints" not in state:
        state["checkpoints"] = {}
    state["checkpoints"][phase_key] = {
        "status": status.value,
        "updated_at_utc": now,
        "summary": summary or {},
        "error": error,
    }

    _atomic_write_json(config.paths.checkpoint_file, state)
    logger.info("Checkpoint transition [%s -> %s] status=%s", old_phase, new_phase.value, status.value)


# ---------------------------------------------------------------------------
# Isolation and Validation Helpers ("never mix legacy Claude data")
# ---------------------------------------------------------------------------

def verify_not_legacy_claude_data(
    record_or_meta: dict[str, Any],
    source_desc: str = "record",
) -> None:
    """Strictly verify that record or metadata does not contain legacy Claude data.

    Raises CampaignIsolationError if legacy Claude data is detected.
    """
    protocol = record_or_meta.get("protocol_version")
    if protocol in LEGACY_CLAUDE_PROTOCOLS:
        raise CampaignIsolationError(
            f"Isolation violation: {source_desc} contains legacy Claude protocol '{protocol}'."
        )

    # Reject runs or traces marked with claude harness
    harness = record_or_meta.get("harness") or (record_or_meta.get("provenance") or {}).get("harness")
    if harness in ("claude", "claude-code"):
        raise CampaignIsolationError(
            f"Isolation violation: {source_desc} specifies harness '{harness}', which is legacy Claude."
        )

    # Reject run dirs matching legacy Claude run patterns
    run_dir = str(record_or_meta.get("run_dir", ""))
    if "claude" in run_dir.lower() and "pi" not in run_dir.lower():
        raise CampaignIsolationError(
            f"Isolation violation: {source_desc} references legacy Claude run_dir '{run_dir}'."
        )


def verify_store_root_isolated(pi_store_root: Path, legacy_store_root: Path = ND) -> None:
    """Verify that the Pi store root is distinct from the legacy dataset root."""
    resolved_pi = pi_store_root.resolve()
    resolved_legacy = legacy_store_root.resolve()
    if resolved_pi == resolved_legacy:
        raise CampaignIsolationError(
            f"Isolation violation: Pi store root ({resolved_pi}) cannot be identical to "
            f"legacy dataset root ({resolved_legacy})."
        )
    # Ensure Pi store does not point into legacy datasets directory
    if resolved_pi == (resolved_legacy / "datasets").resolve():
        raise CampaignIsolationError(
            f"Isolation violation: Pi store root cannot be legacy datasets directory ({resolved_pi})."
        )


# ---------------------------------------------------------------------------
# Preflight 21 Tasks and 42 Cells Validation
# ---------------------------------------------------------------------------

def get_expected_tasks(source_files: tuple[Path, ...] = SOURCE_FILES_PREFLIGHT) -> list[str]:
    """Extract exactly the 21 unique preflight task instance IDs across source files."""
    seen: set[str] = set()
    ordered: list[str] = []
    for sf in source_files:
        if not sf.is_file():
            continue
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict) and "instance_id" in row:
                        iid = str(row["instance_id"])
                        if iid not in seen:
                            seen.add(iid)
                            ordered.append(iid)
        except Exception as exc:
            raise PreflightTaskCountError(f"Failed to read source file {sf}: {exc}") from exc
    if len(ordered) != EXPECTED_TASK_COUNT:
        raise PreflightTaskCountError(
            f"Expected exactly {EXPECTED_TASK_COUNT} unique preflight tasks, found {len(ordered)}"
        )
    return ordered


def get_expected_cell_ids(source_files: tuple[Path, ...] = SOURCE_FILES_PREFLIGHT) -> set[str]:
    """Derive exactly the 42 expected cell IDs (21 tasks x 2 backends)."""
    tasks = get_expected_tasks(source_files)
    cell_ids = {f"{t}__{backend}" for t in tasks for backend in BACKENDS_PREFLIGHT}
    if len(cell_ids) != EXPECTED_CELL_COUNT:
        raise PreflightTaskCountError(
            f"Expected exactly {EXPECTED_CELL_COUNT} cells, got {len(cell_ids)}"
        )
    return cell_ids


def validate_terminal_capture_classifications(
    w3_capture_root: Path,
    expected_cell_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate that all 42 expected cells have a terminal capture classification.

    Terminal classifications supported:
    completed, timeout, repeat, process, setup, protocol.
    Fails closed if any cell is missing, unreadable, pending, running, or legacy Claude.
    """
    if expected_cell_ids is None:
        expected_cell_ids = get_expected_cell_ids()

    if len(expected_cell_ids) != EXPECTED_CELL_COUNT:
        raise PreflightTaskCountError(
            f"Expected exactly {EXPECTED_CELL_COUNT} cells, got {len(expected_cell_ids)}"
        )

    cell_statuses: dict[str, dict[str, Any]] = {}
    incomplete: list[dict[str, str]] = []

    for cell_id in sorted(expected_cell_ids):
        cell_dir = w3_capture_root / cell_id
        meta_file = cell_dir / "run_meta.json"
        if not cell_dir.is_dir():
            incomplete.append({"cell_id": cell_id, "reason": "directory_missing"})
            continue
        if not meta_file.is_file():
            incomplete.append({"cell_id": cell_id, "reason": "meta_missing"})
            continue

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as exc:
            incomplete.append({"cell_id": cell_id, "reason": f"unreadable_meta: {exc}"})
            continue

        verify_not_legacy_claude_data(meta, source_desc=f"cell {cell_id}")
        verify_no_turn_limit_field(meta, context=f"cell {cell_id}")

        classification = meta.get("classification")
        status = meta.get("status")

        if classification not in TERMINAL_CAPTURE_CLASSIFICATIONS or status in ("pending", "running"):
            incomplete.append({
                "cell_id": cell_id,
                "reason": f"non_terminal (status={status}, classification={classification})",
            })
            continue

        cell_statuses[cell_id] = {
            "status": status,
            "classification": classification,
            "session_id": meta.get("session_id"),
        }

    if incomplete:
        raise IncompleteCaptureError(
            f"Terminal capture validation failed: {len(incomplete)}/{len(expected_cell_ids)} cells "
            f"lack terminal classification: {incomplete}"
        )

    return {
        "total_cells": len(expected_cell_ids),
        "terminal_cells": len(cell_statuses),
        "cell_statuses": cell_statuses,
    }


def validate_grading_status(
    w3_capture_root: Path,
    expected_cell_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate that official SWE-Bench-Fork grades are recorded for all completed cells.

    Fails closed if any completed cell is missing its grade status or report.
    """
    if expected_cell_ids is None:
        expected_cell_ids = get_expected_cell_ids()

    graded_cells: list[str] = []
    missing_grades: list[str] = []

    for cell_id in sorted(expected_cell_ids):
        cell_dir = w3_capture_root / cell_id
        meta_file = cell_dir / "run_meta.json"
        if not meta_file.is_file():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("status") != "completed":
            continue

        grade_status = meta.get("official_grade_status")
        official_pass = meta.get("official_pass")
        if grade_status != "graded" or not isinstance(official_pass, bool):
            missing_grades.append(cell_id)
            continue

        # Check grade report artifact existence
        report = _load_official_grade_report(cell_dir, w3_capture_root)
        if report is None:
            missing_grades.append(f"{cell_id} (missing report file)")
            continue

        graded_cells.append(cell_id)

    if missing_grades:
        raise MissingGradeError(
            f"Grading validation failed: {len(missing_grades)} completed cells lack valid official grades: {missing_grades}"
        )

    return {
        "completed_and_graded_count": len(graded_cells),
        "graded_cells": graded_cells,
    }


def _load_official_grade_report(cell_dir: Path, w3_capture_root: Path) -> dict[str, Any] | None:
    """Locate and load the official grade report artifact for a cell."""
    meta_file = cell_dir / "run_meta.json"
    if not meta_file.is_file():
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Check grader reference in meta
    report_rel = meta.get("official_grader_reference") or (meta.get("official_grade_details") or {}).get("report_path")
    if report_rel:
        p = (w3_capture_root / report_rel).resolve()
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass

    # Check attempt directory pattern
    attempts = meta.get("attempts", [])
    if attempts:
        latest = attempts[-1]
        att_dir_name = latest.get("attempt_dir", "")
        att_dir = cell_dir / att_dir_name
        if att_dir.is_dir():
            for gfile in att_dir.glob("official_grade_*.json"):
                try:
                    return json.loads(gfile.read_text(encoding="utf-8"))
                except Exception:
                    pass

    # Check sibling results file
    sibling = w3_capture_root / f"grading_results_{cell_dir.name}.json"
    if sibling.is_file():
        try:
            return json.loads(sibling.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Check cell_dir/grading_result.json
    local_g = cell_dir / "grading_result.json"
    if local_g.is_file():
        try:
            return json.loads(local_g.read_text(encoding="utf-8"))
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Remote Trace Synchronization
# ---------------------------------------------------------------------------

def sync_remote_traces(
    config: CampaignConfig,
    *,
    sync_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Synchronize isolated remote trace roots into the versioned local campaign trace root.

    Remote paths:
      /workspace/flowmesh/traces/pi_dataset_ac_v1/cloud
      /workspace/flowmesh/traces/pi_dataset_ac_v1/edge
    Connection:
      ssh -p 31226 gw@lum.id
    Destination:
      config.paths.campaign_trace_root

    Scrubs all secrets from commands, logs, and error messages.
    """
    trace_root = config.paths.campaign_trace_root
    trace_root.mkdir(parents=True, exist_ok=True)
    cloud_dir = trace_root / "cloud"
    edge_dir = trace_root / "edge"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)

    remote_target = config.paths.remote_ssh_target
    remote_port = config.paths.remote_ssh_port
    remote_cloud = config.paths.remote_trace_root_cloud
    remote_edge = config.paths.remote_trace_root_edge

    cloud_cmd = [
        "rsync",
        "-avz",
        "-e", f"ssh -p {remote_port} -o StrictHostKeyChecking=accept-new -o BatchMode=yes",
        f"{remote_target}:{remote_cloud}/",
        f"{cloud_dir}/",
    ]
    edge_cmd = [
        "rsync",
        "-avz",
        "-e", f"ssh -p {remote_port} -o StrictHostKeyChecking=accept-new -o BatchMode=yes",
        f"{remote_target}:{remote_edge}/",
        f"{edge_dir}/",
    ]

    try:
        if sync_runner is not None:
            sync_runner(cloud_cmd)
            sync_runner(edge_cmd)
        else:
            # Check if local trace files already exist (e.g. offline testing/resumption)
            local_cloud_traces = list(cloud_dir.glob("*.jsonl"))
            local_edge_traces = list(edge_dir.glob("*.jsonl"))
            if local_cloud_traces or local_edge_traces:
                logger.info("Local traces pre-exist in %s; synchronizing updates.", trace_root)
            subprocess.run(cloud_cmd, capture_output=True, text=True, check=True)
            subprocess.run(edge_cmd, capture_output=True, text=True, check=True)

        cloud_files = [p.name for p in cloud_dir.glob("*.jsonl")]
        edge_files = [p.name for p in edge_dir.glob("*.jsonl")]
        summary = {
            "local_trace_root": str(trace_root),
            "cloud_files_count": len(cloud_files),
            "edge_files_count": len(edge_files),
            "cloud_files": cloud_files,
            "edge_files": edge_files,
        }
        logger.info("Synchronized remote traces into %s (cloud=%d, edge=%d)",
                    trace_root, len(cloud_files), len(edge_files))
        return summary
    except Exception as exc:
        safe_msg = scrub_secrets(str(exc))
        logger.error("Trace synchronization failed: %s", safe_msg)
        raise RuntimeError(f"Trace synchronization failed: {safe_msg}") from None


def run_sync_traces_phase(
    config: CampaignConfig,
    state: dict[str, Any],
    *,
    sync_fn: Callable[[CampaignConfig], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute remote trace synchronization phase."""
    checkpoint_phase_transition(config, state, CampaignPhase.SYNC_TRACES, PhaseStatus.RUNNING)
    try:
        if sync_fn is not None:
            summary = sync_fn(config)
        else:
            summary = sync_remote_traces(config)
        checkpoint_phase_transition(
            config, state, CampaignPhase.SYNC_TRACES, PhaseStatus.COMPLETED, summary
        )
        return summary
    except Exception as exc:
        err = f"Remote trace synchronization phase failed: {scrub_secrets(str(exc))}"
        checkpoint_phase_transition(config, state, CampaignPhase.FAILED, PhaseStatus.FAILED, error=err)
        raise RuntimeError(err) from exc


# ---------------------------------------------------------------------------
# Phase 1: W3 Pi A/C Capture
# ---------------------------------------------------------------------------

def run_capture_ac_phase(
    config: CampaignConfig,
    state: dict[str, Any],
    *,
    runner_fn: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    expected_cell_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Invoke the W3 pi_dataset_ac_v1 capture runner and validate 42 terminal cells."""
    checkpoint_phase_transition(config, state, CampaignPhase.CAPTURE_AC, PhaseStatus.RUNNING)
    w3_script = config.paths.w3_capture_script
    w3_out = config.paths.w3_capture_root
    w3_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(w3_script),
        "--out-root", str(w3_out),
        "--protocol-version", config.protocol_version_ac,
        "--campaign-id", config.campaign_id,
    ]

    try:
        if runner_fn is not None:
            proc = runner_fn(cmd)
        else:
            if not w3_script.is_file():
                raise FileNotFoundError(f"W3 capture script does not exist: {w3_script}")
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Require a terminal capture classification for all 42 before grading
        cell_validation = validate_terminal_capture_classifications(w3_out, expected_cell_ids)
        summary = {
            "output_root": str(w3_out),
            "total_cells": cell_validation["total_cells"],
            "terminal_cells": cell_validation["terminal_cells"],
            "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
        }
        checkpoint_phase_transition(config, state, CampaignPhase.CAPTURE_AC, PhaseStatus.COMPLETED, summary)
        return summary
    except Exception as exc:
        err = f"W3 capture phase failed: {exc}"
        checkpoint_phase_transition(config, state, CampaignPhase.FAILED, PhaseStatus.FAILED, error=err)
        raise RuntimeError(err) from exc


# ---------------------------------------------------------------------------
# Phase 2: Official Grading
# ---------------------------------------------------------------------------

def run_grade_ac_phase(
    config: CampaignConfig,
    state: dict[str, Any],
    *,
    grader_fn: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    expected_cell_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Invoke official SWE-Bench-Fork grading on completed Pi capture cells.

    Validates that all 42 cells have terminal capture classifications before grading,
    and requires all completed cells to have official grades after grading.
    """
    # 1. Require all 42 cells have terminal capture classifications before grading starts
    w3_out = config.paths.w3_capture_root
    validate_terminal_capture_classifications(w3_out, expected_cell_ids)

    checkpoint_phase_transition(config, state, CampaignPhase.GRADE_AC, PhaseStatus.RUNNING)
    grade_script = config.paths.w3_grade_script
    w3_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(grade_script),
        "--capture-root", str(w3_out),
    ]

    try:
        if grader_fn is not None:
            proc = grader_fn(cmd)
        else:
            if not grade_script.is_file():
                raise FileNotFoundError(f"W3 grade script does not exist: {grade_script}")
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # 2. Validate official grades for all completed cells
        grading_summary = validate_grading_status(w3_out, expected_cell_ids)
        summary = {
            "capture_root": str(w3_out),
            "completed_and_graded_count": grading_summary["completed_and_graded_count"],
            "graded_cells": grading_summary["graded_cells"],
            "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
        }
        checkpoint_phase_transition(config, state, CampaignPhase.GRADE_AC, PhaseStatus.COMPLETED, summary)
        return summary
    except Exception as exc:
        err = f"Official grading phase failed: {exc}"
        checkpoint_phase_transition(config, state, CampaignPhase.FAILED, PhaseStatus.FAILED, error=err)
        raise RuntimeError(err) from exc


# ---------------------------------------------------------------------------
# Phase 3: Isolated A/C Ingestion and Export
# ---------------------------------------------------------------------------

def _extract_session_id_from_trace_record(rec: dict[str, Any]) -> str | None:
    """Extract session_id from trace record headers, call, or top-level."""
    call = rec.get("call") or {}
    headers = rec.get("headers") or {}
    sid = (
        call.get("session_id")
        or headers.get("x-claude-code-session-id")
        or rec.get("session_id")
    )
    return str(sid) if sid else None


def _load_synced_traces(trace_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Read all synced trace files and group records by session_id."""
    traces_by_session: dict[str, list[dict[str, Any]]] = {}
    trace_files = sorted(trace_root.rglob("*.jsonl"))
    if not trace_files:
        raise TraceJoinError(f"No edgeproxy trace files found in campaign trace root: {trace_root}")

    for trace_file in trace_files:
        with trace_file.open(encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                verify_not_legacy_claude_data(rec, source_desc=f"{trace_file.name}:{line_num}")
                verify_no_turn_limit_field(rec, context=f"{trace_file.name}:{line_num}")

                sid = _extract_session_id_from_trace_record(rec)
                if sid:
                    # Include valid message or structured calls
                    if rec.get("path") == "/v1/messages" or rec.get("call") or (rec.get("request") and rec.get("response")):
                        traces_by_session.setdefault(sid, []).append(rec)

    for sid in traces_by_session:
        traces_by_session[sid].sort(key=lambda r: r.get("ts") or r.get("timestamp") or 0)

    return traces_by_session


def run_ingest_export_ac_phase(
    config: CampaignConfig,
    state: dict[str, Any],
    *,
    custom_ingester: Callable[[CampaignConfig], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ingest graded Pi cells into isolated Pi JobStore and export datasets A and C."""
    checkpoint_phase_transition(config, state, CampaignPhase.INGEST_EXPORT_AC, PhaseStatus.RUNNING)
    verify_store_root_isolated(config.paths.pi_store_root)

    try:
        if custom_ingester is not None:
            summary = custom_ingester(config)
        else:
            summary = _default_ingest_and_export(config)

        checkpoint_phase_transition(
            config, state, CampaignPhase.INGEST_EXPORT_AC, PhaseStatus.COMPLETED, summary
        )
        return summary
    except Exception as exc:
        err = f"A/C ingestion and export phase failed: {exc}"
        checkpoint_phase_transition(config, state, CampaignPhase.FAILED, PhaseStatus.FAILED, error=err)
        raise RuntimeError(err) from exc


def _default_ingest_and_export(config: CampaignConfig) -> dict[str, Any]:
    """Evidence-faithful ingestion and export implementation using JobStore.

    Requirements:
    - No synthetic Dataset A/C records: ingests only real Pi metadata plus real
      edgeproxy request/response traces joined by session_id and official grade artifacts.
    - Fails closed on absent/mismatched traces, duplicate sessions, or missing grades.
    - No turn-count budget anywhere; 1800-second trajectory budget comes from W3 metadata.
    """
    store_root = config.paths.pi_store_root
    store_root.mkdir(parents=True, exist_ok=True)

    # Lazy imports from w1_storage
    sys.path.insert(0, str(ND))
    try:
        from w1_storage.store import JobStore
        from w1_storage.schemas import PrefixOutcome, ServingCall
        from w1_storage.features import extract_predecision_features, FEATURES_VERSION
        from w1_storage.prefix_diff import compute_prefix_diff, session_cache_progress
    except ImportError as exc:
        raise RuntimeError(f"Failed to import w1_storage components: {exc}") from exc

    store = JobStore(store_root)
    w3_out = config.paths.w3_capture_root
    trace_root = config.paths.campaign_trace_root

    # Load synced traces indexed by session_id
    traces_by_session = _load_synced_traces(trace_root)

    total_a = 0
    total_c = 0
    seen_sessions: dict[str, str] = {}

    cell_dirs = sorted(p for p in w3_out.iterdir() if p.is_dir() and (p / "run_meta.json").is_file())
    if not cell_dirs:
        raise IncompleteCaptureError(f"No cell directories with run_meta.json found in {w3_out}")

    # Check for duplicate session IDs across cells first (fail closed)
    for cell_dir in cell_dirs:
        meta_path = cell_dir / "run_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        verify_not_legacy_claude_data(meta, source_desc=f"cell {cell_dir.name}")
        verify_no_turn_limit_field(meta, context=f"cell {cell_dir.name}")

        sid = meta.get("session_id")
        if not sid:
            raise IncompleteCaptureError(f"Cell {cell_dir.name} has missing session_id")
        if sid in seen_sessions:
            raise DuplicateSessionError(
                f"Duplicate session_id '{sid}' detected across cells: {cell_dir.name} matches {seen_sessions[sid]}"
            )
        seen_sessions[sid] = cell_dir.name

    for cell_dir in cell_dirs:
        meta_path = cell_dir / "run_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sid = meta["session_id"]
        iid = meta.get("instance_id", cell_dir.name.split("__")[0])
        policy = meta.get("backend", meta.get("policy", "edge"))
        task_id = f"gym:{iid}"
        trajectory_id = f"{task_id}:{policy}:{sid}"
        backend_fp = "local:Inferact/Qwen3.8-27B-NVFP4" if "edge" in policy else "cloud:deepseek-v4-flash"

        # 1. Join edgeproxy traces (fail closed if absent)
        recs = traces_by_session.get(sid, [])
        if not recs:
            raise TraceJoinError(
                f"Missing trace join: no edgeproxy trace records found for session_id '{sid}' in cell {cell_dir.name}"
            )

        # 2. Join official grade artifact (fail closed if missing/unparseable)
        grade_info = _load_official_grade_report(cell_dir, w3_out)
        if grade_info is None or meta.get("official_grade_status") != "graded":
            raise MissingGradeError(
                f"Missing official grade artifact for completed cell {cell_dir.name} (status={meta.get('official_grade_status')})"
            )

        report_resolved = grade_info.get("resolved")
        meta_pass = meta.get("official_pass")
        if not isinstance(report_resolved, bool) and not isinstance(meta_pass, bool):
            raise MissingGradeError(
                f"Official grade artifact for {cell_dir.name} lacks valid boolean resolved status"
            )
        resolved = report_resolved if isinstance(report_resolved, bool) else meta_pass
        if isinstance(report_resolved, bool) and meta_pass is not None and report_resolved != meta_pass:
            raise GradeMismatchError(
                f"Grade mismatch in {cell_dir.name}: report resolved={report_resolved} but meta official_pass={meta_pass}"
            )

        grade_report_payload = grade_info.get("report", grade_info)
        verify_not_legacy_claude_data(grade_report_payload, source_desc=f"grade report for {cell_dir.name}")
        verify_no_turn_limit_field(grade_report_payload, context=f"grade report for {cell_dir.name}")
        grade_art = store.put_artifact(json.dumps(grade_report_payload, sort_keys=True).encode("utf-8"))
        grade_report_ref = grade_art["ref"]

        # 3. 1800-second trajectory budget comes from W3 metadata (no turn-count budget anywhere!)
        trajectory_budget_s = float(meta.get("max_active_seconds", 1800))
        remaining_budget = {
            "trajectory_budget_seconds": trajectory_budget_s,
            "max_active_seconds": trajectory_budget_s,
        }
        verify_no_turn_limit_field(remaining_budget, context=f"remaining_budget for {cell_dir.name}")

        # 4. Ingest real request/response records for each call
        n = len(recs)
        session_cache_baseline = None
        for i, rec in enumerate(recs):
            req_body = rec.get("request")
            resp_body = rec.get("response")
            if not req_body or not resp_body:
                raise TraceJoinError(
                    f"Trace record {i} for session {sid} in {cell_dir.name} lacks real request or response payload"
                )

            verify_not_legacy_claude_data(req_body, source_desc=f"request {i} in {cell_dir.name}")
            verify_not_legacy_claude_data(resp_body, source_desc=f"response {i} in {cell_dir.name}")
            verify_no_turn_limit_field(req_body, context=f"request {i} in {cell_dir.name}")
            verify_no_turn_limit_field(resp_body, context=f"response {i} in {cell_dir.name}")

            req_art = store.put_artifact(json.dumps(req_body, sort_keys=True).encode("utf-8"))
            resp_art = store.put_artifact(json.dumps(resp_body, sort_keys=True).encode("utf-8"))

            prefix_id = f"{trajectory_id}:call{i}"
            is_last = (i == n - 1)

            # Extract predecision features from request history
            messages = req_body.get("messages", [])
            has_prior = (i > 0)
            prior_stop = None
            if has_prior:
                prior_resp = recs[i - 1].get("response") or {}
                prior_stop = prior_resp.get("stop_reason") if isinstance(prior_resp, dict) else None
            features = extract_predecision_features(
                messages, req_body, LOCAL_CONTEXT_WINDOW_DEFAULT, has_prior, prior_stop
            )
            prior_request = recs[i - 1].get("request") if has_prior else None
            features["prefix_diff"] = compute_prefix_diff(prior_request, req_body)

            # Session prefix cache
            local_resources = rec.get("local_resources") or {}
            vllm_snapshot = local_resources.get("vllm")
            session_cache = local_resources.get("session_prefix_cache")
            if session_cache is None and vllm_snapshot:
                if session_cache_baseline is None:
                    session_cache_baseline = vllm_snapshot
                session_cache = session_cache_progress(session_cache_baseline, vllm_snapshot)
            if session_cache is not None:
                features["session_prefix_cache"] = session_cache

            # Ingest PrefixOutcome (Dataset A)
            a_record = PrefixOutcome(
                schema_version=1,
                protocol_version=config.protocol_version_ac,
                campaign_id=config.campaign_id,
                task_id=task_id,
                trajectory_id=trajectory_id,
                source="pi-live-capture",
                cohort=config.cohort,
                split="train",
                provenance={
                    "instance_id": iid,
                    "policy": policy,
                    "session_id": sid,
                    "run_dir": cell_dir.name,
                },
                missing_reasons={
                    "tool_schema_ref": "not_extracted_this_protocol",
                    "tool_schema_hash": "not_extracted_this_protocol",
                },
                prefix_id=prefix_id,
                source_policy=policy,
                backend_fingerprint_id=backend_fp,
                call_index=i,
                predecision_request_ref=req_art["ref"],
                predecision_request_hash=req_art["sha256"],
                tool_schema_ref=None,
                tool_schema_hash=None,
                history_integrity="complete_from_proxy_boundary",
                predecision_features=features,
                features_version=FEATURES_VERSION,
                remaining_budget_at_prefix=remaining_budget,
                core_sample_selected=True,
                selection_probability=1.0,
                terminal_grade_ref=grade_report_ref,
                resolved=resolved,
                label_valid=True,
                termination_reason="official_grade_joined" if is_last else "trajectory_continues",
                outcome_censored=False,
                sampling_metadata={"trajectory_length": n, "is_last_prefix": is_last},
            )
            a_record.validate()
            verify_no_turn_limit_field(a_record, context=f"a_record {prefix_id}")
            store.append_record("A", a_record)
            total_a += 1

            # Ingest ServingCall (Dataset C)
            call_info = rec.get("call") or {}
            invocation_id = call_info.get("call_id") or f"{prefix_id}:invocation{i}"
            logical_call_id = call_info.get("call_id") or f"{prefix_id}:logical{i}"
            c_record = ServingCall(
                schema_version=1,
                protocol_version=config.protocol_version_ac,
                campaign_id=config.campaign_id,
                task_id=task_id,
                trajectory_id=trajectory_id,
                source="pi-live-capture",
                cohort=config.cohort,
                split="train",
                provenance={
                    "instance_id": iid,
                    "policy": policy,
                    "session_id": sid,
                },
                invocation_id=invocation_id,
                logical_call_id=logical_call_id,
                prefix_id=prefix_id,
                branch_id=None,
                preference_pair_id=None,
                purpose="original",
                invocation_kind="original",
                backend_fingerprint_id=backend_fp,
                logical_request_hash=req_art["sha256"],
                rendered_request_hash=req_art["sha256"],
                attempt_index=0,
                pre_dispatch_snapshot={"local_resources": rec.get("local_resources")} if rec.get("local_resources") else {},
                snapshot_timestamp=str(rec.get("local_resources", {}).get("sampled_at")) if rec.get("local_resources") else None,
                snapshot_age_ms=None,
                snapshot_source="edgeproxy_local_resources" if rec.get("local_resources") else None,
                request_start=str(rec.get("ts") or rec.get("timestamp") or _utc_now()),
                first_byte=None,
                first_content=None,
                end=None,
                measured_timings=rec.get("timing") or {},
                raw_usage=rec.get("usage") or {},
                normalised_usage={
                    k: v
                    for k, v in (call_info.get("tokens") or {}).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                },
                usage_integrity=(call_info.get("tokens") or {}).get("usage_integrity", "unknown"),
                status=str(rec.get("status", "200")),
                response_ref=resp_art["ref"],
                error_ref=None,
                cost_basis=None,
                rate_version=None,
                observed_or_estimated_cost=None,
                latency_censored=False,
                measurement_quality_flags=["pi_live_edgeproxy_capture"],
                missing_reasons={
                    "snapshot_age_ms": "not_computed",
                    "cost_basis": "not_priced",
                    "rate_version": "not_priced",
                    "observed_or_estimated_cost": "not_priced",
                    "first_byte": "not_recorded_this_protocol",
                    "first_content": "not_recorded_this_protocol",
                    "end": "not_recorded_this_protocol",
                    "error_ref": "no_transport_error",
                    "branch_id": "not_a_branch_invocation",
                    "preference_pair_id": "not_a_preference_invocation",
                    **({} if rec.get("local_resources") else {
                        "snapshot_timestamp": "local_resources_absent",
                        "snapshot_source": "local_resources_absent",
                    }),
                },
            )
            c_record.validate()
            verify_no_turn_limit_field(c_record, context=f"c_record {invocation_id}")
            store.append_record("C", c_record)
            total_c += 1

    exported_paths = store.finalize_export()
    logger.info("Ingestion completed: A=%d, C=%d, exported=%s",
                total_a, total_c, [str(p) for p in exported_paths])
    return {
        "store_root": str(store_root),
        "total_a_ingested": total_a,
        "total_c_ingested": total_c,
        "exported_files": [str(p) for p in exported_paths],
    }


# ---------------------------------------------------------------------------
# Phase 4 & 5: Certification Gate and W4 Dataset B Collector
# ---------------------------------------------------------------------------

def evaluate_b_certification_gate(
    cert_data: dict[str, Any] | None,
    cert_path: Path | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate whether Dataset B prospective certification gate permits continuation.

    Returns:
        (is_permitted, blockers_list)
    """
    if cert_data is None:
        if cert_path and cert_path.is_file():
            try:
                cert_data = json.loads(cert_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                return False, [f"Unreadable certificate at {cert_path}: {exc}"]
        else:
            return False, ["No checkpoint certificate found for Dataset B."]

    verify_not_legacy_claude_data(cert_data, source_desc="Dataset B certificate")
    verify_no_turn_limit_field(cert_data, context="Dataset B certificate")

    status = str(cert_data.get("status", "")).upper()
    blockers = list(cert_data.get("blockers", []))

    blocker_reason = cert_data.get("blocker_reason")
    if blocker_reason and blocker_reason not in blockers:
        blockers.append(str(blocker_reason))

    if cert_data.get("quarantine") is True and "Certificate is marked quarantined" not in blockers:
        blockers.append("Certificate is marked quarantined.")

    if status in ("CERTIFIED", "PASS", "PASSED") and not blockers:
        return True, []

    if not blockers and status not in ("CERTIFIED", "PASS", "PASSED"):
        blockers.append(f"Certificate status is '{status}', not CERTIFIED/PASS.")

    return False, blockers


def run_certify_b_phase(
    config: CampaignConfig,
    state: dict[str, Any],
    *,
    cert_data: dict[str, Any] | None = None,
    certifier_fn: Callable[[], tuple[bool, list[str]]] | None = None,
) -> tuple[bool, list[str]]:
    """Check certification gate before permitting prospective Dataset B collection."""
    checkpoint_phase_transition(config, state, CampaignPhase.CERTIFY_B, PhaseStatus.RUNNING)

    if certifier_fn is not None:
        permitted, blockers = certifier_fn()
    else:
        permitted, blockers = evaluate_b_certification_gate(cert_data, config.paths.w4_cert_file)

    summary = {
        "permitted": permitted,
        "blockers": blockers,
        "certificate_path": str(config.paths.w4_cert_file),
    }

    if not permitted:
        logger.warning("Dataset B certification gate BLOCKED: %s", blockers)
        checkpoint_phase_transition(
            config, state, CampaignPhase.B_CERTIFICATION_BLOCKED, PhaseStatus.BLOCKED, summary
        )
        return False, blockers

    logger.info("Dataset B certification gate PERMITTED.")
    checkpoint_phase_transition(
        config, state, CampaignPhase.CERTIFY_B, PhaseStatus.COMPLETED, summary
    )
    return True, []


def run_collect_b_phase(
    config: CampaignConfig,
    state: dict[str, Any],
    *,
    collector_fn: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Invoke W4 Pi Dataset B collector ONLY when certification gate permits."""
    checkpoint_phase_transition(config, state, CampaignPhase.COLLECT_B, PhaseStatus.RUNNING)
    w4_script = config.paths.w4_b_script
    w4_out = config.paths.w4_b_root
    w4_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(w4_script),
        "--out-root", str(w4_out),
        "--protocol-version", config.protocol_version_b,
        "--campaign-id", config.campaign_id,
        "--ac-capture-root", str(config.paths.w3_capture_root),
    ]

    try:
        if collector_fn is not None:
            proc = collector_fn(cmd)
        else:
            if not w4_script.is_file():
                raise FileNotFoundError(f"W4 B collector script does not exist: {w4_script}")
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

        summary = {
            "output_root": str(w4_out),
            "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
        }
        checkpoint_phase_transition(config, state, CampaignPhase.COLLECT_B, PhaseStatus.COMPLETED, summary)
        checkpoint_phase_transition(config, state, CampaignPhase.COMPLETE, PhaseStatus.COMPLETED)
        return summary
    except Exception as exc:
        err = f"Dataset B collection phase failed: {exc}"
        checkpoint_phase_transition(config, state, CampaignPhase.FAILED, PhaseStatus.FAILED, error=err)
        raise RuntimeError(err) from exc


# ---------------------------------------------------------------------------
# Campaign Orchestrator Main Loop
# ---------------------------------------------------------------------------

def run_campaign(
    config: CampaignConfig,
    *,
    resume: bool = True,
    target_phase: CampaignPhase | None = None,
    dry_run: bool = False,
    runner_fn: Any = None,
    grader_fn: Any = None,
    sync_fn: Any = None,
    ingest_fn: Any = None,
    certifier_fn: Any = None,
    collector_fn: Any = None,
) -> dict[str, Any]:
    """Run or resume the permanent Pi collection campaign.

    Steps sequentially:
    CAPTURE_AC -> GRADE_AC -> SYNC_TRACES -> INGEST_EXPORT_AC -> CERTIFY_B -> COLLECT_B -> COMPLETE
    Halts safely at B_CERTIFICATION_BLOCKED if certification gate fails.
    """
    state = load_or_init_state(config)
    curr_phase = CampaignPhase(state.get("current_phase", CampaignPhase.INITIAL.value))

    if dry_run:
        logger.info("DRY-RUN: Campaign state loaded. Current phase=%s, Target=%s", curr_phase, target_phase)
        return state

    # Step 1: Capture AC
    if curr_phase in (CampaignPhase.INITIAL, CampaignPhase.CAPTURE_AC):
        run_capture_ac_phase(config, state, runner_fn=runner_fn)
        curr_phase = CampaignPhase.GRADE_AC
        if target_phase == CampaignPhase.CAPTURE_AC:
            return state

    # Step 2: Grade AC
    if curr_phase == CampaignPhase.GRADE_AC:
        run_grade_ac_phase(config, state, grader_fn=grader_fn)
        curr_phase = CampaignPhase.SYNC_TRACES
        if target_phase == CampaignPhase.GRADE_AC:
            return state

    # Step 3: Sync Traces
    if curr_phase == CampaignPhase.SYNC_TRACES:
        run_sync_traces_phase(config, state, sync_fn=sync_fn)
        curr_phase = CampaignPhase.INGEST_EXPORT_AC
        if target_phase == CampaignPhase.SYNC_TRACES:
            return state

    # Step 4: Ingest & Export AC
    if curr_phase == CampaignPhase.INGEST_EXPORT_AC:
        run_ingest_export_ac_phase(config, state, custom_ingester=ingest_fn)
        curr_phase = CampaignPhase.CERTIFY_B
        if target_phase == CampaignPhase.INGEST_EXPORT_AC:
            return state

    # Step 5: Certify B Gate
    if curr_phase == CampaignPhase.CERTIFY_B:
        permitted, blockers = run_certify_b_phase(config, state, certifier_fn=certifier_fn)
        if not permitted:
            logger.warning("Halting campaign before Dataset B: Certification gate blocked (%s).", blockers)
            return state
        curr_phase = CampaignPhase.COLLECT_B
        if target_phase == CampaignPhase.CERTIFY_B:
            return state

    # Step 6: Collect Dataset B (only reached if permitted)
    if curr_phase == CampaignPhase.COLLECT_B:
        run_collect_b_phase(config, state, collector_fn=collector_fn)
        curr_phase = CampaignPhase.COMPLETE

    return state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Permanent Pi Coder Campaign Orchestrator")
    parser.add_argument("--campaign-id", default=CAMPAIGN_ID_DEFAULT, help="Campaign identifier")
    parser.add_argument("--campaign-root", type=Path, default=ND / "campaigns/pi_permanent_v1",
                        help="Root directory for campaign checkpoints and artifacts")
    parser.add_argument("--status", action="store_true", help="Print current campaign status and exit")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and checkpoint without executing")
    parser.add_argument("--target-phase", choices=[p.value for p in CampaignPhase], default=None,
                        help="Run only up to and including the specified phase")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    paths = CampaignPaths.from_root(args.campaign_root)
    config = CampaignConfig(
        campaign_id=args.campaign_id,
        paths=paths,
    )

    if args.status:
        state = load_or_init_state(config)
        print(json.dumps(state, indent=2))
        return 0

    target_phase = CampaignPhase(args.target_phase) if args.target_phase else None
    state = run_campaign(config, target_phase=target_phase, dry_run=args.dry_run)
    print(json.dumps({
        "campaign_id": config.campaign_id,
        "phase": state.get("current_phase"),
        "status": state.get("current_status"),
        "checkpoint_file": str(config.paths.checkpoint_file),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
