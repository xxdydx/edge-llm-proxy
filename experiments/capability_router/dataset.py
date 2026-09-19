"""Select the historical calls this experiment replays.

Source: completed SWE-bench Pro campaigns under
``eval-suite/swebench/results/*/verdicts/*__cloud__*.json`` whose task
verdict passed (real ``fail_to_pass``/``pass_to_pass`` checks, not a
self-report). For each passing cloud-only job, pull the matching real
request/response call records out of ``traces/`` and use the recorded cloud
response as the teacher action for every call in that trajectory.

Grouped by SWE-bench task instance (never by individual call) so any
train/test split downstream cannot leak the same task's calls across the
split -- a model that memorizes one task's request shapes would otherwise
look like it generalizes.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from edgeproxy.trace import record as trace_record
from edgeproxy.trace.replay import calls as messages_calls

from .config import EXCLUDED_RESULT_DIRS, REPO_ROOT, ExperimentConfig
from .schema import DatasetManifest, TeacherCall

RESULTS_ROOT = REPO_ROOT / "eval-suite" / "swebench" / "results"
TRACES_ROOT = REPO_ROOT / "traces"


def discover_passing_cloud_verdicts(
    results_root: Path = RESULTS_ROOT,
    excluded_dirs: frozenset[str] = EXCLUDED_RESULT_DIRS,
) -> list[dict[str, Any]]:
    """One entry per passing cloud-only job: campaign, task_group, verdict path."""
    out: list[dict[str, Any]] = []
    if not results_root.is_dir():
        return out
    for campaign_dir in sorted(results_root.iterdir()):
        if not campaign_dir.is_dir() or campaign_dir.name in excluded_dirs:
            continue
        verdicts_dir = campaign_dir / "verdicts"
        if not verdicts_dir.is_dir():
            continue
        for verdict_path in sorted(verdicts_dir.glob("*__cloud__*.json")):
            try:
                verdict = json.loads(verdict_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not verdict.get("passed"):
                continue
            base = verdict_path.stem  # "<task_group>__cloud__seedN"
            task_group = base.split("__cloud__")[0]
            out.append(
                {
                    "campaign": campaign_dir.name,
                    "task_group": task_group,
                    "verdict_path": str(verdict_path),
                    "verdict": verdict,
                }
            )
    return out


def locate_trace_file(campaign: str, task_group: str) -> Path | None:
    """Find the real trace file for one (campaign, task_group) cloud job.

    Trace directories are named
    ``<suite>-run-cloud-<run_stamp>-<task_group>-seed<n>``; the run stamp is
    not derivable from the campaign name, so candidates are matched by
    pattern and disambiguated by reading each candidate's own
    ``experiment_id`` field -- the one field guaranteed to equal the
    campaign name that produced it.
    """
    suffix = task_group.removeprefix("swebench-")
    pattern = f"*-run-cloud-*-swebench-{suffix}-seed*"
    for candidate_dir in sorted(TRACES_ROOT.glob(pattern)):
        for jsonl_path in sorted(candidate_dir.glob("*.jsonl")):
            try:
                with jsonl_path.open() as fh:
                    first_line = fh.readline()
            except OSError:
                continue
            if not first_line.strip():
                continue
            try:
                first_record = json.loads(first_line)
            except json.JSONDecodeError:
                continue
            if first_record.get("experiment_id") == campaign:
                return jsonl_path
    return None


def load_trace_records(trace_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with trace_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def request_fingerprint(request: dict[str, Any]) -> str:
    """Stable dedup key -- reuses the existing lineage/turn identity hash
    rather than hand-rolling a new one, so a byte-identical request from two
    different campaigns collapses to the same fingerprint."""
    identity = trace_record.request_identity(request)
    return str(identity["turn_id"])


def teacher_calls_from_trace(
    records: list[dict[str, Any]], task_group: str, campaign: str, trace_path: Path
) -> list[TeacherCall]:
    out: list[TeacherCall] = []
    for index, rec in enumerate(messages_calls(records)):
        if rec.get("placement") != "cloud":
            continue
        response = rec.get("response")
        if not isinstance(response, dict) or not response.get("content"):
            continue  # transport/HTTP failure recorded without a real reply
        if rec.get("status") not in (200, None):
            continue
        request = rec["request"]
        out.append(
            TeacherCall(
                call_id=f"{request_fingerprint(request)}:{index}",
                task_group=task_group,
                source_campaign=campaign,
                source_trace_path=str(trace_path),
                record_index=index,
                request=request,
                teacher_response=response,
                teacher_placement="cloud",
            )
        )
    return out


def deduplicate(calls: Iterable[TeacherCall]) -> tuple[list[TeacherCall], int]:
    """Keep the first occurrence of each distinct request fingerprint.

    Repeated identical requests happen when the same trajectory prefix was
    replayed by more than one campaign (e.g. a task rerun); counting it
    twice would double-weight that one request in the trained model without
    adding real information.
    """
    seen: set[str] = set()
    kept: list[TeacherCall] = []
    dropped = 0
    for call in calls:
        fp = call.call_id.rsplit(":", 1)[0]
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)
        kept.append(call)
    return kept, dropped


def stratified_sample(
    grouped: dict[str, list[TeacherCall]], cfg: ExperimentConfig
) -> list[TeacherCall]:
    """Round-robin across task groups (capped per group) until the target
    count is reached or every group is exhausted -- guarantees every
    available group contributes rather than the largest campaign dominating
    the sample, without requiring equal group sizes.
    """
    rng = random.Random(cfg.seed)
    shuffled: dict[str, list[TeacherCall]] = {}
    for group, group_calls in grouped.items():
        pool = list(group_calls)
        rng.shuffle(pool)
        shuffled[group] = pool[: cfg.max_calls_per_task_group]

    selected: list[TeacherCall] = []
    groups = sorted(shuffled)  # deterministic order; rng already shuffled contents
    cursor = {g: 0 for g in groups}
    while len(selected) < cfg.target_examples:
        progressed = False
        for group in groups:
            if len(selected) >= cfg.target_examples:
                break
            i = cursor[group]
            pool = shuffled[group]
            if i >= len(pool):
                continue
            selected.append(pool[i])
            cursor[group] = i + 1
            progressed = True
        if not progressed:
            break  # every group's pool is exhausted
    return selected


def build_dataset(
    cfg: ExperimentConfig | None = None,
) -> tuple[list[TeacherCall], DatasetManifest]:
    cfg = cfg or ExperimentConfig()
    verdicts = discover_passing_cloud_verdicts()

    all_calls: list[TeacherCall] = []
    for entry in verdicts:
        trace_path = locate_trace_file(entry["campaign"], entry["task_group"])
        if trace_path is None:
            continue
        records = load_trace_records(trace_path)
        all_calls.extend(
            teacher_calls_from_trace(
                records, entry["task_group"], entry["campaign"], trace_path
            )
        )

    deduped, dropped = deduplicate(all_calls)

    grouped: dict[str, list[TeacherCall]] = {}
    for call in deduped:
        grouped.setdefault(call.task_group, []).append(call)

    selected = stratified_sample(grouped, cfg)

    manifest = DatasetManifest(
        seed=cfg.seed,
        target_examples=cfg.target_examples,
        selected_count=len(selected),
        task_group_counts={
            g: sum(1 for c in selected if c.task_group == g) for g in sorted(grouped)
        },
        excluded_dirs=sorted(EXCLUDED_RESULT_DIRS),
        dedup_dropped=dropped,
    )
    return selected, manifest


def save_dataset(calls: list[TeacherCall], manifest: DatasetManifest, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    calls_path = out_dir / "teacher_calls.jsonl"
    with calls_path.open("w") as fh:
        for call in calls:
            fh.write(json.dumps(asdict(call)) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n")


def load_dataset(out_dir: Path) -> list[TeacherCall]:
    calls_path = out_dir / "teacher_calls.jsonl"
    calls: list[TeacherCall] = []
    with calls_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            calls.append(TeacherCall(**json.loads(line)))
    return calls
