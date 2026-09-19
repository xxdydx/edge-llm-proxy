#!/usr/bin/env python3
"""Ties dataset -> dual replay -> labels -> split -> models -> analysis
together.

    python -m experiments.capability_router.orchestrator --dataset-only
    python -m experiments.capability_router.orchestrator --smoke-test
    python -m experiments.capability_router.orchestrator --full-batch   # NOT to be run without explicit review sign-off

No fallback anywhere in this pipeline: a failed preflight/postflight
identity check aborts the run (SystemExit), rather than continuing on an
endpoint whose identity could not be confirmed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from edgeproxy import router

from . import dataset as dataset_mod
from . import executor
from . import labels as labels_mod
from . import splits as splits_mod
from .analysis import run_analysis
from .config import CLOUD_DEEPSEEK, EXPERIMENT, LOCAL_7B, LOCAL_27B, new_run_dir
from .models import build_feature_matrix, train_lightgbm, train_logistic_regression
from .plots import save_all_plots
from .schema import LabeledExample, ReplayOutcome, TeacherCall

log = logging.getLogger("capability_router")

ALL_LOCAL_BACKENDS = {"local_27b": LOCAL_27B, "local_7b": LOCAL_7B}


def _label_one(
    call: TeacherCall, local_outcome: ReplayOutcome, cloud_outcome: ReplayOutcome
) -> LabeledExample:
    features = router.extract_features(call.request)
    local_label, local_detail, _ = labels_mod.action_equivalence(call.request, call.teacher_response, local_outcome)
    cloud_label, cloud_detail, _ = labels_mod.action_equivalence(call.request, call.teacher_response, cloud_outcome)
    return LabeledExample(
        call=call,
        features=asdict(features),
        local_outcome=local_outcome,
        cloud_outcome=cloud_outcome,
        local_label=local_label,
        cloud_label=cloud_label,
        local_schema_valid=labels_mod.schema_validity(call.request, local_outcome),
        cloud_schema_valid=labels_mod.schema_validity(call.request, cloud_outcome),
        local_label_detail=local_detail,
        cloud_label_detail=cloud_detail,
    )


def build_or_load_dataset(fresh: bool) -> list[TeacherCall]:
    out_dir = EXPERIMENT.data_dir
    manifest_path = out_dir / "manifest.json"
    if not fresh and manifest_path.exists():
        log.info("loading existing dataset from %s", out_dir)
        return dataset_mod.load_dataset(out_dir)
    log.info("building dataset (seed=%s, target=%s)", EXPERIMENT.seed, EXPERIMENT.target_examples)
    calls, manifest = dataset_mod.build_dataset(EXPERIMENT)
    dataset_mod.save_dataset(calls, manifest, out_dir)
    n_groups = len(manifest.task_group_counts)
    log.info(
        "dataset: %d calls across %d task groups (target min groups=%d)%s",
        manifest.selected_count, n_groups, EXPERIMENT.min_task_groups,
        "" if n_groups >= EXPERIMENT.min_task_groups
        else " -- BELOW TARGET, proceeding per allow_group_shortfall",
    )
    return calls


def _replay_backend_for_call(
    call: TeacherCall, backend, checkpoint_path: Path, checkpoint: dict, checkpoint_lock: threading.Lock
) -> ReplayOutcome:
    """The atomic, thread-safe unit: replay (or reuse a checkpointed) one
    (call, backend) pair. Every check-then-act on `checkpoint` and every
    file append happens under `checkpoint_lock`, so concurrent calls to
    this function -- from different backends racing on the same call, as
    `_replay_call_all_backends` below does -- can never append the same
    (call_id, backend) pair twice or interleave two partial file writes."""
    key = executor.checkpoint_key(call.call_id, backend.name)

    with checkpoint_lock:
        cached = checkpoint.get(key)
    if cached is not None:
        return _row_to_outcome(cached)

    # The real, potentially slow, network call happens OUTSIDE the lock --
    # only the cheap bookkeeping around it is serialized -- so two backends
    # for the same call genuinely overlap on the network, which is the
    # entire point of this function existing separately from a plain
    # sequential loop.
    outcome, transform = executor.replay_call(call, backend)
    row = _outcome_to_row(call, outcome, transform)

    with checkpoint_lock:
        existing = checkpoint.get(key)
        if existing is not None:
            # Lost a race for this exact key (shouldn't happen -- each
            # (call, backend) pair is only ever submitted once per call --
            # but never trust a duplicate append over an already-recorded row).
            return _row_to_outcome(existing)
        executor.append_checkpoint(checkpoint_path, row)
        checkpoint[key] = row
    return _row_to_outcome(row)


def _replay_call_all_backends(
    call: TeacherCall,
    local_backends: list,
    checkpoint_path: Path,
    checkpoint: dict,
    checkpoint_lock: threading.Lock,
) -> dict[str, LabeledExample]:
    """Call-major, concurrent: cloud plus every requested local backend for
    ONE call are submitted to a thread pool together and run at the same
    time, not one after another. Returns {local_backend_name: LabeledExample}."""
    backends = [CLOUD_DEEPSEEK, *local_backends]
    outcomes: dict[str, ReplayOutcome] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(backends)) as pool:
        future_to_backend = {
            pool.submit(_replay_backend_for_call, call, b, checkpoint_path, checkpoint, checkpoint_lock): b
            for b in backends
        }
        for future in concurrent.futures.as_completed(future_to_backend):
            backend = future_to_backend[future]
            outcomes[backend.name] = future.result()

    cloud_outcome = outcomes[CLOUD_DEEPSEEK.name]
    return {
        lb.name: _label_one(call, outcomes[lb.name], cloud_outcome)
        for lb in local_backends
    }


def _outcome_to_row(call: TeacherCall, outcome: ReplayOutcome, transform: dict) -> dict:
    resp = outcome.response or {}
    usage = resp.get("_usage") or {}
    timing = resp.get("_timing") or {}
    return {
        "call_id": call.call_id,
        "task_group": call.task_group,
        "backend": outcome.backend,
        "status": outcome.status,
        "detail": outcome.detail,
        "model_identity": resp.get("model"),
        "stop_reason": resp.get("stop_reason"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "total_latency_s": timing.get("total_latency_s", outcome.latency_s),
        "ttft_s": timing.get("ttft_s"),
        "tpot_ms": timing.get("tpot_ms"),
        "tps": timing.get("tps"),
        # Bounded fail-fast accounting (2026-09-10), persisted per row so a
        # resume and the artifact pass can report attempt count and the true
        # end-to-end cost separately from the final attempt's own latency.
        "retry_attempts": outcome.retry_attempts,
        "end_to_end_wall_s": outcome.end_to_end_wall_s,
        "final_attempt_latency_s": outcome.final_attempt_latency_s,
        "attempts_meta": outcome.attempts_meta,
        "transform": transform,
        "response": resp if resp else None,
    }


def _row_to_outcome(row: dict) -> ReplayOutcome:
    return ReplayOutcome(
        backend=row["backend"], status=row["status"], response=row.get("response"),
        latency_s=row.get("total_latency_s"), detail=row.get("detail"),
        retry_attempts=row.get("retry_attempts"),
        end_to_end_wall_s=row.get("end_to_end_wall_s"),
        final_attempt_latency_s=row.get("final_attempt_latency_s"),
        attempts_meta=row.get("attempts_meta"),
    )


def run_full_replay(calls: list[TeacherCall], local_backend_names: list[str], run_dir: Path) -> dict[str, list[LabeledExample]]:
    """Returns {local_backend_name: [LabeledExample, ...]}. Checkpointed:
    safe to re-invoke on the same run_dir after an interruption."""
    executor.load_env()
    local_backends = [ALL_LOCAL_BACKENDS[n] for n in local_backend_names]

    evidence_path = run_dir / "endpoint_evidence.json"
    evidence: dict[str, list] = {"preflight": [], "postflight": []}
    for backend in [CLOUD_DEEPSEEK, *local_backends]:
        log.info("preflight: %s", backend.name)
        evidence["preflight"].append(executor.preflight(backend))

    checkpoint_path = run_dir / "dual_execution_results.jsonl"
    checkpoint_on_disk = executor.load_checkpoint(checkpoint_path)
    executor.validate_resume_checkpoint(calls, checkpoint_on_disk)
    checkpoint = executor.retryable_checkpoint(checkpoint_on_disk)
    n_retrying = len(checkpoint_on_disk) - len(checkpoint)
    if checkpoint_on_disk:
        log.info(
            "resuming from existing checkpoint: %d rows already done (terminal), %d TRANSPORT_ERROR row(s) will be retried",
            len(checkpoint), n_retrying,
        )

    checkpoint_lock = threading.Lock()
    results: dict[str, list[LabeledExample]] = {n: [] for n in local_backend_names}
    total = len(calls)
    for i, call in enumerate(calls):
        examples = _replay_call_all_backends(call, local_backends, checkpoint_path, checkpoint, checkpoint_lock)
        for name, example in examples.items():
            results[name].append(example)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            log.info("replayed %d/%d calls (all backends concurrently: %s)", i + 1, total, ",".join(local_backend_names))

    for backend in [CLOUD_DEEPSEEK, *local_backends]:
        log.info("postflight: %s", backend.name)
        evidence["postflight"].append(executor.postflight(backend))

    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    return results


def run_smoke_test(calls: list[TeacherCall], run_dir: Path) -> dict:
    """One representative call, replayed against BOTH local backends and
    cloud, with full richness printed and saved. Never proceeds to a full
    batch -- caller must decide that separately."""
    executor.load_env()
    sample_call = sorted(calls, key=lambda c: c.task_group)[0]
    log.info("smoke-test call: %s (task_group=%s)", sample_call.call_id, sample_call.task_group)

    smoke_dir = run_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"call_id": sample_call.call_id, "task_group": sample_call.task_group, "backends": {}}

    for backend in (CLOUD_DEEPSEEK, LOCAL_27B, LOCAL_7B):
        log.info("smoke: preflight %s", backend.name)
        preflight_evidence = executor.preflight(backend)
        log.info("smoke: replay %s", backend.name)
        outcome, transform = executor.replay_call(sample_call, backend)
        row = _outcome_to_row(sample_call, outcome, transform)
        report["backends"][backend.name] = {"preflight": preflight_evidence, "replay": row}
        log.info(
            "smoke result backend=%s status=%s model=%s stop_reason=%s ttft_s=%s tps=%s tokens_in=%s tokens_out=%s",
            backend.name, outcome.status, row["model_identity"], row["stop_reason"],
            row["ttft_s"], row["tps"], row["input_tokens"], row["output_tokens"],
        )

    for local_name in ("local_27b", "local_7b"):
        local_row = report["backends"][local_name]["replay"]
        cloud_row = report["backends"]["cloud_deepseek"]["replay"]
        local_outcome = _row_to_outcome(local_row)
        cloud_outcome = _row_to_outcome(cloud_row)
        label, detail, components = labels_mod.action_equivalence(
            sample_call.request, sample_call.teacher_response, local_outcome
        )
        report["backends"][local_name]["label_vs_teacher"] = {
            "label": label, "detail": detail, "components": asdict(components),
        }
        log.info("smoke label local=%s vs teacher: %s (%s)", local_name, label, detail)

    (smoke_dir / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def summarize(examples: list[LabeledExample], backend_name: str) -> None:
    from collections import Counter

    local_counts = Counter(ex.local_label for ex in examples)
    cloud_counts = Counter(ex.cloud_label for ex in examples)
    log.info("[%s] local labels: %s", backend_name, dict(local_counts))
    log.info("[%s] cloud labels (sanity -- should be ~all EQUIVALENT): %s", backend_name, dict(cloud_counts))


def _write_experiment_config(run_dir: Path) -> None:
    (run_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "seed": EXPERIMENT.seed,
                "target_examples": EXPERIMENT.target_examples,
                "min_task_groups": EXPERIMENT.min_task_groups,
                "epsilon_grid": EXPERIMENT.epsilon_grid,
                "train_frac": EXPERIMENT.train_frac,
                "val_frac": EXPERIMENT.val_frac,
                "local_27b": {"base_url": LOCAL_27B.base_url, "max_context_tokens": LOCAL_27B.max_context_tokens},
                "local_7b": {"base_url": LOCAL_7B.base_url, "max_context_tokens": LOCAL_7B.max_context_tokens},
                "cloud_deepseek": {"base_url": CLOUD_DEEPSEEK.base_url},
            },
            indent=2,
        )
        + "\n"
    )


def write_artifacts_for_backend(name: str, examples: list[LabeledExample], run_dir: Path) -> dict:
    """Everything downstream of "we have labeled examples for one local
    backend": quality_labels, split, models, analysis report, plots.
    Deliberately takes no dependency on how `examples` was produced -- real
    replay or synthetic/offline test data both go through this exact same
    path, so a synthetic run genuinely exercises the same artifact-writing
    code the real batch uses."""
    from collections import Counter

    local_status_counts = Counter(ex.local_outcome.status for ex in examples)
    cloud_status_counts = Counter(ex.cloud_outcome.status for ex in examples)
    n_invalid_identity = sum(
        1 for ex in examples
        if ex.local_outcome.status == "IDENTITY_MISMATCH" or ex.cloud_outcome.status == "IDENTITY_MISMATCH"
    )
    n_genuine_local_ok = local_status_counts.get("OK", 0)
    arm_valid = True
    reasons: list[str] = []
    if n_invalid_identity:
        arm_valid = False
        reasons.append(f"{name}: {n_invalid_identity} identity-mismatched samples")
    if n_genuine_local_ok == 0:
        arm_valid = False
        reasons.append(f"{name}: zero genuine local OK replies (all {len(examples)} calls were TRANSPORT_ERROR/HTTP_ERROR/INVALID_CAPACITY/IDENTITY_MISMATCH)")
    summarize(examples, name)

    quality_path = run_dir / f"quality_labels_{name}.jsonl"
    with quality_path.open("w") as fh:
        for ex in examples:
            f = ex.features
            budget = LOCAL_27B.max_context_tokens if name == "local_27b" else LOCAL_7B.max_context_tokens
            prompt_tokens = f.get("local_prompt_tokens")
            fh.write(json.dumps({
                "call_id": ex.call.call_id,
                "task_group": ex.call.task_group,
                "source_campaign": ex.call.source_campaign,
                "source_trace_path": ex.call.source_trace_path,
                "record_index": ex.call.record_index,
                "branch_turn_ordinal": f.get("branch_turn_ordinal"),
                "n_messages": f.get("n_messages"),
                "n_tools": f.get("n_tools"),
                "prompt_tokens": prompt_tokens,
                "context_utilization": (prompt_tokens / budget) if prompt_tokens is not None and budget else None,
                "is_tool_continuation": f.get("is_tool_continuation"),
                "errored_tool_result_density": f.get("errored_tool_result_density"),
                "local_status": ex.local_outcome.status,
                "cloud_status": ex.cloud_outcome.status,
                "local_label": ex.local_label, "cloud_label": ex.cloud_label,
                "local_label_detail": ex.local_label_detail, "cloud_label_detail": ex.cloud_label_detail,
                "local_schema_valid": ex.local_schema_valid, "cloud_schema_valid": ex.cloud_schema_valid,
            }) + "\n")

    groups = [ex.call.task_group for ex in examples]
    split = splits_mod.make_split(groups, EXPERIMENT.seed, EXPERIMENT.train_frac, EXPERIMENT.val_frac)
    splits_mod.save_split(split, run_dir / f"split_definition_{name}.json")
    train, val, test = splits_mod.apply_split(examples, split, lambda e: e.call.task_group)

    X_train, y_train, _ = build_feature_matrix(train, "local_label")
    trained = {"logistic_regression": train_logistic_regression(X_train, y_train, EXPERIMENT.seed)}
    lgb_model = train_lightgbm(X_train, y_train, EXPERIMENT.seed)
    if lgb_model is not None:
        trained["lightgbm"] = lgb_model

    report = run_analysis(train, val, test, trained, EXPERIMENT)
    (run_dir / f"analysis_report_{name}.json").write_text(
        json.dumps(_report_to_json(report), indent=2, default=str) + "\n"
    )
    plot_paths = save_all_plots(report, run_dir / name)

    return {
        "valid": arm_valid,
        "reasons": reasons,
        "local_status_counts": dict(local_status_counts),
        "cloud_status_counts": dict(cloud_status_counts),
        "genuine_local_ok": n_genuine_local_ok,
        "total": len(examples),
        "plot_paths": {k: str(v) for k, v in plot_paths.items()},
    }


def run_full_batch(calls: list[TeacherCall], local_backend_names: list[str], resume_run_dir: Path | None = None) -> Path:
    if resume_run_dir is not None:
        if not resume_run_dir.is_dir():
            raise SystemExit(f"--resume-run-dir does not exist: {resume_run_dir}")
        run_dir = resume_run_dir
        log.info("resuming existing run_dir: %s", run_dir)
    else:
        run_dir = new_run_dir()
        log.info("run_dir: %s", run_dir)
        _write_experiment_config(run_dir)
        (run_dir / "replay_dataset.jsonl").write_text(
            "\n".join(json.dumps(asdict(c)) for c in calls) + "\n"
        )

    per_backend_examples = run_full_replay(calls, local_backend_names, run_dir)

    validity: dict = {"run_valid": True, "reasons": [], "by_backend": {}}
    for name, examples in per_backend_examples.items():
        arm_report = write_artifacts_for_backend(name, examples, run_dir)
        validity["by_backend"][name] = arm_report
        validity["reasons"].extend(arm_report["reasons"])
        if not arm_report["valid"]:
            validity["run_valid"] = False

    (run_dir / "validity_manifest.json").write_text(json.dumps(validity, indent=2) + "\n")
    log.info("wrote all artifacts under %s (run_valid=%s)", run_dir, validity["run_valid"])
    return run_dir


def _report_to_json(report) -> dict:
    return {
        "train_size": report.train_size,
        "val_size": report.val_size,
        "test_size": report.test_size,
        "q_cloud_test": report.q_cloud_test,
        "opportunity_2x2_test": report.opportunity_2x2_test,
        "p_local_given_cloud_test": report.p_local_given_cloud_test,
        "baselines_test": {k: asdict(v) for k, v in report.baselines_test.items()},
        "model_val_thresholds": report.model_val_thresholds,
        "model_test_at_epsilon": {
            m: {str(eps): (asdict(r) if r else None) for eps, r in d.items()}
            for m, d in report.model_test_at_epsilon.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-dataset", action="store_true")
    ap.add_argument("--dataset-only", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--full-batch", action="store_true")
    ap.add_argument("--backends", default="local_27b,local_7b")
    ap.add_argument(
        "--resume-run-dir", type=Path, default=None,
        help="resume an existing --full-batch run_dir from its checkpoint instead of starting a new timestamped run",
    )
    args = ap.parse_args(argv)

    calls = build_or_load_dataset(args.fresh_dataset)
    if args.dataset_only:
        return 0

    if args.smoke_test:
        run_dir = new_run_dir()
        log.info("smoke-test run_dir: %s", run_dir)
        try:
            run_smoke_test(calls, run_dir)
        except executor.BackendIdentityError as exc:
            log.error("smoke test aborted: %s", exc)
            return 1
        return 0

    if not args.full_batch:
        log.info("nothing to do -- pass --dataset-only, --smoke-test, or --full-batch")
        return 0

    log.warning("FULL BATCH requested -- this spends real endpoint calls across the whole dataset")
    try:
        run_full_batch(calls, args.backends.split(","), resume_run_dir=args.resume_run_dir)
    except executor.BackendIdentityError as exc:
        log.error("aborting: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
