"""Run a bounded DeepSeek-only ICL shadow probe on the frozen v3 pilot.

No reserved task is queried.  Each target is scored from examples and
classifier fits on other task groups only.  Every call is checkpointed before
the next begins.  This is exploratory and cannot establish task success or a
deployable threshold.

Usage: ``uv run --no-sync python -m experiments.agentic_router.run_icl_shadow``.
The caller must coordinate with the live campaign before starting.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.capability_router import executor
from experiments.capability_router.config import CLOUD_DEEPSEEK
from experiments.capability_router.schema import TeacherCall

from . import analysis, icl_router, model_comparison

MAX_CALLS = 20
PER_GROUP = 5
DEADLINE = datetime(2026, 9, 17, 16, 30, tzinfo=ZoneInfo("Asia/Singapore"))
OUTPUT = model_comparison.RESULTS / "icl_shadow_v3_dev20.jsonl"
SUMMARY = model_comparison.RESULTS / "icl_shadow_v3_dev20_summary.json"


def choose_targets(events: list[model_comparison.LabeledEvent]) -> list[model_comparison.LabeledEvent]:
    rng = random.Random(20260916)
    by_group: dict[str, list[model_comparison.LabeledEvent]] = {}
    for event in events:
        if event.label == "UNKNOWN" or event.task_group in model_comparison.RESERVED_PILOT_HOLDOUT_TASKS:
            continue
        by_group.setdefault(event.task_group, []).append(event)
    chosen = []
    for group in sorted(by_group):
        harm = sorted((e for e in by_group[group] if e.label == "HARM"), key=lambda e: e.call_id)
        safe = sorted((e for e in by_group[group] if e.label == "SAFE"), key=lambda e: e.call_id)
        rng.shuffle(harm)
        rng.shuffle(safe)
        selected = harm[: min(2, len(harm))] + safe[:PER_GROUP]
        chosen.extend(selected[:PER_GROUP])
    if len(chosen) > MAX_CALLS:
        chosen = chosen[:MAX_CALLS]
    if len({e.call_id for e in chosen}) != len(chosen):
        raise ValueError("duplicate selected target")
    return chosen


def _text(response: dict | None) -> str:
    if not response:
        return ""
    return "\n".join(str(b.get("text", "")) for b in response.get("content", [])
                     if isinstance(b, dict) and b.get("type") == "text")


def _usage(response: dict | None) -> dict:
    usage = response.get("usage") if response else None
    usage = usage if isinstance(usage, dict) else {}
    values = {name: usage.get(name) if isinstance(usage.get(name), (int, float)) else None
              for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
    # The gateway has returned literal 0/0 for visibly nonempty prompts and
    # completions.  Preserve raw usage in checkpoint rows; in summaries,
    # these are unknown rather than free tokens.
    return values


def _reportable_usage(row: dict, name: str) -> int | float | None:
    value = row["usage"].get(name)
    if name == "input_tokens" and value == 0 and row.get("prompt"):
        return None
    if name == "output_tokens" and value == 0 and row.get("raw_response"):
        return None
    return value


def _existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done = {}
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("malformed ICL checkpoint; refusing duplicate spend") from exc
            cid = row["call_id"]
            if cid in done:
                raise ValueError(f"duplicate ICL checkpoint: {cid}")
            done[cid] = row
    return done


def baseline_predictions(events: list[model_comparison.LabeledEvent], targets: list[model_comparison.LabeledEvent]) -> dict[str, dict[str, float]]:
    # Identical heldout targets as ICL.  For each task, train on *other*
    # tasks only.  No target label enters its fitted baseline.
    out: dict[str, dict[str, float]] = {}
    for group in sorted({e.task_group for e in targets}):
        train = [e for e in events if e.label != "UNKNOWN" and e.task_group != group
                 and e.task_group not in model_comparison.RESERVED_PILOT_HOLDOUT_TASKS]
        test = [e for e in targets if e.task_group == group]
        Xtr = np.asarray([[e.features[n] for n in analysis.FEATURE_NAMES] for e in train])
        ytr = np.asarray([int(e.label == "HARM") for e in train])
        Xte = np.asarray([[e.features[n] for n in analysis.FEATURE_NAMES] for e in test])
        if len(set(ytr)) < 2:
            raise ValueError(f"single-class baseline training set for {group}")
        for kind in ("logreg", "gbm", "rf", "knn"):
            model = model_comparison._model(kind)
            if kind == "knn":
                model[-1].n_neighbors = min(5, len(train))
            model.fit(Xtr, ytr)
            for event, prob in zip(test, model.predict_proba(Xte)[:, 1]):
                out.setdefault(event.call_id, {})[kind] = float(prob)
    return out


def _metric(labels: list[int], probs: list[float]) -> dict:
    if not probs:
        return {"n": 0, "roc_auc": None, "average_precision": None}
    return {"n": len(probs),
            "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None,
            "average_precision": float(average_precision_score(labels, probs)) if sum(labels) else None}


def summarize(rows: list[dict], baseline: dict[str, dict[str, float]]) -> dict:
    labels = [int(r["true_label"] == "HARM") for r in rows]
    out = {
        "status": "exploratory_dev_only_n20_max",
        "n_calls": len(rows),
        "n_task_groups": len({r["task_group"] for r in rows}),
        "sample_scheme": "up to 5/task, deliberately enrich up to 2 HARM/task; not population prevalence",
        "icl_abstentions": sum(r["parsed_verdict"] == "UNKNOWN" for r in rows),
        "n_latency_known": sum(r["latency_s"] is not None for r in rows),
        "n_input_usage_known": sum(_reportable_usage(r, "input_tokens") is not None for r in rows),
        "n_output_usage_known": sum(_reportable_usage(r, "output_tokens") is not None for r in rows),
        "mean_latency_s": float(np.mean([r["latency_s"] for r in rows if r["latency_s"] is not None])) if any(r["latency_s"] is not None for r in rows) else None,
        "mean_input_tokens": float(np.mean([_reportable_usage(r, "input_tokens") for r in rows if _reportable_usage(r, "input_tokens") is not None])) if any(_reportable_usage(r, "input_tokens") is not None for r in rows) else None,
        "mean_output_tokens": float(np.mean([_reportable_usage(r, "output_tokens") for r in rows if _reportable_usage(r, "output_tokens") is not None])) if any(_reportable_usage(r, "output_tokens") is not None for r in rows) else None,
        "mean_estimated_prompt_tokens_chars_div4": float(np.mean([len(r["prompt"]) / 4 for r in rows])) if rows else None,
        "mean_estimated_completion_tokens_chars_div4": float(np.mean([len(r["raw_response"]) / 4 for r in rows])) if rows else None,
        "token_estimate_note": "character-count proxy only; raw gateway usage is zero/zero on nonempty calls and treated as unknown",
        "provider_zero_usage_rows": sum(r["usage"].get("input_tokens") == 0 and r["usage"].get("output_tokens") == 0 and bool(r.get("raw_response")) for r in rows),
        "metrics": {},
        "limitations": "Judge-derived proxy labels, enriched tiny sample, same DeepSeek family as labeler; no live task-success inference.",
    }
    for kind in ("logreg", "gbm", "rf", "knn"):
        out["metrics"][kind] = _metric(labels, [baseline[r["call_id"]][kind] for r in rows])
    answered = [r for r in rows if r["parsed_verdict"] != "UNKNOWN" and r["harm_probability"] is not None]
    evaluable = [(int(r["true_label"] == "HARM"), r["harm_probability"]) for r in answered]
    out["metrics"]["icl"] = _metric([x[0] for x in evaluable], [x[1] for x in evaluable])
    out["baseline_on_icl_answered_subset"] = {
        kind: _metric([int(r["true_label"] == "HARM") for r in answered],
                      [baseline[r["call_id"]][kind] for r in answered])
        for kind in ("logreg", "gbm", "rf", "knn")
    }
    return out


def main() -> None:
    if CLOUD_DEEPSEEK.request_model != "deepseek-v4-flash" or CLOUD_DEEPSEEK.expected_model_exact != "deepseek-v4-flash":
        raise RuntimeError("cloud identity config changed; refusing ICL probe")
    events, meta = model_comparison.load_events()
    if meta["judge_protocol_versions"] != ["v3-unbounded-full-tools-2026-09-16"]:
        raise RuntimeError("ICL pilot requires frozen v3 historical labels")
    targets = choose_targets(events)
    if len(targets) != MAX_CALLS or len({e.task_group for e in targets}) != 4:
        raise RuntimeError(f"pilot selection unexpected: {len(targets)} targets")
    baseline = baseline_predictions(events, targets)
    done = _existing(OUTPUT)
    selected = {e.call_id for e in targets}
    if set(done) - selected:
        raise ValueError("checkpoint contains calls outside frozen selection")
    executor.load_env()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for event in targets:
        if event.call_id in done:
            continue
        if datetime.now(ZoneInfo("Asia/Singapore")) >= DEADLINE:
            break
        train = [e for e in events if e.label in ("SAFE", "HARM") and e.task_group != event.task_group
                 and e.task_group not in model_comparison.RESERVED_PILOT_HOLDOUT_TASKS]
        exemplars = icl_router.choose_exemplars(event, train)
        prompt = icl_router.build_prompt(event, exemplars)
        request = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 128, "temperature": 0}
        call = TeacherCall(event.call_id, event.task_group, "icl-shadow-v3", "historical-v3",
                           0, request, {}, "cloud")
        outcome, _ = executor.replay_call(call, CLOUD_DEEPSEEK)
        model_id = outcome.response.get("model") if outcome.response else None
        raw_text = _text(outcome.response)
        verdict, prob = icl_router.parse_response(raw_text) if outcome.status == "OK" else ("UNKNOWN", None)
        row = {
            "call_id": event.call_id, "task_group": event.task_group,
            "true_label": event.label, "exemplar_call_ids": [e.call_id for e in exemplars],
            "prompt": prompt, "raw_response": raw_text,
            "status": outcome.status, "model_identity": model_id,
            "latency_s": outcome.end_to_end_wall_s or outcome.latency_s,
            "usage": _usage(outcome.response),
            "parsed_verdict": verdict, "harm_probability": prob,
            "baseline_probabilities": baseline[event.call_id],
            "timestamp_sgt": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        }
        with OUTPUT.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        done[event.call_id] = row
        print(f"checkpointed {len(done)}/{len(targets)} ICL shadow calls; status={outcome.status}; model={model_id}; parsed={verdict}", flush=True)
        if outcome.status != "OK":
            # A failed identity, HTTP 5xx, or transport event is evidence of
            # an unhealthy endpoint.  Preserve it and stop; never spend the
            # rest of the 20-call budget in a failure loop.
            break
    summary = summarize([done[e.call_id] for e in targets if e.call_id in done], baseline)
    summary["provenance"] = {"judge_file": meta["judge_file"], "judge_sha256": meta["judge_sha256"],
                             "feature_derivation_version": meta["feature_derivation_version"],
                             "max_calls": MAX_CALLS, "deadline_sgt": DEADLINE.isoformat()}
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
