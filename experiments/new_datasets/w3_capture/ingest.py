"""Ingest w3_capture run_dirs + edgeproxy traces into Dataset A/C.

Usage: python3 w3_capture/ingest.py <run_dir> [<run_dir> ...]
Run from the flowmesh repo root. Each run_dir is a directory name under
w3_capture/ (e.g. getmoto__moto-5752__edge-only-v1__r2) with a matching
grading_results_<tag>.json placed at w3_capture/grading_results_<run_dir>.json
containing a single-element list: [{"session_id", "resolved", "report"}].

Real predecision_features per SONNET_MASTER_PROMPT.md section 7 (see
w1_storage/features.py) -- computed from each call's own request history,
never from future information.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "experiments/new_datasets")
from w1_storage.store import JobStore
from w1_storage.schemas import PrefixOutcome, ServingCall
from w1_storage.features import extract_predecision_features, FEATURES_VERSION
from w1_storage.prefix_diff import compute_prefix_diff, session_cache_progress

STORE_ROOT = "experiments/new_datasets"
CAMPAIGN_ID = "smoke-v1"
LOCAL_CONTEXT_WINDOW = 262144


def _seeded_rank(campaign_id: str, trajectory_id: str, i: int) -> str:
    return hashlib.sha256(f"{campaign_id}|core-sample|{trajectory_id}|{i}".encode()).hexdigest()


def ingest_run_dir(store: JobStore, run_dir: str, campaign_id: str = CAMPAIGN_ID,
                    protocol_version: str = "smoke-v1", cohort: str = "smoke") -> tuple[int, int]:
    nd = Path("experiments/new_datasets")
    meta = json.loads((nd / "w3_capture" / run_dir / "run_meta.json").read_text())
    grading_tag = run_dir.replace("/", "__")
    grading_path = nd / "w3_capture" / f"grading_results_{grading_tag}.json"
    grading_list = json.loads(grading_path.read_text()) if grading_path.exists() else []
    grading = {r["session_id"]: r for r in grading_list}

    sid = meta["session_id"]
    iid = meta["instance_id"]
    policy = meta["policy"]

    trace_files = [nd / "w3_capture/edgeproxy_traces/local_only.jsonl",
                   nd / "w3_capture/edgeproxy_traces/cloud_only.jsonl"]
    recs = []
    for fname in trace_files:
        if not fname.exists():
            continue
        for line in fname.open():
            rec = json.loads(line)
            if rec.get("path") != "/v1/messages":
                continue
            if (rec.get("call") or {}).get("session_id") == sid:
                recs.append(rec)
    recs.sort(key=lambda r: r["ts"])
    n = len(recs)

    grade = grading.get(sid)
    resolved = grade["resolved"] if grade else None
    grade_report_ref = store.put_artifact(json.dumps(grade["report"], sort_keys=True).encode())["ref"] if grade else None

    task_id = f"gym:{iid}"
    trajectory_id = f"{task_id}:{policy}:{sid}"
    backend_fp = "local:Inferact/Qwen3.8-27B-NVFP4" if policy == "edge-only-v1" else "cloud:deepseek-v4-flash"

    k = min(5, n)
    ranked = sorted(range(n), key=lambda i: _seeded_rank(campaign_id, trajectory_id, i))
    core_selected = set(ranked[:k])
    selection_probability = (k / n) if n else None

    total_a = total_c = 0
    session_cache_baseline = None
    for i, rec in enumerate(recs):
        call = rec.get("call") or {}
        vllm_snapshot = (rec.get("local_resources") or {}).get("vllm")
        if vllm_snapshot and session_cache_baseline is None:
            session_cache_baseline = vllm_snapshot
        session_cache = session_cache_progress(session_cache_baseline, vllm_snapshot)
        request_raw = json.dumps(rec.get("request") or {}, sort_keys=True).encode()
        response_raw = json.dumps(rec.get("response") or {}, sort_keys=True).encode()
        req_art = store.put_artifact(request_raw)
        resp_art = store.put_artifact(response_raw)

        prefix_id = f"{trajectory_id}:call{i}"
        is_last = (i == n - 1)

        messages = (rec.get("request") or {}).get("messages", [])
        has_prior = i > 0
        prior_stop = None
        if has_prior:
            prior_resp = recs[i - 1].get("response") or {}
            prior_stop = prior_resp.get("stop_reason") if isinstance(prior_resp, dict) else None
        features = extract_predecision_features(messages, rec.get("request") or {}, LOCAL_CONTEXT_WINDOW,
                                                  has_prior, prior_stop)
        prior_request = recs[i - 1].get("request") if has_prior else None
        features["prefix_diff"] = compute_prefix_diff(prior_request, rec.get("request") or {})
        features["session_prefix_cache"] = session_cache

        a_missing = {"tool_schema_ref": "not_extracted_this_protocol", "tool_schema_hash": "not_extracted_this_protocol"}
        if grade_report_ref is None:
            a_missing["terminal_grade_ref"] = "no_grade_available"
            a_missing["resolved"] = "no_grade_available"
        a = PrefixOutcome(
            schema_version=1, protocol_version=protocol_version, campaign_id=campaign_id,
            task_id=task_id, trajectory_id=trajectory_id, source="edgeproxy-live-capture",
            cohort=cohort, split="train",
            provenance={"instance_id": iid, "policy": policy, "session_id": sid, "run_dir": run_dir,
                        "docker_image": f"sweb.eval.arm64.{iid}:latest", "repeat_index": meta.get("repeat_index", 0),
                        "execution_order": meta.get("execution_order")},
            missing_reasons=a_missing,
            prefix_id=prefix_id, source_policy=policy, backend_fingerprint_id=backend_fp,
            call_index=i,
            predecision_request_ref=req_art["ref"], predecision_request_hash=req_art["sha256"],
            tool_schema_ref=None, tool_schema_hash=None,
            history_integrity="complete_from_proxy_boundary",
            predecision_features=features,
            features_version=FEATURES_VERSION,
            remaining_budget_at_prefix={"max_main_logical_calls": 40, "calls_used_so_far": i},
            core_sample_selected=(i in core_selected),
            selection_probability=selection_probability,
            terminal_grade_ref=grade_report_ref,
            resolved=(resolved if grade_report_ref else None),
            label_valid=bool(grade_report_ref),
            termination_reason=("official_grade_joined" if is_last else "trajectory_continues"),
            outcome_censored=False,
            sampling_metadata={"trajectory_length": n, "is_last_prefix": is_last},
        )
        store.append_record("A", a)
        total_a += 1

        c = ServingCall(
            schema_version=1, protocol_version=protocol_version, campaign_id=campaign_id,
            task_id=task_id, trajectory_id=trajectory_id, source="edgeproxy-live-capture",
            cohort=cohort, split="train",
            provenance={"instance_id": iid, "policy": policy, "session_id": sid},
            invocation_id=call.get("call_id") or f"{prefix_id}:invocation0",
            logical_call_id=call.get("call_id") or f"{prefix_id}:logical",
            prefix_id=prefix_id, branch_id=None, preference_pair_id=None,
            purpose="original", invocation_kind="original",
            backend_fingerprint_id=backend_fp,
            logical_request_hash=req_art["sha256"], rendered_request_hash=req_art["sha256"],
            attempt_index=0,
            pre_dispatch_snapshot={"local_resources": rec.get("local_resources")} if rec.get("local_resources") else {},
            snapshot_timestamp=str(rec.get("local_resources", {}).get("sampled_at")) if rec.get("local_resources") else None,
            snapshot_age_ms=None, snapshot_source=("edgeproxy_local_resources" if rec.get("local_resources") else None),
            request_start=str(rec["ts"]), first_byte=None, first_content=None, end=None,
            measured_timings=rec.get("timing") or {},
            raw_usage=rec.get("usage") or {},
            normalised_usage={k: v for k, v in (call.get("tokens") or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
            usage_integrity=(call.get("tokens") or {}).get("usage_integrity", "unknown"),
            status=str(rec.get("status")),
            response_ref=resp_art["ref"], error_ref=None,
            cost_basis=None, rate_version=None, observed_or_estimated_cost=None,
            latency_censored=False,
            measurement_quality_flags=["live_edgeproxy_capture"],
            missing_reasons={"snapshot_age_ms": "not_computed", "cost_basis": "not_priced", "rate_version": "not_priced",
                              "observed_or_estimated_cost": "not_priced", "first_byte": "not_recorded_this_protocol",
                              "first_content": "not_recorded_this_protocol", "end": "not_recorded_this_protocol",
                              "error_ref": "no_transport_error",
                              "branch_id": "not_a_branch_invocation", "preference_pair_id": "not_a_preference_invocation",
                              **({} if rec.get("local_resources") else {"snapshot_timestamp": "local_resources_absent", "snapshot_source": "local_resources_absent"})},
        )
        store.append_record("C", c)
        total_c += 1

    print(f"{run_dir}: {n} real calls, resolved={resolved}, A+={total_a}, C+={total_c}")
    return total_a, total_c


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--campaign-id", default=CAMPAIGN_ID)
    ap.add_argument("--protocol-version", default="smoke-v1")
    ap.add_argument("--cohort", default="smoke")
    args = ap.parse_args()
    store = JobStore(STORE_ROOT)
    total_a = total_c = 0
    for run_dir in args.run_dirs:
        a, c = ingest_run_dir(store, run_dir, campaign_id=args.campaign_id,
                               protocol_version=args.protocol_version, cohort=args.cohort)
        total_a += a; total_c += c
    paths = store.finalize_export()
    print(f"TOTAL A+={total_a} C+={total_c}")
    print("exported:", [str(p) for p in paths])


if __name__ == "__main__":
    main()
