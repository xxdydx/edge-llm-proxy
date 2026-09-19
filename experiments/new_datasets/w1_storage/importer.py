"""Read-only adapter for edgeproxy's JSONL trace envelopes.

This module never modifies the input trace and intentionally does not infer a
second branch from a later record in the same task.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from .schemas import PrefixOutcome, ServingCall
from .store import JobStore

def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",",":"), default=str).encode()).hexdigest()

def import_legacy_trace(path: str | Path, store: JobStore, *, campaign_id: str = "legacy-import", split: str = "train", cohort: str = "legacy") -> dict[str, int]:
    """Import trace records into A/C; returns counts. No B rows are inferred."""
    counts={"A":0,"C":0,"B":0,"preference":0,"skipped":0}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            try: envelope=json.loads(line)
            except json.JSONDecodeError: counts["skipped"]+=1; continue
            request=envelope.get("request"); response=envelope.get("response")
            call=envelope.get("call") or {}
            if not isinstance(request, dict): counts["skipped"]+=1; continue
            task_id=str(envelope.get("task_id") or envelope.get("experiment_id") or call.get("experiment_id") or "legacy-task")
            trajectory_id=str(envelope.get("trajectory_id") or envelope.get("episode_id") or call.get("episode_id") or task_id+":trajectory")
            logical=str(call.get("logical_call_id") or call.get("call_id") or envelope.get("id") or "legacy-call")
            invocation=str(call.get("invocation_id") or call.get("call_id") or envelope.get("id") or logical+":attempt0")
            prefix_id=f"{trajectory_id}:{logical}"
            request_raw=json.dumps(request,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
            req_art=store.put_artifact(request_raw)
            missing={"tool_schema_ref":"not_recorded","tool_schema_hash":"not_recorded","selection_probability":"not_sampled","terminal_grade_ref":"no_unambiguous_grade_join","resolved":"no_unambiguous_grade_join"}
            if response is None: missing["response_ref"]="response_absent_in_legacy_trace"
            response_ref=None
            if response is not None:
                response_raw=json.dumps(response,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
                response_ref=store.put_artifact(response_raw)["ref"]
            common=dict(schema_version=1,protocol_version="legacy-import-v1",campaign_id=campaign_id,task_id=task_id,trajectory_id=trajectory_id,source="edgeproxy-trace",cohort=cohort,split=split,provenance={"trace_path":str(path),"legacy_record_id":str(envelope.get("id") or logical)},missing_reasons=missing)
            a=PrefixOutcome(**common,prefix_id=prefix_id,source_policy="legacy-recorded-policy",backend_fingerprint_id=str(call.get("backend") or envelope.get("placement") or "unknown-backend"),call_index=int(envelope.get("call_index") or 0),predecision_request_ref=req_art["ref"],predecision_request_hash=req_art["sha256"],tool_schema_ref=None,tool_schema_hash=None,history_integrity="unknown",features_version="legacy-unavailable",remaining_budget_at_prefix={},core_sample_selected=False,selection_probability=None,terminal_grade_ref=None,resolved=None,label_valid=False,termination_reason="unknown",outcome_censored=False,sampling_metadata={"legacy_labels_present":bool(envelope.get("local_label") or envelope.get("cloud_label"))})
            store.append_record("A",a); counts["A"]+=1
            cm={"branch_id":"not a branch","preference_pair_id":"not a preference pair","snapshot_timestamp":"legacy unavailable","snapshot_age_ms":"legacy unavailable","snapshot_source":"legacy unavailable","request_start":"legacy unavailable","first_byte":"legacy unavailable","first_content":"legacy unavailable","end":"legacy unavailable","response_ref":"response_absent" if response_ref is None else "present","error_ref":"no legacy error artifact","cost_basis":"legacy unavailable","rate_version":"legacy unavailable","observed_or_estimated_cost":"legacy unavailable"}
            c=ServingCall(**{**common,"missing_reasons":cm},invocation_id=invocation,logical_call_id=logical,prefix_id=prefix_id,branch_id=None,preference_pair_id=None,purpose="original",invocation_kind="original",backend_fingerprint_id=str(call.get("backend") or envelope.get("placement") or "unknown-backend"),logical_request_hash=_hash(request),rendered_request_hash=_hash(request),attempt_index=0,pre_dispatch_snapshot={},snapshot_timestamp=None,snapshot_age_ms=None,snapshot_source=None,request_start=None,first_byte=None,first_content=None,end=None,measured_timings=call.get("timing") or {},raw_usage=(call.get("tokens") or {}),normalised_usage={},usage_integrity="unknown",status=str(envelope.get("status") or "unknown"),response_ref=response_ref,error_ref=None,cost_basis=None,rate_version=None,observed_or_estimated_cost=None,latency_censored=False,measurement_quality_flags=["legacy_format"])
            store.append_record("C",c); counts["C"]+=1
    return counts
