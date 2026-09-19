"""Synthetic fixtures only; no real campaign rows or trace paths are used."""
from __future__ import annotations
import gzip, json, tempfile, unittest
from dataclasses import fields
from pathlib import Path
from experiments.new_datasets.w1_storage.importer import import_legacy_trace
from experiments.new_datasets.w1_storage.schemas import *
from experiments.new_datasets.w1_storage.store import ArtifactMismatch, JobStore

def base():
    return dict(schema_version=1,protocol_version="synthetic-v1",campaign_id="syn-campaign",task_id="syn-task",trajectory_id="syn-task:traj",source="synthetic-fixture",cohort="synthetic",split="train",provenance={"fixture":True})

class StorageTests(unittest.TestCase):
    def test_typed_round_trip_and_malformed(self):
        x=PrefixOutcome(**base(),prefix_id="syn-task:traj:call0",source_policy="edge-only-v1",backend_fingerprint_id="syn-edge",call_index=0,predecision_request_ref="artifact:"+"a"*64,predecision_request_hash="a"*64,tool_schema_ref=None,tool_schema_hash=None,history_integrity="complete",features_version="syn-v1",remaining_budget_at_prefix={},core_sample_selected=False,selection_probability=None,terminal_grade_ref=None,resolved=None,label_valid=False,termination_reason="unknown",outcome_censored=False,missing_reasons={"tool_schema_ref":"not_recorded","tool_schema_hash":"not_recorded","selection_probability":"not_sampled","terminal_grade_ref":"not_graded","resolved":"no_unambiguous_grade_join"})
        self.assertEqual(parse_record("A",json.loads(x.to_json())).prefix_id,x.prefix_id)
        with self.assertRaises(ValidationError): parse_record("A",{**json.loads(x.to_json()),"call_index":-1})

    def test_pre_call_shape_excludes_response_and_future(self):
        self.assertNotIn("response", {f.name for f in fields(PreCallRequest)})
        self.assertNotIn("future", {f.name for f in fields(PreCallRequest)})

    def test_original_and_replay_are_distinct(self):
        kw=base(); common=dict(**kw,logical_call_id="syn-logical",prefix_id="syn-prefix",branch_id=None,preference_pair_id=None,purpose="original",backend_fingerprint_id="syn-edge",logical_request_hash="a"*64,rendered_request_hash="b"*64,attempt_index=0,pre_dispatch_snapshot={},snapshot_timestamp=None,snapshot_age_ms=None,snapshot_source=None,request_start=None,first_byte=None,first_content=None,end=None,measured_timings={},raw_usage={},normalised_usage={},usage_integrity="unknown",status="completed",response_ref=None,error_ref=None,cost_basis=None,rate_version=None,observed_or_estimated_cost=None,latency_censored=False,measurement_quality_flags=[],missing_reasons={"response_ref":"synthetic-minimal"})
        common["missing_reasons"]={"branch_id":"not a branch","preference_pair_id":"not a preference pair","snapshot_timestamp":"unknown","snapshot_age_ms":"unknown","snapshot_source":"unknown","request_start":"unknown","first_byte":"unknown","first_content":"unknown","end":"unknown","response_ref":"absent","error_ref":"absent","cost_basis":"unknown","rate_version":"unknown","observed_or_estimated_cost":"unknown"}
        a=ServingCall(**common,invocation_id="syn-original",invocation_kind="original")
        b=ServingCall(**{**common,"purpose":"replay"},invocation_id="syn-replay",invocation_kind="replay")
        a.validate(); b.validate(); self.assertNotEqual(a.invocation_id,b.invocation_id)

    def test_atomic_recovery_and_idempotent_finalize(self):
        with tempfile.TemporaryDirectory() as d:
            s=JobStore(d); self.assertEqual(s.recover_journal(),0)
            s.journal.write_text('{"dataset":"A","payload":\n',encoding="utf-8")
            s.finalize_export(); self.assertEqual((Path(d)/"datasets/prefix_outcomes.jsonl").read_text(),"")
            self.assertEqual(s.finalize_export(),s.finalize_export())

    def test_duplicate_claim_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            s=JobStore(d); s.claim("syn-job","worker-1")
            with self.assertRaises(RuntimeError): s.claim("syn-job","worker-2")
            s.release("syn-job","worker-1")

    def test_artifact_hash_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            s=JobStore(d); ref=s.put_artifact(b"synthetic bytes")["ref"]; digest=ref[9:]
            with gzip.open(Path(d)/"artifacts"/(digest+".gz"),"wb") as f: f.write(b"tampered")
            with self.assertRaises(ArtifactMismatch): s.read_artifact(ref)

    def test_legacy_labels_only_do_not_make_b_or_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            trace=Path(d)/"synthetic-trace.jsonl"; trace.write_text(json.dumps({"id":"syn-call","request":{"model":"syn"},"response":{"content":[]},"placement":"local","local_label":"PASS","cloud_label":"FAIL"})+"\n")
            s=JobStore(Path(d)/"store"); counts=import_legacy_trace(trace,s)
            self.assertEqual(counts["B"],0); self.assertEqual(counts["preference"],0); self.assertEqual(counts["A"],1)
            row=json.loads(s.db.execute("SELECT payload FROM events WHERE dataset='A'").fetchone()[0]); self.assertIsNone(row["resolved"]); self.assertNotIn("aggregated_verdict",row)

    def test_integrity_failures(self):
        with self.assertRaises(ValidationError):
            bad=ServingCall(**base(),invocation_id="syn-i",logical_call_id="syn-l",prefix_id=None,branch_id=None,preference_pair_id=None,purpose="original",invocation_kind="original",backend_fingerprint_id="syn",logical_request_hash="a"*64,rendered_request_hash="b"*64,attempt_index=0,pre_dispatch_snapshot={},snapshot_timestamp=None,snapshot_age_ms=None,snapshot_source=None,request_start=None,first_byte=None,first_content=None,end=None,measured_timings={"client_total_ms":-1},raw_usage={"input_tokens":-1},normalised_usage={},usage_integrity="unknown",status="unknown",response_ref=None,error_ref=None,cost_basis=None,rate_version=None,observed_or_estimated_cost=None,latency_censored=False,measurement_quality_flags=[],missing_reasons={"prefix_id":"not applicable","branch_id":"not applicable","preference_pair_id":"not applicable","snapshot_timestamp":"unknown","snapshot_age_ms":"unknown","snapshot_source":"unknown","request_start":"unknown","first_byte":"unknown","first_content":"unknown","end":"unknown","response_ref":"absent","error_ref":"absent","cost_basis":"unknown","rate_version":"unknown","observed_or_estimated_cost":"unknown"})
            bad.validate()

if __name__ == "__main__": unittest.main()
