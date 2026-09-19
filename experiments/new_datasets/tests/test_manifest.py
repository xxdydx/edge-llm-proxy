"""Synthetic metadata tests only: no source downloads, LLM calls or task execution."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import harvest_task_manifest as h

SPEC=json.loads((h.HERE/'TASK_SOURCE_LOCK.json').read_text())

def fixture_row(source,repo,i,instance=None):
    rid=instance or f"{repo.replace('/','__')}-synthetic-fixture-{i}"
    raw={"instance_id":rid,"repo":repo if source!='smith' else 'swesmith/'+repo.replace('/','__')+'.abcdef12',
         "problem_statement":f"SYNTHETIC UNIT TEST FIXTURE ONLY. Reproduce issue {i} in {repo}; distinct case {rid}.",
         "patch":f"diff --git a/pkg/f{i}.py b/pkg/f{i}.py\n--- a/pkg/f{i}.py\n+++ b/pkg/f{i}.py\n@@ -1,1 +1,1 @@ def function_{i}():\n-return {i}\n+return {i+1}\n",
         "FAIL_TO_PASS":[f'test_{i}'],"PASS_TO_PASS":[f'other_{i}'],"image_name":"unit-test-image", "base_commit":"1"*40}
    return h.source_meta(raw,source,SPEC['sources'][source],{'synthetic_fixture':True})

def fixture_catalogue():
    out=[]
    for q in SPEC['real_quotas']:
        for i in range(q['train']+q['validation']+25):
            sid=SPEC['smoke_instance_ids'][i] if q['repo']=='getmoto/moto' and i<2 else None
            r=fixture_row('gym',q['repo'],i,sid);out.append(r)
            if i<8:
                m=copy.deepcopy(r);m.update(source_key='lite',dataset_id=SPEC['sources']['lite']['dataset_id'],dataset_revision=SPEC['sources']['lite']['revision'],task_uid='lite:'+r['instance_id']);out.append(m)
    for repo in SPEC['synthetic_train_repos']+SPEC['synthetic_validation_repos']:
        out.extend(fixture_row('smith',repo,i) for i in range(18))
    for repo in ['fixture/unseen-a','fixture/unseen-b','fixture/unseen-c']:
        out.extend(fixture_row('verified',repo,i) for i in range(50))
    return out

class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.rows=fixture_catalogue();self.cat=self.root/'catalogue.jsonl';h.atomic_jsonl(self.cat,self.rows)
        self.plan=self.root/'plan'
        with contextlib.redirect_stdout(io.StringIO()):
            h.select(argparse.Namespace(catalogue=self.cat,exclude_ids=None,out=self.plan),SPEC)
    def tearDown(self): self.temp.cleanup()
    def rows_for(self,split): return h.read_jsonl(self.plan/f'{split}.provisional.jsonl')
    def ledger(self,overrides=None):
        rows=[]
        for r in self.rows_for('train')+self.rows_for('validation')+h.read_jsonl(self.plan/'reserves.jsonl'):
            rows.append({'task_uid':r['task_uid'],'status':'pass','selection_used_model_outcome':False,'warm_grade_seconds':1.0,
                         'certificate_ref':'unit-test-only/'+r['task_uid'],'image_digest':'sha256:'+'a'*64})
        overrides=overrides or {}
        for r in rows:
            if r['task_uid'] in overrides:r.update(overrides[r['task_uid']])
        path=self.root/f'preflight-{len(list(self.root.glob("preflight*")))}.jsonl';h.atomic_jsonl(path,rows);return path
    def finish(self,stage='smoke',ledger=None,out='freeze',previous=None):
        args=argparse.Namespace(plan=self.plan,preflight=ledger or self.ledger(),stage=stage,out=self.root/out,previous_freeze=previous)
        with contextlib.redirect_stdout(io.StringIO()):h.finalise(args,SPEC)
        return args.out
    def test_exact_counts_and_groups(self):
        t,v=self.rows_for('train'),self.rows_for('validation')
        self.assertEqual((len(t),len(v)),(500,100))
        self.assertEqual(len({r['upstream_repo'] for r in t}),28)
        self.assertEqual(len({r['upstream_repo'] for r in t+v}),32)
        self.assertFalse({r['duplicate_group_id'] for r in t}&{r['duplicate_group_id'] for r in v})
        self.assertEqual([r['instance_id'] for r in t[:2]],SPEC['smoke_instance_ids'])
        self.assertEqual(len(h.read_jsonl(self.plan/'branch_task_slots.jsonl')),100)
        self.assertEqual(len(h.read_jsonl(self.plan/'final_reserved.ids.jsonl')),100)
    def test_exact_source_quotas(self):
        for q in SPEC['real_quotas']:
            for split in ['train','validation']:
                self.assertEqual(sum(r['upstream_repo']==q['repo'] for r in self.rows_for(split)),q[split])
    def test_synthetic_validation_repos_held_out(self):
        t={r['upstream_repo'] for r in self.rows_for('train') if r['source_key']=='smith'}
        v={r['upstream_repo'] for r in self.rows_for('validation') if r['source_key']=='smith'}
        self.assertFalse(t&v);self.assertEqual(len(v),4)
    def test_input_order_invariance(self):
        random.Random(991).shuffle(self.rows);h.atomic_jsonl(self.cat,self.rows)
        out=self.root/'plan2'
        with contextlib.redirect_stdout(io.StringIO()):h.select(argparse.Namespace(catalogue=self.cat,exclude_ids=None,out=out),SPEC)
        for name in ['train.ids.txt','validation.ids.txt']:
            self.assertEqual((out/name).read_text(),(self.plan/name).read_text())
    def test_model_outcomes_do_not_select(self):
        for i,r in enumerate(self.rows): r['irrelevant_future_resolved']=bool(i%2)
        h.atomic_jsonl(self.cat,self.rows);out=self.root/'plan-outcome'
        with contextlib.redirect_stdout(io.StringIO()):h.select(argparse.Namespace(catalogue=self.cat,exclude_ids=None,out=out),SPEC)
        self.assertEqual((out/'train.ids.txt').read_text(),(self.plan/'train.ids.txt').read_text())
    def test_alias_group_exposure_exclusion(self):
        target=next(r for r in self.rows if r['source_key']=='gym' and r['upstream_repo']=='python/mypy')
        alias=copy.deepcopy(target);alias['instance_id']='fixture_alias';alias['task_uid']='gym:fixture_alias';self.rows.append(alias)
        ex=self.root/'exclude.txt';ex.write_text('fixture_alias\n');h.atomic_jsonl(self.cat,self.rows);out=self.root/'plan-excluded'
        with contextlib.redirect_stdout(io.StringIO()):h.select(argparse.Namespace(catalogue=self.cat,exclude_ids=ex,out=out),SPEC)
        selected=h.read_jsonl(out/'train.provisional.jsonl')+h.read_jsonl(out/'validation.provisional.jsonl')+h.read_jsonl(out/'reserves.jsonl')
        self.assertNotIn(target['task_uid'],{r['task_uid'] for r in selected})
    def test_shortfall_fails_instead_of_inventing(self):
        h.atomic_jsonl(self.cat,[r for r in self.rows if r['upstream_repo']!='pallets/click']);out=self.root/'short'
        with self.assertRaises(RuntimeError),contextlib.redirect_stdout(io.StringIO()):
            h.select(argparse.Namespace(catalogue=self.cat,exclude_ids=None,out=out),SPEC)
        self.assertEqual(json.loads((out/'manifest_report.json').read_text())['state'],'BLOCKED_SHORTFALL')
    def test_source_revision_mismatch(self):
        self.rows[0]['dataset_revision']='wrong';h.atomic_jsonl(self.cat,self.rows)
        with self.assertRaises(ValueError):h.select(argparse.Namespace(catalogue=self.cat,exclude_ids=None,out=self.root/'wrong'),SPEC)
    def test_finalise_smoke_then_all(self):
        ledger=self.ledger();smoke=self.finish(ledger=ledger)
        self.assertEqual(len(h.read_jsonl(smoke/'train.jsonl')),2)
        full=self.finish(stage='all600',ledger=ledger,out='all',previous=smoke)
        self.assertEqual(len(h.read_jsonl(full/'train.jsonl')),500)
        self.assertEqual(len(h.read_jsonl(full/'validation.jsonl')),100)
    def test_pending_does_not_freeze(self):
        uid=self.rows_for('train')[0]['task_uid'];ledger=self.ledger({uid:{'status':'pending','reason':'transient_network'}})
        with self.assertRaises(RuntimeError):self.finish(ledger=ledger)
        self.assertFalse((self.root/'freeze'/'FROZEN.json').exists())
    def test_semantic_failure_not_preflight_replacement(self):
        uid=self.rows_for('train')[0]['task_uid'];ledger=self.ledger({uid:{'status':'reject','reason':'agent_failed'}})
        with self.assertRaises(RuntimeError):self.finish(ledger=ledger)
    def test_preflight_replacement_same_repo_split(self):
        original=self.rows_for('train')[0]
        ledger=self.ledger({original['task_uid']:{'status':'reject','reason':'task_sanity_failed'}})
        out=self.finish(ledger=ledger);new=h.read_jsonl(out/'train.jsonl')[0]
        self.assertNotEqual(new['task_uid'],original['task_uid'])
        for k in ['split','source_key','upstream_repo','slot_id']:self.assertEqual(new[k],original[k])
    def test_previous_frozen_assignment_immutable(self):
        old=self.finish();uid=self.rows_for('train')[0]['task_uid'];ledger=self.ledger({uid:{'status':'reject','reason':'task_sanity_failed'}})
        with self.assertRaises(ValueError):self.finish(stage='train50',ledger=ledger,out='later',previous=old)
    def test_bad_runtime_rejected(self):
        uid=self.rows_for('train')[0]['task_uid'];ledger=self.ledger({uid:{'warm_grade_seconds':-1}})
        with self.assertRaises(ValueError):self.finish(ledger=ledger)
    def test_source_field_semantics(self):
        r=fixture_row('smith','pallets/click',3)
        self.assertEqual(r['upstream_repo'],'pallets/click');self.assertEqual(r['fail_to_pass_count'],1)
        self.assertEqual(r['setup_semantics'],'apply_official_bug_patch_or_task_branch')
        self.assertNotIn('patch',r);self.assertNotIn('FAIL_TO_PASS',r)
        self.assertEqual(h.sequence('["one"]'),['one'])
    def test_non_python_or_test_mutation_excluded(self):
        for path in ['pkg/thing.js','tests/test_a.py','pkg/conftest.py','setup.py']:
            r=fixture_row('smith','pallets/click',3);r['changed_files']=[path]
            self.assertFalse(h.eligible(r,SPEC)[0])

if __name__=='__main__':unittest.main()
