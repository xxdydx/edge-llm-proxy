#!/usr/bin/env python3
"""Deterministic task catalogue/manifest builder. No LLM, Docker or benchmark execution.

Commands: fetch, select, finalise. Frozen public task inputs stay outside /testbed.
A selection is NOT runtime-verified. `finalise` consumes the engineering pipeline's
preflight ledger. It never replaces a task on the basis of an agent's final outcome.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def jd(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(jd(row) + "\n")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sequence(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Test list is not a valid JSON list") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Unexpected test-list type {type(value)}")
    return list(value)


def upstream(row: dict, source: str) -> str:
    repo = row.get("repo", "").strip()
    if source != "smith":
        return repo.lower().removesuffix(".git")
    # Official smith mirror: swesmith/owner__repo.<commit>. Preserve full mirror
    # identity separately; only use this canonicalisation for grouping/quotas.
    mirror = repo.split("/")[-1]
    m = re.fullmatch(r"(.+)__([^/]+)\.([0-9a-fA-F]{7,40})", mirror)
    if not m:
        raise ValueError(f"Unsupported SWE-smith mirror identity: {repo!r}")
    return f"{m[1]}/{m[2]}".lower()


def changed_files(patch: str) -> list[str]:
    return sorted(set(re.findall(r"^diff --git a/(.+?) b/[^\n]+$", patch, re.M)))


def mutation_locations(patch: str) -> list[str]:
    """Conservative review keys, not proof that all semantic near-duplicates match."""
    current = ""
    result = []
    for line in patch.splitlines():
        m = re.match(r"diff --git a/(.+?) b/", line)
        if m:
            current = m[1]
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@(.*)", line)
        if m:
            context = m[2].strip()
            result.append(f"{current}|{context or ('old40bin:'+str(int(m[1])//40))}")
    return sorted(set(result))


def source_meta(row: dict, key: str, src: dict, locator: dict) -> dict:
    repo = upstream(row, key)
    issue = str(row.get("problem_statement") or "").strip()
    patch = str(row.get("patch") or "")
    f2p, p2p = sequence(row.get("FAIL_TO_PASS")), sequence(row.get("PASS_TO_PASS"))
    normal_patch = "\n".join(x for x in patch.splitlines() if not x.startswith("index "))
    return {
        "source_key":key, "dataset_id":src["dataset_id"], "dataset_revision":src["revision"],
        "source_split":src["split"], "instance_id":str(row["instance_id"]),
        "task_uid":f"{key}:{row['instance_id']}", "upstream_repo":repo,
        "source_repo":row.get("repo"), "base_commit":row.get("base_commit"),
        "image_name":row.get("image_name"), "version":row.get("version"),
        "problem_statement":issue, "problem_sha256":sha(re.sub(r"\s+", " ", issue)),
        "patch_sha256":sha(normal_patch), "fail_to_pass_count":len(f2p),
        "pass_to_pass_count":len(p2p), "changed_files":changed_files(patch),
        "mutation_locations":mutation_locations(patch) if key == "smith" else [],
        "source_locator":locator,
        "setup_semantics":"apply_official_bug_patch_or_task_branch" if key == "smith" else "official_base_checkout",
        "runtime_preflight_status":"not_run", "selected_using_model_outcome":False,
    }


def fetch(args: argparse.Namespace, spec: dict) -> None:
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError("Install requirements-task-manifest.txt first") from e
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # Raw parquet includes protected gold/test metadata. Keep it out of an agent
    # workspace and never mount this cache into a benchmark container.
    os.chmod(cache, 0o700)
    all_meta, downloads, counts = [], [], {}
    for key, src in spec["sources"].items():
        count = 0
        for rel in src["files"]:
            print(f"Fetch {key}: {rel} @ {src['revision']}", flush=True)
            path = Path(hf_hub_download(repo_id=src["dataset_id"], repo_type="dataset",
                revision=src["revision"], filename=rel, cache_dir=str(cache / "hf")))
            digest = file_sha(path)
            downloads.append({"source_key":key,"file":rel,"sha256":digest,"bytes":path.stat().st_size})
            pf = pq.ParquetFile(path)
            wanted = [x for x in ["instance_id","repo","base_commit","image_name","version",
                "problem_statement","patch","FAIL_TO_PASS","PASS_TO_PASS"] if x in pf.schema_arrow.names]
            row_idx = 0
            for batch in pf.iter_batches(batch_size=8, columns=wanted):
                for row in batch.to_pylist():
                    loc = {"parquet_file":rel,"row_index_in_file":row_idx,"parquet_sha256":digest}
                    all_meta.append(source_meta(row,key,src,loc))
                    row_idx += 1
                    count += 1
        counts[key] = count
        if count != src["published_rows"]:
            raise RuntimeError(f"Pinned {key}: expected {src['published_rows']} rows, obtained {count}; inspect source/config; do not guess")
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.out / "catalogue.jsonl", all_meta)
    atomic_json(args.out / "source_downloads.json", {"plan_sha256":sha(jd(spec)),"counts":counts,"files":downloads})
    print(f"Wrote {len(all_meta)} source records to {args.out/'catalogue.jsonl'}; no tasks executed")


class DSU:
    def __init__(self, n: int): self.p=list(range(n))
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self, a: int,b: int) -> None:
        a,b=self.find(a),self.find(b)
        if a != b: self.p[max(a,b)]=min(a,b)


def groups(rows: list[dict]) -> list[list[dict]]:
    dsu, known = DSU(len(rows)), {}
    for i,r in enumerate(rows):
        keys = [f"id|{r['upstream_repo']}|{r['instance_id']}",
                f"issue|{r['upstream_repo']}|{r['problem_sha256']}",
                f"patch|{r['upstream_repo']}|{r['patch_sha256']}"]
        if r["source_key"] == "smith" and r["mutation_locations"]:
            keys.append(f"locations|{r['upstream_repo']}|{jd(r['mutation_locations'])}")
        for k in keys:
            if k in known: dsu.union(i,known[k])
            else: known[k]=i
    result = collections.defaultdict(list)
    for i,r in enumerate(rows): result[dsu.find(i)].append(r)
    return list(result.values())


def eligible(r: dict, spec: dict) -> tuple[bool,str]:
    s=spec["selection"]
    if len(r["problem_statement"]) < s["minimum_problem_characters"]: return False,"missing_or_short_issue"
    if r["fail_to_pass_count"] == 0: return False,"missing_failing_tests"
    if r["source_key"] == "smith":
        if not r["image_name"]: return False,"missing_image"
        if not r["pass_to_pass_count"]: return False,"no_regression_tests"
        files=r["changed_files"]
        if not files or len(files)>s["synthetic_max_changed_files"]: return False,"synthetic_changed_file_scope"
        if any(not p.endswith(".py") for p in files): return False,"non_python_mutation"
        if any(re.search(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]+|(^|/)conftest\.py$|(^|/)(setup|noxfile)\.py$",p) for p in files):
            return False,"test_or_build_mutation"
    return True,"eligible_metadata"


def rank(r: dict,spec: dict,lite: set[str]) -> tuple:
    priority=0 if r["source_key"] == "gym" and r["instance_id"] in lite else 1
    return priority,sha(f"{spec['seed']}|task-order|{r['source_key']}|{r['instance_id']}")


def interleave(rows: list[dict], spec: dict) -> list[dict]:
    buckets=collections.defaultdict(list)
    for r in rows: buckets[r["upstream_repo"]].append(r)
    for b in buckets.values(): b.sort(key=lambda r:r["source_rank"])
    # Fixed repository interleaving, no outcome-driven order.
    repos=sorted(buckets,key=lambda r:sha(f"{spec['seed']}|repo-order|{r}"))
    out=[]
    for i in range(max([len(b) for b in buckets.values()]+[0])):
        out.extend(buckets[k][i] for k in repos if i < len(buckets[k]))
    # Smoke tasks are deliberately early integration work and remain train only.
    smoke=spec["smoke_instance_ids"]
    front=sorted([r for r in out if r["instance_id"] in smoke],key=lambda r:smoke.index(r["instance_id"]))
    return front+[r for r in out if r["instance_id"] not in smoke]


def select(args: argparse.Namespace,spec: dict) -> None:
    all_rows=read_jsonl(args.catalogue)
    seen_ids=set()
    for row in all_rows:
        src=spec['sources'].get(row['source_key'])
        if not src or row['dataset_id']!=src['dataset_id'] or row['dataset_revision']!=src['revision']:
            raise ValueError('Catalogue source/revision does not match the supplied source lock')
        if row['task_uid'] in seen_ids: raise ValueError('Duplicate task UID in catalogue')
        seen_ids.add(row['task_uid'])
    lite={r["instance_id"] for r in all_rows if r["source_key"]=="lite"}
    blocked=set()
    if args.exclude_ids:
        blocked={x.strip() for x in args.exclude_ids.read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')}
    candidates,excluded=[],[]
    for r in all_rows:
        if r["source_key"]=="lite": continue # Membership only; do not duplicate Gym tasks.
        good,reason=eligible(r,spec)
        if good: candidates.append(r)
        else: excluded.append({"task_uid":r["task_uid"],"reason":reason})
    representatives=[]
    for members in groups(candidates):
        if any(x["instance_id"] in blocked or x["task_uid"] in blocked for x in members):
            excluded.extend({"task_uid":x["task_uid"],"reason":"prior_exposure_or_explicit_exclusion_group"} for x in members)
            continue
        members.sort(key=lambda r:rank(r,spec,lite))
        r=dict(members[0]); r["duplicate_group_id"]=sha("|".join(sorted(x['task_uid'] for x in members)))[:24]
        r["group_members"]=[x["task_uid"] for x in members]
        representatives.append(r)
        for x in members[1:]: excluded.append({"task_uid":x["task_uid"],"reason":"duplicate_or_mutation_location_group","representative":r["task_uid"]})
    buckets=collections.defaultdict(list)
    for r in representatives: buckets[(r["source_key"],r["upstream_repo"])].append(r)
    for b in buckets.values(): b.sort(key=lambda r:rank(r,spec,lite))
    quotas=[]
    for q in spec["real_quotas"]: quotas.append(("gym",q["repo"],q["train"],q["validation"]))
    quotas += [("smith",r,spec["synthetic_train_per_repo"],0) for r in spec["synthetic_train_repos"]]
    quotas += [("smith",r,0,spec["synthetic_validation_per_repo"]) for r in spec["synthetic_validation_repos"]]
    selected={"train":[],"validation":[]}; reserves=[];shortages=[];repo_report=[]
    for source,repo,ntrain,nval in quotas:
        pool=buckets.get((source,repo),[])
        # Smoke members are intentionally train examples; force inclusion before partition.
        pool=sorted(pool,key=lambda r:(0 if r['instance_id'] in spec['smoke_instance_ids'] else 1,rank(r,spec,lite)))
        need=ntrain+nval
        if len(pool)<need: shortages.append({"source":source,"repo":repo,"required":need,"eligible_unique_groups":len(pool)})
        primary=pool[:need]
        nonsmoke=[r for r in primary if r['instance_id'] not in spec['smoke_instance_ids']]
        vals=sorted(nonsmoke,key=lambda r:sha(f"{spec['seed']}|validation|{r['task_uid']}"))[:nval]
        vids={r['task_uid'] for r in vals}
        for idx,r0 in enumerate(primary):
            r=dict(r0); split="validation" if r["task_uid"] in vids else "train"
            r.update({"split":split,"selection_status":"provisional_primary","source_rank":idx,
                      "sampling_stratum":"real_issue" if source=="gym" else "synthetic_repo_disjoint" if split=="validation" else "synthetic_train"})
            r["slot_id"]=f"{split}:{source}:{repo}:{len([x for x in selected[split] if x['upstream_repo']==repo])+1:03d}"
            selected[split].append(r)
        for idx,r0 in enumerate(pool[need:]):
            r=dict(r0)
            # Separate train/validation reserve ownership before observing runtime.
            owner = "validation" if nval and (not ntrain or int(sha(f"{spec['seed']}|reserve-split|{r['task_uid']}"),16)%6==0) else "train"
            r.update({"split":owner,"selection_status":"reserve_only","source_rank":need+idx})
            reserves.append(r)
        repo_report.append({"source":source,"repo":repo,"train_target":ntrain,"validation_target":nval,"eligible_groups":len(pool)})
    for split in selected:
        selected[split]=interleave(selected[split],spec)
        for i,r in enumerate(selected[split]): r['collection_order']=i+1
    active_repos={r['upstream_repo'] for rows in selected.values() for r in rows}
    vrows=[r for r in representatives if r['source_key']=='verified' and r['upstream_repo'] not in active_repos]
    vrows=sorted(vrows,key=lambda r:sha(f"{spec['seed']}|final|{r['task_uid']}"))
    # Round robin gives repository-balanced final reservations, not an official full-set score.
    for i,r in enumerate(vrows): r['source_rank']=i
    vrows=interleave(vrows,spec)
    final=[]
    for r0 in vrows[:spec['final_reserved_tasks']]:
        r=dict(r0);r.update(split='final_reserved',selection_status='reserve_do_not_execute',collection_order=len(final)+1)
        # Do not emit final issue/solution/test details into the task plan.
        r.pop('problem_statement',None);r.pop('changed_files',None);r.pop('mutation_locations',None)
        final.append(r)
    if len(final)<spec['final_reserved_tasks']: shortages.append({"source":"verified","required":spec['final_reserved_tasks'],"available":len(final)})
    for sid in spec['smoke_instance_ids']:
        if not any(r['instance_id']==sid for r in selected['train']): shortages.append({'smoke_id':sid,'reason':'not_available_as_unique_training_task'})
    branch_slots=[]
    for split,n,realn in [('train',80,48),('validation',20,12)]:
        for source,amount in [('gym',realn),('smith',n-realn)]:
            rows=[r for r in selected[split] if r['source_key']==source]
            # Diversity-first deterministic order, without results.
            rows=interleave(rows,spec)
            for r in rows[:amount]: branch_slots.append({'slot_id':r['slot_id'],'task_uid':r['task_uid'],'split':split,'origins':spec['branch_task_slots']['origins']})
    args.out.mkdir(parents=True,exist_ok=True)
    for split,rows in selected.items():
        atomic_jsonl(args.out/f'{split}.provisional.jsonl',rows)
        (args.out/f'{split}.ids.txt').write_text('\n'.join(r['instance_id'] for r in rows)+'\n')
    atomic_jsonl(args.out/'reserves.jsonl',reserves)
    atomic_jsonl(args.out/'final_reserved.ids.jsonl',final)
    atomic_jsonl(args.out/'branch_task_slots.jsonl',branch_slots)
    atomic_jsonl(args.out/'excluded.jsonl',excluded)
    smoke=[r for r in selected['train'] if r['instance_id'] in spec['smoke_instance_ids']]
    atomic_jsonl(args.out/'smoke.jsonl',smoke)
    summary={'plan_version':spec['plan_version'],'spec_sha256':sha(jd(spec)),
             'catalogue_sha256':file_sha(args.catalogue),'state':'BLOCKED_SHORTFALL' if shortages else 'PROVISIONAL_NOT_RUNTIME_VERIFIED',
             'train':len(selected['train']),'validation':len(selected['validation']),'final_reserved':len(final),
             'train_repositories':len({r['upstream_repo'] for r in selected['train']}),
             'validation_repositories':len({r['upstream_repo'] for r in selected['validation']}),
             'branch_task_slots':len(branch_slots),'reserves':len(reserves),'shortages':shortages,'repo_inventory':repo_report}
    atomic_json(args.out/'manifest_report.json',summary)
    print(json.dumps(summary,indent=2))
    if shortages: raise RuntimeError('Task supply shortfall; reports written. Do not invent IDs or change quotas silently.')


def finalise(args: argparse.Namespace,spec: dict) -> None:
    report=json.loads((args.plan/'manifest_report.json').read_text())
    if report['spec_sha256']!=sha(jd(spec)):
        raise ValueError('Plan/source lock mismatch')
    shortfall_acceptance=None
    if report['state']=='BLOCKED_SHORTFALL':
        accept_path=getattr(args,'accept_shortfall',None)
        if not accept_path or not accept_path.exists():
            raise RuntimeError('Resolve the documented metadata shortfall before freezing this campaign (pass --accept-shortfall <justification file>, or fix the real supply)')
        justification=accept_path.read_text()
        expected_marker=sha(jd(report['shortages']))
        if expected_marker not in justification:
            raise RuntimeError(f'--accept-shortfall file does not contain the exact current shortages hash ({expected_marker}); write a fresh justification tied to the current documented shortfall, do not reuse a stale one')
        shortfall_acceptance={'shortages':report['shortages'],'shortages_sha256':expected_marker,
            'justification_ref':str(accept_path),'justification_sha256':sha(justification)}
    train=read_jsonl(args.plan/'train.provisional.jsonl')
    validation=read_jsonl(args.plan/'validation.provisional.jsonl')
    all_primary=train+validation
    stage_counts={'smoke':2,'train50':50,'train200':200,'train500':500,'all600':500}
    primary=train[:stage_counts[args.stage]] + (validation if args.stage=='all600' else [])
    reserves=read_jsonl(args.plan/'reserves.jsonl')
    ledger=read_jsonl(args.preflight)
    byid={}
    for x in ledger:
        uid=x['task_uid']
        if uid in byid: raise ValueError(f'Multiple preflight records for {uid}; supply an explicitly resolved versioned ledger')
        byid[uid]=x
    used={r['task_uid'] for r in all_primary}; used_groups={r['duplicate_group_id'] for r in all_primary}
    final=[];pending=[];replacements=[]
    def state(r: dict) -> tuple[str,dict|None]:
        x=byid.get(r['task_uid'])
        if not x: return 'pending',None
        if x.get('selection_used_model_outcome') is not False: raise ValueError('Preflight must explicitly exclude experimental model outcomes')
        if x.get('status')=='pass':
            if not x.get('certificate_ref') or not x.get('image_digest'): raise ValueError('Passing preflight needs certificate and digest')
            runtime=x.get('warm_grade_seconds')
            if isinstance(runtime,bool) or not isinstance(runtime,(int,float)) or not math.isfinite(runtime) or runtime<0:
                raise ValueError('Missing/invalid measured grading time')
            if x['warm_grade_seconds']>spec['selection']['preflight_warm_grade_seconds']: return 'reject',x
            return 'pass',x
        if x.get('status')=='reject' and x.get('reason') in {'task_sanity_failed','runtime_over_limit','unsupported_architecture','network_dependency','unavailable_image','duplicate_exposure','unsupported_grader','invalid_source_metadata'}:
            return 'reject',x
        return 'pending',x # Transient network/download/preflight errors do not justify replacements.
    for r0 in primary:
        s,x=state(r0);candidate=r0;attempts=1
        if s=='reject':
            pool=sorted([r for r in reserves if r['source_key']==r0['source_key'] and r['upstream_repo']==r0['upstream_repo'] and r['split']==r0['split']],key=lambda r:r['source_rank'])
            for r in pool:
                if attempts>=spec['selection']['max_preflight_candidates_per_slot']: break
                if r['task_uid'] in used or r['duplicate_group_id'] in used_groups: continue
                attempts+=1;candidate=r;s,x=state(r)
                # Allocate even a pending candidate so two slots cannot request it.
                used.add(r['task_uid']);used_groups.add(r['duplicate_group_id'])
                if s!='reject': break
        if s=='pass':
            row=dict(candidate);row.update({k:r0[k] for k in ['slot_id','collection_order','split']})
            row['selection_status']='runtime_preflight_passed';row['runtime_preflight_status']='pass';row['preflight']=x
            final.append(row)
            if candidate['task_uid']!=r0['task_uid']: replacements.append({'slot_id':r0['slot_id'],'old':r0['task_uid'],'new':candidate['task_uid'],'reason':'predeclared_preflight_replacement'})
        else:
            pending.append({'slot_id':r0['slot_id'],'task_uid':candidate['task_uid'],'state':s,'attempts_in_current_selection':attempts})
    args.out.mkdir(parents=True,exist_ok=True)
    atomic_jsonl(args.out/'pending_preflight.jsonl',pending)
    atomic_jsonl(args.out/'replacements.jsonl',replacements)
    atomic_jsonl(args.out/'accepted.preview.jsonl',final)
    if pending:
        atomic_json(args.out/'FREEZE_BLOCKED.json',{'pending_slots':len(pending),'accepted':len(final)})
        raise RuntimeError('Final manifest not frozen: run pending preflights or report capped shortfalls; do not substitute outcomes.')
    previous=getattr(args,'previous_freeze',None)
    if previous:
        old_header=json.loads((previous/'FROZEN.json').read_text())
        if old_header['spec_sha256']!=sha(jd(spec)): raise ValueError('Earlier frozen stage used a different source lock')
        now={r['slot_id']:r for r in final}
        old_rows=read_jsonl(previous/'train.jsonl')+read_jsonl(previous/'validation.jsonl')
        for old in old_rows:
            current=now.get(old['slot_id'])
            if not current or current['task_uid']!=old['task_uid'] or current['preflight']!=old['preflight']:
                raise ValueError('An earlier frozen task or its certificate changed; do not revise after seeing outcomes')
    expected=stage_counts[args.stage]+(100 if args.stage=='all600' else 0)
    if len(final)!=expected: raise RuntimeError(f'Expected {expected} slots, received {len(final)}')
    for split in ['train','validation']:
        atomic_jsonl(args.out/f'{split}.jsonl',sorted([r for r in final if r['split']==split],key=lambda r:r['collection_order']))
    atomic_json(args.out/'FROZEN.json',{'spec_sha256':sha(jd(spec)),'preflight_sha256':file_sha(args.preflight),
        'train_sha256':file_sha(args.out/'train.jsonl'),'validation_sha256':file_sha(args.out/'validation.jsonl'),
        'stage':args.stage,'counts':dict(collections.Counter(r['split'] for r in final)),
        'shortfall_acceptance':shortfall_acceptance})
    print(f'Frozen {len(final)} task slots for {args.stage}; zero experimental LLM calls made by this tool.')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,default=HERE/'TASK_SOURCE_LOCK.json')
    sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fetch');f.add_argument('--cache',type=Path,required=True);f.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('select');s.add_argument('--catalogue',type=Path,required=True);s.add_argument('--exclude-ids',type=Path);s.add_argument('--out',type=Path,required=True)
    z=sub.add_parser('finalise');z.add_argument('--plan',type=Path,required=True);z.add_argument('--preflight',type=Path,required=True);z.add_argument('--out',type=Path,required=True);z.add_argument('--previous-freeze',type=Path);z.add_argument('--stage',choices=['smoke','train50','train200','train500','all600'],default='smoke');z.add_argument('--accept-shortfall',type=Path,help='Path to a written, versioned justification file explicitly accepting the exact documented shortages in manifest_report.json. Required to freeze any stage while report.state==BLOCKED_SHORTFALL. Does not relax filters, invent IDs, or change quotas; it only lets an already-verified real supply shortfall (e.g. one repo short of its quota) stop blocking stages that do not depend on the missing supply.')
    args=p.parse_args();spec=json.loads(args.spec.read_text())
    for target in [getattr(args,'out',None)]:
        if target and target.exists() and any(target.iterdir()):
            raise RuntimeError(f'Output is nonempty: {target}. Use a new versioned directory; do not overwrite a frozen plan.')
    {'fetch':fetch,'select':select,'finalise':finalise}[args.command](args,spec)


if __name__=='__main__':
    try: main()
    except (RuntimeError,ValueError,KeyError,OSError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);sys.exit(2)
