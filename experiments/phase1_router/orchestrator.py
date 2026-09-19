#!/usr/bin/env python3
"""Phase 1 pilot orchestrator: dataset -> paired generation -> Docker grading
-> features -> policies -> analysis -> plots.

  python -m experiments.phase1_router.orchestrator --dataset-only
  python -m experiments.phase1_router.orchestrator --smoke-test 3
  python -m experiments.phase1_router.orchestrator --full-batch
  python -m experiments.phase1_router.orchestrator --resume-run-dir results/run-...
  python -m experiments.phase1_router.orchestrator --analysis-only --resume-run-dir results/run-...

Model code never runs on the host: generation is HTTP only; grading is Docker
only. Preflight/postflight identity checks reuse capability_router.executor.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path

from experiments.capability_router import executor

from .config import CLOUD_DEEPSEEK, LOCAL_27B, EXPERIMENT, new_run_dir
from .dataset import build_dataset, load_dataset, save_dataset, split_by_group
from .features import extract_features
from .generate import generate_one, reference_input_tokens
from .grading import grade_completion
from .plots import save_all
from .schema import GenerationOutcome, GradeResult, PairedExample, Problem
from .analysis import run_analysis

log = logging.getLogger("phase1_router")


# --- checkpoint (reuse capability_router's append-only jsonl helpers) --------

def _gen_key(task_id: str, backend: str) -> str:
    return f"{task_id}::{backend}"


def _load_gen_checkpoint(path: Path) -> dict[str, dict]:
    return executor.load_checkpoint(path) if path.exists() else {}


def _outcome_to_row(p: Problem, o: GenerationOutcome) -> dict:
    return {
        "call_id": p.task_id, "backend": o.backend, "status": o.status,
        "completion": o.completion, "raw_text": o.raw_text, "usage": o.usage,
        "detail": o.detail, "latency_s": o.latency_s,
        "retry_attempts": o.retry_attempts, "end_to_end_wall_s": o.end_to_end_wall_s,
        "final_attempt_latency_s": o.final_attempt_latency_s, "attempts_meta": o.attempts_meta,
        "stop_reason": o.stop_reason, "model_identity": o.model_identity,
    }


def _row_to_outcome(r: dict) -> GenerationOutcome:
    return GenerationOutcome(
        backend=r["backend"], status=r["status"], completion=r.get("completion"),
        raw_text=r.get("raw_text"), usage=r.get("usage"), detail=r.get("detail"),
        latency_s=r.get("latency_s"), retry_attempts=r.get("retry_attempts"),
        end_to_end_wall_s=r.get("end_to_end_wall_s"),
        final_attempt_latency_s=r.get("final_attempt_latency_s"),
        attempts_meta=r.get("attempts_meta"), stop_reason=r.get("stop_reason"),
        model_identity=r.get("model_identity"),
    )


# --- stages --------------------------------------------------------------

def _do_dataset(run_dir: Path, cfg) -> list[Problem]:
    problems, manifest, provenance = build_dataset(cfg)
    save_dataset(problems, manifest, provenance, run_dir)
    log.info("dataset: %d problems, %d groups, windows=%s  (evalplus_hash=%s)",
             manifest.selected, len(manifest.task_group_counts),
             manifest.window_counts, provenance["evalplus_mbpp_plus_hash"][:16])
    return problems


def _reference_tokens(problems: list[Problem], run_dir: Path, cfg, parallelism: int) -> dict[str, dict]:
    """One reference input-token count per task (same prompt both backends),
    from the local vLLM tokenizer where reachable, else the deterministic
    estimate. Checkpointed."""
    path = run_dir / "ref_input_tokens.jsonl"
    done: dict[str, dict] = {}
    if path.exists():
        for ln in path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                done[r["task_id"]] = r
    lock = threading.Lock()

    def one(p: Problem):
        with lock:
            if p.task_id in done:
                return
        n, method = reference_input_tokens(p, cfg)
        row = {"task_id": p.task_id, "ref_input_tokens": n, "method": method}
        with lock:
            if p.task_id in done:
                return
            with path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            done[p.task_id] = row

    with cf.ThreadPoolExecutor(max_workers=parallelism) as pool:
        list(cf.as_completed([pool.submit(one, p) for p in problems]))
    exact = sum(1 for r in done.values() if r["method"] == "vllm_tokenizer")
    log.info("reference input tokens: %d/%d via vllm_tokenizer, rest estimated",
             exact, len(done))
    return done


ALL_BACKENDS = ("cloud_deepseek", "local_27b")


def _generate_all(problems, run_dir: Path, cfg, gen_parallelism: int, backends) -> dict[str, dict]:
    cp_path = run_dir / "generations.jsonl"
    cp = executor.retryable_checkpoint(_load_gen_checkpoint(cp_path))
    lock = threading.Lock()
    log.info("generation resume: %d rows terminal (transport errors retried); backends=%s",
             len(cp), ",".join(backends))

    def one(p: Problem, backend: str):
        key = _gen_key(p.task_id, backend)
        with lock:
            if key in cp:
                return
        o = generate_one(p, backend, cfg)
        row = _outcome_to_row(p, o)
        with lock:
            if key in cp:
                return
            executor.append_checkpoint(cp_path, row)  # append-only
            cp[key] = row

    jobs = [(p, b) for p in problems for b in backends]
    with cf.ThreadPoolExecutor(max_workers=gen_parallelism) as pool:
        futs = [pool.submit(one, p, b) for p, b in jobs]
        for i, f in enumerate(cf.as_completed(futs), 1):
            f.result()
            if i % 40 == 0 or i == len(futs):
                log.info("generated %d/%d (call,backend) pairs", i, len(futs))
    return _load_gen_checkpoint(cp_path)


def _grade_all(problems, gens: dict[str, dict], run_dir: Path, cfg, parallelism: int, backends) -> dict[str, dict]:
    cp_path = run_dir / "grades.jsonl"
    done = _load_gen_checkpoint(cp_path)
    lock = threading.Lock()

    def _stale(key: str) -> bool:
        # a grade is stale if a later resume superseded a TRANSPORT_ERROR
        # generation with a real completion but the grade still reflects the
        # old (no-completion) NO_CODE/ERROR outcome.
        gr = done.get(key)
        g = gens.get(key)
        return bool(
            gr and g and g.get("status") == "OK"
            and (g.get("completion") or "").strip()
            and gr.get("status") in ("NO_CODE", "ERROR")
        )

    def one(p: Problem, backend: str):
        key = _gen_key(p.task_id, backend)
        with lock:
            if key in done and not _stale(key):
                return
        g = gens.get(key)
        code = (g or {}).get("completion") if (g or {}).get("status") == "OK" else None
        gr = grade_completion(p, code, cfg)
        row = {"call_id": p.task_id, "backend": backend, **asdict(gr)}
        with lock:
            if key in done and not _stale(key):
                return
            executor.append_checkpoint(cp_path, row)  # append-only, supersedes
            done[key] = row

    jobs = [(p, b) for p in problems for b in backends if _gen_key(p.task_id, b) in gens]
    with cf.ThreadPoolExecutor(max_workers=parallelism) as pool:
        futs = [pool.submit(one, p, b) for p, b in jobs]
        for i, f in enumerate(cf.as_completed(futs), 1):
            f.result()
            if i % 40 == 0 or i == len(futs):
                log.info("graded %d/%d completions", i, len(futs))
    return _load_gen_checkpoint(cp_path)


def _assemble(problems, gens, grades) -> list[PairedExample]:
    out = []
    for p in problems:
        lg = _row_to_outcome(gens[_gen_key(p.task_id, "local_27b")])
        cg = _row_to_outcome(gens[_gen_key(p.task_id, "cloud_deepseek")])
        lr = grades[_gen_key(p.task_id, "local_27b")]
        cr = grades[_gen_key(p.task_id, "cloud_deepseek")]
        lgr = GradeResult(**{k: lr[k] for k in GradeResult.__dataclass_fields__ if k in lr})
        cgr = GradeResult(**{k: cr[k] for k in GradeResult.__dataclass_fields__ if k in cr})
        out.append(PairedExample(
            problem=p, features=extract_features(p),
            local_gen=lg, cloud_gen=cg, local_grade=lgr, cloud_grade=cgr,
            local_ok=lgr.plus_pass, cloud_ok=cgr.plus_pass,
        ))
    return out


def _load_ref_tokens(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "ref_input_tokens.jsonl"
    out: dict[str, dict] = {}
    if path.exists():
        for ln in path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                out[r["task_id"]] = r
    return out


def _analyze(problems, gens, grades, run_dir: Path, cfg, *, is_smoke: bool = False) -> None:
    # analysis needs BOTH arms for a full, grouped split -- skip cleanly if
    # this is a smoke run or an arm has not finished yet.
    if is_smoke:
        log.info("smoke run: skipping analysis (inspect the per-row generation/grade output above)")
        return
    if len(problems) < cfg.n_tasks:
        log.warning("skipping analysis: only %d/%d problems present (partial run)", len(problems), cfg.n_tasks)
        return
    missing = [
        p.task_id for p in problems
        for b in ("cloud_deepseek", "local_27b")
        if _gen_key(p.task_id, b) not in gens or _gen_key(p.task_id, b) not in grades
    ]
    if missing:
        log.warning(
            "skipping analysis: %d (problem,backend) gen/grade rows still missing "
            "(e.g. %s). Re-run --analysis-only once both arms are complete.",
            len(missing), missing[:4],
        )
        return
    ref_tokens = _load_ref_tokens(run_dir)
    paired = _assemble(problems, gens, grades)
    by_id = {ex.problem.task_id: ex for ex in paired}
    tr_p, va_p, te_p = split_by_group(problems, cfg)
    tr = [by_id[p.task_id] for p in tr_p]
    va = [by_id[p.task_id] for p in va_p]
    te = [by_id[p.task_id] for p in te_p]
    ordered = sorted(paired, key=lambda ex: ex.problem.order)

    report = run_analysis(tr, va, te, ordered, cfg, ref_tokens=ref_tokens)
    # exec/label status summary
    import collections
    for arm in ("local", "cloud"):
        gs = collections.Counter(getattr(ex, f"{arm}_gen").status for ex in paired)
        grs = collections.Counter(getattr(ex, f"{arm}_grade").status for ex in paired)
        oks = sum(1 for ex in paired if getattr(ex, f"{arm}_ok"))
        report.setdefault("execution", {})[arm] = {
            "gen_status": dict(gs), "grade_status": dict(grs),
            "plus_pass": oks, "plus_pass_rate": oks / len(paired) if paired else None,
        }
    (run_dir / "analysis_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    made = save_all(report, run_dir)
    (run_dir / "validity_manifest.json").write_text(json.dumps({
        "n_problems": len(problems),
        "splits": {"train": len(tr), "val": len(va), "test": len(te)},
        "contamination_gate": {
            "cloud_gen_all_identity_ok": all(ex.cloud_gen.model_identity == "deepseek-v4-flash"
                                             for ex in paired if ex.cloud_gen.status == "OK"),
            "local_gen_all_identity_ok": all(ex.local_gen.model_identity == "local"
                                             for ex in paired if ex.local_gen.status == "OK"),
        },
        "existence_verdict": report["existence_verdict"],
        "plots": made,
    }, indent=2) + "\n")
    log.info("wrote analysis_report.json, validity_manifest.json, %d plot groups", len(made))
    ev = report["existence_verdict"]
    log.info("EXISTENCE: oracle local_share=%.3f  quality_delta_vs_cloud=%.3f  meaningful=%s",
             ev["oracle_local_share"], ev["oracle_quality_delta_vs_cloud"],
             ev["meaningful_local_traffic_at_cloud_quality"])


# --- entrypoint --------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-only", action="store_true")
    ap.add_argument("--smoke-test", type=int, metavar="N", default=0)
    ap.add_argument("--full-batch", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--resume-run-dir", type=Path, default=None)
    ap.add_argument("--gen-parallelism", type=int, default=4)
    ap.add_argument("--grade-parallelism", type=int, default=6)
    ap.add_argument(
        "--backends", default="cloud_deepseek,local_27b",
        help="comma list; run only these arms. Preflight/postflight and reference-token "
             "probing are skipped for backends not listed, so the cloud arm can run "
             "before the local box is up.",
    )
    args = ap.parse_args(argv)
    cfg = EXPERIMENT
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    bad = [b for b in backends if b not in ALL_BACKENDS]
    if bad:
        raise SystemExit(f"unknown backend(s): {bad}; valid: {ALL_BACKENDS}")

    run_dir = args.resume_run_dir
    if run_dir is None:
        run_dir = new_run_dir()
        log.info("run_dir: %s", run_dir)
    elif not run_dir.is_dir():
        raise SystemExit(f"--resume-run-dir does not exist: {run_dir}")

    problems = load_dataset(run_dir) if (run_dir / "problems.jsonl").exists() else _do_dataset(run_dir, cfg)
    if args.dataset_only:
        return 0

    if args.smoke_test:
        problems = problems[: args.smoke_test]
        log.info("SMOKE: %d problems", len(problems))

    if args.analysis_only:
        gens = _load_gen_checkpoint(run_dir / "generations.jsonl")
        grades = _load_gen_checkpoint(run_dir / "grades.jsonl")
        _analyze(problems, gens, grades, run_dir, cfg, is_smoke=False)
        return 0

    if not (args.full_batch or args.smoke_test):
        log.info("nothing to do -- pass --dataset-only / --smoke-test N / --full-batch")
        return 0

    executor.load_env()
    be_objs = {"cloud_deepseek": CLOUD_DEEPSEEK, "local_27b": LOCAL_27B}
    active = [be_objs[b] for b in backends]
    for be in active:
        ev = executor.preflight(be)  # exact identity allowlist check
        log.info("preflight %s: %s model=%s", be.name, ev["status"], ev["response_model"])

    # reference input tokens use the LOCAL vLLM tokenizer -- only meaningful
    # when the local arm is running; otherwise deferred to the local pass.
    if "local_27b" in backends:
        _reference_tokens(problems, run_dir, cfg, args.gen_parallelism)

    gens = _generate_all(problems, run_dir, cfg, args.gen_parallelism, backends)
    grades = _grade_all(problems, gens, run_dir, cfg, args.grade_parallelism, backends)

    for be in active:
        ev = executor.postflight(be)
        log.info("postflight %s: %s", be.name, ev["status"])

    _analyze(problems, gens, grades, run_dir, cfg, is_smoke=bool(args.smoke_test))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
