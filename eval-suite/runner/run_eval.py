#!/usr/bin/env python3
"""Run eval-suite tasks through isolated cloud-only and routing edgeproxy instances.

Generalizes scripts/run_fanout_policy_pair.sh from one fixed fan-out prompt to a
suite of tasks with predeclared programmatic checkers, run over multiple seeds
per (task, condition) cell so pass rates come with a confidence interval instead
of resting on n=1.

Each job (task x condition x seed) gets its own edgeproxy instance, port, trace
folder under the project's top-level traces/ directory, and sandbox copy of the
task's fixture, so cloud and routing jobs - and different tasks - can run
concurrently without interfering with each other. Trace folders are named
<suite-name>-run-<condition>-<run-stamp>-<task-id>-seed<seed>, e.g.
eval-suite-1-run-cloud-20260830T120000Z-explore-pricing-api-seed1, matching the
project's existing convention of keeping every recorded trace under traces/
rather than inside a tool-specific results tree.

The "routing" condition still shares one real vLLM engine on one GPU, so its
jobs default to running one at a time (--local-parallelism 1); raise that only
if you are deliberately studying concurrency effects.

Usage:
    python3 eval-suite/runner/run_eval.py \\
        --repo-dir . --tasks all --conditions cloud,routing --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

RUNNER_DIR = Path(__file__).resolve().parent
EVAL_SUITE_DIR = RUNNER_DIR.parent

sys.path.insert(0, str(RUNNER_DIR))
from stats import binomial_rate_ci  # noqa: E402
from task_spec import TaskSpec, discover_tasks, select_tasks  # noqa: E402

CONDITION_POLICY = {"cloud": "cloud-only", "routing": "static"}


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_health(port: int, proc: subprocess.Popen, deadline_s: float = 30) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.2)
    return False


def wait_for_trace(trace_source: Path, deadline_s: float = 30) -> bool:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if trace_source.exists() and trace_source.stat().st_size > 0:
            return True
        time.sleep(0.2)
    return False


@dataclass
class RunContext:
    repo_dir: Path
    results_dir: Path
    traces_root: Path
    suite_name: str
    python_bin: str
    claude_bin: str
    claude_model: str
    vllm_url: str
    upstream: str
    max_local_tokens: int
    local_token_margin: float
    experiment_id: str
    run_stamp: str
    claude_timeout_extra_s: int = 30


@dataclass
class Job:
    task: TaskSpec
    condition: str
    seed: int

    @property
    def job_id(self) -> str:
        return f"{self.task.id}__{self.condition}__seed{self.seed}"

    def trace_dir_name(self, ctx: "RunContext") -> str:
        """Name of this job's trace folder directly under the project's traces/ dir.

        e.g. eval-suite-1-run-cloud-20260830T120000Z-explore-pricing-api-seed1 -
        "eval-suite-1" identifies this suite (as opposed to any future eval
        suite), matching the rest of the project's convention of keeping every
        recorded trace under the top-level, gitignored traces/ directory rather
        than buried inside a tool-specific results tree.
        """
        return f"{ctx.suite_name}-run-{self.condition}-{ctx.run_stamp}-{self.task.id}-seed{self.seed}"


@dataclass
class JobResult:
    job_id: str
    task_id: str
    category: str
    condition: str
    seed: int
    passed: bool
    score: float
    reason: str
    wall_time_s: float
    claude_returncode: int | None
    claude_timed_out: bool
    proxy_ok: bool
    error: str | None
    sandbox_dir: str
    trace_path: str | None
    graph_path: str | None
    verdict_path: str | None


def build_proxy_cmd(ctx: RunContext, condition: str, port: int, trace_dir: Path, episode_id: str) -> list[str]:
    policy = CONDITION_POLICY[condition]
    return [
        ctx.python_bin,
        "-m",
        "edgeproxy.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--upstream",
        ctx.upstream,
        "--vllm-url",
        ctx.vllm_url,
        "--trace-dir",
        str(trace_dir),
        "--experiment-id",
        ctx.experiment_id,
        "--episode-id",
        episode_id,
        "--cohort-tracking",
        "observe",
        "--policy",
        policy,
        "--local-cache-tracking",
        "observe",
        "--cloud-cache-tracking",
        "observe",
        "--max-local-tokens",
        str(ctx.max_local_tokens),
        "--local-token-margin",
        str(ctx.local_token_margin),
        "--local-output-reserve-tokens",
        "0",
        "--shaping",
        "none",
    ]


def start_proxy(ctx: RunContext, job: Job, port: int, trace_dir: Path, log_path: Path):
    trace_dir.mkdir(parents=True, exist_ok=True)
    episode_id = f"{ctx.experiment_id}-{job.job_id}"
    cmd = build_proxy_cmd(ctx, job.condition, port, trace_dir, episode_id)
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ctx.repo_dir, stdout=log_fh, stderr=subprocess.STDOUT)
    if not wait_for_health(port, proc):
        proc.kill()
        proc.wait(timeout=10)
        log_fh.close()
        raise RuntimeError(f"{job.job_id}: edgeproxy did not become healthy; see {log_path}")
    return proc, log_fh


def stop_proxy(proc: subprocess.Popen, log_fh) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    log_fh.close()


def build_user_event(prompt: str) -> dict[str, Any]:
    return {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
        "parent_tool_use_id": None,
    }


def run_claude(ctx: RunContext, job: Job, sandbox_dir: Path, port: int, stream_out_path: Path):
    """Run one non-interactive Claude Code turn against the isolated proxy."""
    session_id = str(uuid.uuid4())
    stdin_line = json.dumps(build_user_event(job.task.prompt), ensure_ascii=False) + "\n"
    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    cmd = [
        ctx.claude_bin,
        "-p",
        "--model",
        ctx.claude_model,
        "--session-id",
        session_id,
        "--dangerously-skip-permissions",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--replay-user-messages",
        "--verbose",
    ]
    timed_out = False
    stdout = ""
    returncode: int | None = None
    try:
        proc = subprocess.run(
            cmd,
            cwd=sandbox_dir,
            env=env,
            input=stdin_line,
            capture_output=True,
            text=True,
            timeout=job.task.timeout_s + ctx.claude_timeout_extra_s,
        )
        stdout, returncode = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
    stream_out_path.write_text(stdout, encoding="utf-8")
    return stdout, returncode, timed_out, session_id


def read_stream_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def extract_report_text(task: TaskSpec, events: list[dict[str, Any]]) -> str:
    if not task.expects_report:
        return ""
    from edgeproxy.report_capture import assemble_parent_report

    return assemble_parent_report(events)


def collect_trace(ctx: RunContext, trace_dir: Path, claude_stream_path: Path):
    """Rename edgeproxy's date-stamped raw trace and add graph/Mermaid sidecars,
    all inside `trace_dir` (the job's own folder under the project's traces/)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    trace_source = trace_dir / f"{today}.jsonl"
    if not wait_for_trace(trace_source, deadline_s=15):
        return None, None
    trace_dest = trace_dir / "trace.jsonl"
    if trace_source != trace_dest:
        shutil.copy(trace_source, trace_dest)
    graph_dest = trace_dir / "trace.graph.json"
    mermaid_dest = trace_dir / "trace.mmd"
    subprocess.run(
        [
            ctx.python_bin,
            "-m",
            "edgeproxy.trace.graph",
            str(trace_dest),
            "--claude-stream",
            str(claude_stream_path),
            "--json-output",
            str(graph_dest),
            "--mermaid-output",
            str(mermaid_dest),
        ],
        cwd=ctx.repo_dir,
        capture_output=True,
        text=True,
    )
    return trace_dest, (graph_dest if graph_dest.exists() else None)


def run_checker(
    python_bin: str, task: TaskSpec, sandbox_dir: Path, report_path: Path, verdict_path: Path
) -> dict[str, Any]:
    cmd = [
        python_bin,
        str(task.checker_path),
        "--sandbox",
        str(sandbox_dir),
        "--fixture",
        str(task.fixture_dir),
        "--report",
        str(report_path),
        "--out",
        str(verdict_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if verdict_path.exists():
        return json.loads(verdict_path.read_text(encoding="utf-8"))
    return {
        "passed": False,
        "score": 0.0,
        "reason": f"checker produced no verdict (exit {proc.returncode}): {proc.stderr[-2000:]}",
        "details": {},
    }


def run_job(ctx: RunContext, job: Job) -> JobResult:
    started = time.monotonic()
    job_dir_name = job.job_id
    sandbox_dir = ctx.results_dir / "sandboxes" / job_dir_name
    trace_dir = ctx.traces_root / job.trace_dir_name(ctx)
    log_path = ctx.results_dir / "logs" / f"{job_dir_name}.proxy.log"
    stream_path = ctx.results_dir / "logs" / f"{job_dir_name}.stream.jsonl"
    report_path = ctx.results_dir / "reports" / f"{job_dir_name}.md"
    verdict_path = ctx.results_dir / "verdicts" / f"{job_dir_name}.json"

    for p in (sandbox_dir.parent, log_path.parent, stream_path.parent,
              report_path.parent, verdict_path.parent):
        p.mkdir(parents=True, exist_ok=True)

    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    shutil.copytree(job.task.fixture_dir, sandbox_dir)

    proxy_ok = False
    error: str | None = None
    returncode: int | None = None
    timed_out = False
    verdict = {"passed": False, "score": 0.0, "reason": "job did not complete", "details": {}}
    trace_dest = graph_dest = None

    port = free_port()
    proc = log_fh = None
    try:
        proc, log_fh = start_proxy(ctx, job, port, trace_dir, log_path)
        proxy_ok = True
        stdout, returncode, timed_out, _session_id = run_claude(ctx, job, sandbox_dir, port, stream_path)
        events = read_stream_events(stdout)
        report_text = extract_report_text(job.task, events)
        report_path.write_text(report_text, encoding="utf-8")
        trace_dest, graph_dest = collect_trace(ctx, trace_dir, stream_path)
        verdict = run_checker(ctx.python_bin, job.task, sandbox_dir, report_path, verdict_path)
    except Exception as exc:  # noqa: BLE001 - one bad job must not sink the campaign
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None and log_fh is not None:
            stop_proxy(proc, log_fh)

    return JobResult(
        job_id=job.job_id,
        task_id=job.task.id,
        category=job.task.category,
        condition=job.condition,
        seed=job.seed,
        passed=bool(verdict.get("passed", False)) and error is None,
        score=float(verdict.get("score", 0.0)) if error is None else 0.0,
        reason=error or verdict.get("reason", ""),
        wall_time_s=round(time.monotonic() - started, 2),
        claude_returncode=returncode,
        claude_timed_out=timed_out,
        proxy_ok=proxy_ok,
        error=error,
        sandbox_dir=str(sandbox_dir),
        trace_path=str(trace_dest) if trace_dest else None,
        graph_path=str(graph_dest) if graph_dest else None,
        verdict_path=str(verdict_path) if verdict_path.exists() else None,
    )


def build_jobs(tasks: list[TaskSpec], conditions: list[str], seeds: int) -> list[Job]:
    return [
        Job(task=task, condition=condition, seed=seed)
        for task in tasks
        for condition in conditions
        for seed in range(1, seeds + 1)
    ]


def run_campaign(ctx: RunContext, jobs: list[Job], cloud_parallelism: int, local_parallelism: int) -> list[JobResult]:
    results: list[JobResult] = []
    lock = Lock()
    total = len(jobs)
    done = 0

    def on_done(job: Job, future) -> None:
        nonlocal done
        result = future.result()
        with lock:
            results.append(result)
            done += 1
            status = "PASS" if result.passed else "FAIL"
            print(f"[{done}/{total}] {status}  {result.job_id}  ({result.wall_time_s}s)  {result.reason[:120]}")

    cloud_jobs = [j for j in jobs if j.condition == "cloud"]
    routing_jobs = [j for j in jobs if j.condition == "routing"]

    with ThreadPoolExecutor(max_workers=max(cloud_parallelism, 1)) as cloud_pool, \
         ThreadPoolExecutor(max_workers=max(local_parallelism, 1)) as local_pool:
        futures = {}
        for job in cloud_jobs:
            fut = cloud_pool.submit(run_job, ctx, job)
            futures[fut] = job
        for job in routing_jobs:
            fut = local_pool.submit(run_job, ctx, job)
            futures[fut] = job

        for fut in as_completed(futures):
            on_done(futures[fut], fut)

    return results


def aggregate(results: list[JobResult]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[JobResult]] = {}
    for r in results:
        cells.setdefault((r.task_id, r.condition), []).append(r)

    rows = []
    for (task_id, condition), cell_results in sorted(cells.items()):
        passes = sum(1 for r in cell_results if r.passed)
        n = len(cell_results)
        estimate = binomial_rate_ci(passes, n)
        mean_score = sum(r.score for r in cell_results) / n if n else 0.0
        mean_wall_time = sum(r.wall_time_s for r in cell_results) / n if n else 0.0
        rows.append(
            {
                "task_id": task_id,
                "condition": condition,
                "n": n,
                "passes": passes,
                "pass_rate": estimate.rate,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "mean_score": round(mean_score, 4),
                "mean_wall_time_s": round(mean_wall_time, 2),
            }
        )
    return rows


def write_summary(ctx: RunContext, results: list[JobResult], rows: list[dict[str, Any]]) -> None:
    summary = {
        "experiment_id": ctx.experiment_id,
        "run_stamp": ctx.run_stamp,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [asdict(r) for r in results],
        "cells": rows,
    }
    (ctx.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], {})[row["condition"]] = row

    lines = [
        f"# Eval-suite campaign {ctx.experiment_id}",
        "",
        f"Generated {summary['generated_at']}",
        "",
        "| task | condition | n | pass rate | 95% CI | mean score | mean wall time (s) |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for task_id in sorted(by_task):
        for condition in ("cloud", "routing"):
            row = by_task[task_id].get(condition)
            if not row:
                continue
            lines.append(
                f"| {task_id} | {condition} | {row['n']} | {row['pass_rate']:.2f} "
                f"| [{row['ci_low']:.2f}, {row['ci_high']:.2f}] | {row['mean_score']:.2f} "
                f"| {row['mean_wall_time_s']:.1f} |"
            )
    (ctx.results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n==> wrote {ctx.results_dir / 'summary.json'}")
    print(f"==> wrote {ctx.results_dir / 'summary.md'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-dir", type=Path, default=EVAL_SUITE_DIR.parent)
    parser.add_argument("--tasks", default="all", help="'all' or a comma-separated list of task ids")
    parser.add_argument("--conditions", default="cloud,routing", help="comma-separated: cloud,routing")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cloud-parallelism", type=int, default=4)
    parser.add_argument("--local-parallelism", type=int, default=1, help="keep at 1 unless deliberately studying GPU concurrency")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    parser.add_argument("--claude-model", default=os.environ.get("CLAUDE_MODEL", "sonnet"))
    parser.add_argument("--vllm-url", default=os.environ.get("EDGEPROXY_VLLM_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--upstream", default=os.environ.get("EDGEPROXY_UPSTREAM", "https://lum.id/claude"))
    parser.add_argument("--max-local-tokens", type=int, default=int(os.environ.get("EDGEPROXY_MAX_LOCAL_TOKENS", "100000")))
    parser.add_argument("--local-token-margin", type=float, default=float(os.environ.get("EDGEPROXY_LOCAL_TOKEN_MARGIN", "0.90")))
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--suite-name",
        default="eval-suite-1",
        help="identifies this suite in trace folder names, distinct from any future eval suite",
    )
    parser.add_argument(
        "--traces-root",
        type=Path,
        default=None,
        help="where per-job trace folders go; defaults to <repo-dir>/traces",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_dir = args.repo_dir.resolve()
    if not (repo_dir / "edgeproxy").is_dir():
        die(f"expected edgeproxy/ under {repo_dir}")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITION_POLICY:
            die(f"unknown condition: {c} (expected cloud and/or routing)")

    tasks_root = EVAL_SUITE_DIR / "tasks"
    all_tasks = discover_tasks(tasks_root)
    if not all_tasks:
        die(f"no tasks found under {tasks_root}")
    tasks = select_tasks(all_tasks, args.tasks)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = args.experiment_id or f"eval-suite-{run_stamp}"
    results_dir = args.results_dir or (EVAL_SUITE_DIR / "results" / f"{experiment_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_root = args.traces_root or (repo_dir / "traces")
    traces_root.mkdir(parents=True, exist_ok=True)

    # `python -m edgeproxy.server` and `edgeproxy.trace.graph` must resolve the
    # checked-out package regardless of where this script is invoked from.
    sys.path.insert(0, str(repo_dir))

    ctx = RunContext(
        repo_dir=repo_dir,
        results_dir=results_dir,
        traces_root=traces_root,
        suite_name=args.suite_name,
        python_bin=args.python_bin,
        claude_bin=args.claude_bin,
        claude_model=args.claude_model,
        vllm_url=args.vllm_url,
        upstream=args.upstream,
        max_local_tokens=args.max_local_tokens,
        local_token_margin=args.local_token_margin,
        experiment_id=experiment_id,
        run_stamp=run_stamp,
    )

    jobs = build_jobs(tasks, conditions, args.seeds)
    print(f"==> experiment: {experiment_id}")
    print(f"==> tasks: {', '.join(t.id for t in tasks)}")
    print(f"==> conditions: {', '.join(conditions)}  seeds: {args.seeds}  jobs: {len(jobs)}")
    print(f"==> results: {results_dir}")
    print(f"==> traces:  {traces_root}  (suite: {args.suite_name})")

    results = run_campaign(ctx, jobs, args.cloud_parallelism, args.local_parallelism)
    rows = aggregate(results)
    write_summary(ctx, results, rows)

    failures = [r for r in results if not r.passed]
    print(f"\n==> {len(results) - len(failures)}/{len(results)} jobs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
