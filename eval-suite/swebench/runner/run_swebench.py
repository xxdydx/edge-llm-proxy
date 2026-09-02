#!/usr/bin/env python3
"""Container-based runner for the SWE-bench Pro hard tier.

Different execution model from eval-suite/runner/run_eval.py: instead of a
plain host sandbox directory, each job gets a fresh Docker container built
from that instance's prebuilt jefzda/sweap-images environment (the real
repo, real dependencies, exactly as SWE-bench Pro ships it), so the agent's
own Bash tool calls run against the correct toolchain instead of an
approximation copied onto the host. Validated end-to-end against
qutebrowser/qutebrowser instance_...-e64622cd...: baseline fails the 4
official fail_to_pass tests, a live Claude Code session's own independent
fix (not the gold patch) passes all 4, verified by re-running those exact
node IDs after the session ends.

edgeproxy itself is unchanged and reused as-is (imported from run_eval.py):
each job still gets its own isolated proxy instance and trace folder under
the project's traces/ directory. What's new is entirely about *where the
agent's session runs* - inside a container, not a copied directory.

Two container-image quirks discovered and handled generically here, not
hacked around per instance:
- Image layout varies: some set ENTRYPOINT=/bin/bash (qutebrowser,
  openlibrary), others set CMD=["bash"] with no entrypoint (ansible).
  Always passing --entrypoint bash normalizes both.
- The container's default user is root, and `claude --dangerously-skip-
  permissions` refuses to run as root. A non-root "agent" user is created
  and used for every claude/checker invocation.
- qutebrowser's own pytest.ini hardcodes the pytest<7-era `--strict` flag,
  which pytest 7.4.2 in that image escalates to a fatal error during
  argument parsing (before any test runs). `--override-ini=addopts=...`
  clears it. ansible and openlibrary do not have this problem; the override
  is per-instance (instances/*.json's pytest_addopts_override), not global.

Usage:
    python3 eval-suite/swebench/runner/run_swebench.py \\
        --instances all --conditions cloud --seeds 1
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

SWEBENCH_DIR = Path(__file__).resolve().parent.parent
INSTANCES_DIR = SWEBENCH_DIR / "instances"
EVAL_SUITE_DIR = SWEBENCH_DIR.parent
REPO_ROOT = EVAL_SUITE_DIR.parent

sys.path.insert(0, str(EVAL_SUITE_DIR / "runner"))
from run_eval import (  # noqa: E402
    CONDITION_POLICY,
    RunContext,
    build_proxy_cmd,
    free_port,
    load_dotenv_into_environ,
    start_proxy,
    stop_proxy,
    wait_for_health,
)
from stats import binomial_rate_ci  # noqa: E402

CLAUDE_TIMEOUT_EXTRA_S = 60
CONTAINER_SETUP_TIMEOUT_S = 120
CHECKER_TIMEOUT_S = 300
# Real SWE-bench Pro instances (unlike our synthetic fixture tasks) can run
# long multi-turn agent loops, especially on a local model. 1800s silently
# reimposed the 30-min pipeline cap we were told to drop, and produced zero
# real pass/fail signal across a live run. Default raised to 90 min; still
# overridable per-instance via "timeout_s" in the instance JSON.
DEFAULT_CLAUDE_TIMEOUT_S = 5400


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_instances(selector: str) -> list[dict[str, Any]]:
    all_instances = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(INSTANCES_DIR.glob("*.json"))
    ]
    if selector == "all":
        return all_instances
    wanted = {s.strip() for s in selector.split(",") if s.strip()}
    selected = [i for i in all_instances if i["slug"] in wanted]
    missing = wanted - {i["slug"] for i in selected}
    if missing:
        die(f"unknown instance slug(s): {', '.join(sorted(missing))}")
    return selected


def render_prompt(instance: dict[str, Any]) -> str:
    return (
        "Your working directory is /app, a checkout of the "
        f"{instance['repo']} repository. Fix the following issue:\n\n"
        f"{instance['problem_statement']}\n\n"
        "Investigate the relevant source file(s) yourself and implement a fix. "
        "You may run the repository's own tests with pytest to check your work. "
        "Do not modify any test file. Do not run pip install or access the network."
    )


@dataclass
class Job:
    instance: dict[str, Any]
    condition: str
    seed: int

    @property
    def job_id(self) -> str:
        return f"{self.instance['slug']}__{self.condition}__seed{self.seed}"


@dataclass
class JobResult:
    job_id: str
    instance_slug: str
    repo: str
    difficulty_rank: int
    condition: str
    seed: int
    passed: bool
    reason: str
    wall_time_s: float
    n_fail_to_pass: int
    n_fail_to_pass_passed: int
    n_pass_to_pass: int
    n_pass_to_pass_passed: int
    claude_num_turns: int | None
    claude_duration_ms: int | None
    error: str | None


def docker(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, **kwargs)


def start_container(instance: dict[str, Any], container_name: str) -> None:
    proc = docker(
        [
            "run", "-d", "--name", container_name,
            "--add-host=host.docker.internal:host-gateway",
            "--entrypoint", "bash",
            instance["docker_image"],
            "-c", "sleep infinity",
        ],
        timeout=CONTAINER_SETUP_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker run failed: {proc.stderr[-2000:]}")


def exec_in(container_name: str, script: str, user: str | None = None, timeout: int = CONTAINER_SETUP_TIMEOUT_S) -> subprocess.CompletedProcess:
    cmd = ["exec"]
    if user:
        cmd += ["-u", user, "-e", "HOME=/home/" + user]
    cmd += [container_name, "bash", "-c", script]
    return docker(cmd, timeout=timeout)


def setup_container(instance: dict[str, Any], container_name: str) -> None:
    repo_setup = exec_in(container_name, f"cd /app && {instance['before_repo_set_cmd']}")
    if repo_setup.returncode != 0:
        raise RuntimeError(f"before_repo_set_cmd failed: {repo_setup.stderr[-2000:]}")

    install = exec_in(
        container_name,
        "useradd -m -s /bin/bash agent 2>/dev/null; "
        "chown -R agent:agent /app; "
        "which curl >/dev/null 2>&1 || "
        "(apt-get update -qq && apt-get install -y -qq curl ca-certificates); "
        "v=$(curl -fsSL https://downloads.claude.ai/claude-code-releases/latest) && "
        "curl -fsSL -o /usr/local/bin/claude "
        '"https://downloads.claude.ai/claude-code-releases/$v/linux-x64/claude" && '
        "chmod a+rx /usr/local/bin/claude",
        timeout=CONTAINER_SETUP_TIMEOUT_S,
    )
    if install.returncode != 0:
        raise RuntimeError(f"claude install failed: {install.stderr[-2000:]}")


def run_claude_in_container(
    ctx: "SwebenchContext", job: Job, container_name: str, port: int, stream_out_path: Path
) -> tuple[str, int | None, bool]:
    session_id = str(uuid.uuid4())
    prompt = render_prompt(job.instance)
    stdin_line = json.dumps(
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
            "parent_tool_use_id": None,
        },
        ensure_ascii=False,
    ) + "\n"

    env_flags = [
        "-e", f"ANTHROPIC_BASE_URL=http://host.docker.internal:{port}",
        "-e", f"ANTHROPIC_AUTH_TOKEN={os.environ['ANTHROPIC_AUTH_TOKEN']}",
        "-e", "HOME=/home/agent",
    ]
    for key, value in (job.instance.get("extra_env") or {}).items():
        env_flags += ["-e", f"{key}={value}"]

    cmd = [
        "docker", "exec", "-i", "-u", "agent", "-w", "/app", *env_flags, container_name,
        "claude", "-p", "--model", ctx.claude_model, "--session-id", session_id,
        "--dangerously-skip-permissions",
        "--input-format", "stream-json", "--output-format", "stream-json",
        "--replay-user-messages", "--verbose",
    ]
    timed_out = False
    stdout = ""
    returncode: int | None = None
    try:
        proc = subprocess.run(
            cmd, input=stdin_line, capture_output=True, text=True,
            timeout=job.instance.get("timeout_s", DEFAULT_CLAUDE_TIMEOUT_S) + CLAUDE_TIMEOUT_EXTRA_S,
        )
        stdout, returncode = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        raw = exc.stdout or ""
        stdout = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    stream_out_path.write_text(stdout, encoding="utf-8")
    return stdout, returncode, timed_out


def parse_claude_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            return event
    return {}


def run_checker(instance: dict[str, Any], container_name: str, verdict_path: Path) -> dict[str, Any]:
    fail_to_pass = instance["fail_to_pass"]
    pass_to_pass = instance["pass_to_pass"]
    all_ids = fail_to_pass + pass_to_pass
    addopts = instance.get("pytest_addopts_override")
    override = f"--override-ini='addopts={addopts}' " if addopts else ""
    quoted_ids = " ".join(shlex.quote(t) for t in all_ids)
    env_prefix = "".join(f"export {k}={shlex.quote(v)}; " for k, v in (instance.get("extra_env") or {}).items())
    script = (
        f"{env_prefix}cd /app && rm -f /tmp/results.xml && "
        f"python3 -m pytest {override}--junitxml=/tmp/results.xml -q {quoted_ids}"
    )
    proc = exec_in(container_name, script, user="agent", timeout=CHECKER_TIMEOUT_S)

    copy = docker(["cp", f"{container_name}:/tmp/results.xml", str(verdict_path.with_suffix(".xml"))])
    if copy.returncode != 0:
        return {
            "passed": False,
            "n_fail_to_pass_passed": 0,
            "n_pass_to_pass_passed": 0,
            "reason": f"no junit xml produced (pytest exit {proc.returncode}): {proc.stdout[-1500:]}{proc.stderr[-500:]}",
        }

    import xml.etree.ElementTree as ET

    root = ET.parse(verdict_path.with_suffix(".xml")).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    passed_names = set()
    if suite is not None:
        for case in suite.findall("testcase"):
            name = case.attrib.get("name", "")
            classname = case.attrib.get("classname", "")
            failed = case.find("failure") is not None or case.find("error") is not None
            if not failed:
                # node id reconstruction: pytest junit classname uses dotted
                # module path: match by substring against the requested id
                # instead of exact reconstruction (which varies by plugin).
                passed_names.add((classname, name))

    def count_passed(ids: list[str]) -> int:
        count = 0
        for node_id in ids:
            file_part, _, rest = node_id.partition("::")
            test_name = rest.rsplit("::", 1)[-1] if rest else file_part
            if any(test_name == n or test_name in n for _, n in passed_names):
                count += 1
        return count

    n_fail_to_pass_passed = count_passed(fail_to_pass)
    n_pass_to_pass_passed = count_passed(pass_to_pass)
    ok = (
        proc.returncode == 0
        and n_fail_to_pass_passed == len(fail_to_pass)
        and n_pass_to_pass_passed == len(pass_to_pass)
    )
    return {
        "passed": ok,
        "n_fail_to_pass_passed": n_fail_to_pass_passed,
        "n_pass_to_pass_passed": n_pass_to_pass_passed,
        "reason": (
            "all fail_to_pass and pass_to_pass tests pass"
            if ok
            else f"fail_to_pass {n_fail_to_pass_passed}/{len(fail_to_pass)}, "
            f"pass_to_pass {n_pass_to_pass_passed}/{len(pass_to_pass)} "
            f"(pytest exit {proc.returncode})"
        ),
    }


@dataclass
class SwebenchContext:
    ctx: RunContext
    claude_model: str
    results_dir: Path


def run_job(sctx: SwebenchContext, job: Job) -> JobResult:
    started = time.monotonic()
    instance = job.instance
    job_dir = job.job_id
    log_dir = sctx.results_dir / "logs"
    verdict_dir = sctx.results_dir / "verdicts"
    log_dir.mkdir(parents=True, exist_ok=True)
    verdict_dir.mkdir(parents=True, exist_ok=True)
    stream_path = log_dir / f"{job_dir}.stream.jsonl"
    proxy_log_path = log_dir / f"{job_dir}.proxy.log"
    verdict_path = verdict_dir / f"{job_dir}.json"

    container_name = f"swebench-{job_dir}".replace("__", "-").replace(":", "-")[:63]
    trace_dir = sctx.ctx.traces_root / f"{sctx.ctx.suite_name}-run-{job.condition}-{sctx.ctx.run_stamp}-{instance['slug']}-seed{job.seed}"

    error: str | None = None
    claude_result: dict[str, Any] = {}
    checker_verdict: dict[str, Any] = {"passed": False, "n_fail_to_pass_passed": 0, "n_pass_to_pass_passed": 0, "reason": "job did not complete"}
    proc = log_fh = None
    port = free_port()

    try:
        start_container(instance, container_name)
        setup_container(instance, container_name)

        episode_id = f"{sctx.ctx.experiment_id}-{job_dir}"
        proc, log_fh = start_proxy(sctx.ctx, job, port, trace_dir, proxy_log_path)  # type: ignore[arg-type]

        stdout, returncode, timed_out = run_claude_in_container(sctx, job, container_name, port, stream_path)
        claude_result = parse_claude_result(stdout)
        if timed_out:
            error = f"claude timed out after {instance.get('timeout_s', DEFAULT_CLAUDE_TIMEOUT_S) + CLAUDE_TIMEOUT_EXTRA_S}s"
        else:
            checker_verdict = run_checker(instance, container_name, verdict_path)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None and log_fh is not None:
            stop_proxy(proc, log_fh)
        docker(["rm", "-f", container_name])

    verdict_path.write_text(
        json.dumps({**checker_verdict, "claude_result": claude_result, "error": error}, indent=2) + "\n",
        encoding="utf-8",
    )

    return JobResult(
        job_id=job_dir,
        instance_slug=instance["slug"],
        repo=instance["repo"],
        difficulty_rank=instance["difficulty_rank"],
        condition=job.condition,
        seed=job.seed,
        passed=bool(checker_verdict.get("passed")) and error is None,
        reason=error or checker_verdict.get("reason", ""),
        wall_time_s=round(time.monotonic() - started, 2),
        n_fail_to_pass=len(instance["fail_to_pass"]),
        n_fail_to_pass_passed=checker_verdict.get("n_fail_to_pass_passed", 0),
        n_pass_to_pass=len(instance["pass_to_pass"]),
        n_pass_to_pass_passed=checker_verdict.get("n_pass_to_pass_passed", 0),
        claude_num_turns=claude_result.get("num_turns"),
        claude_duration_ms=claude_result.get("duration_ms"),
        error=error,
    )


def build_jobs(instances: list[dict[str, Any]], conditions: list[str], seeds: int) -> list[Job]:
    return [
        Job(instance=instance, condition=condition, seed=seed)
        for instance in instances
        for condition in conditions
        for seed in range(1, seeds + 1)
    ]


def run_campaign(sctx: SwebenchContext, jobs: list[Job], cloud_parallelism: int, local_parallelism: int) -> list[JobResult]:
    results: list[JobResult] = []
    lock = Lock()
    total = len(jobs)
    done = 0

    def on_done(future) -> None:
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
        futures = [cloud_pool.submit(run_job, sctx, j) for j in cloud_jobs]
        futures += [local_pool.submit(run_job, sctx, j) for j in routing_jobs]
        for fut in as_completed(futures):
            on_done(fut)

    return results


def write_summary(sctx: SwebenchContext, results: list[JobResult]) -> None:
    cells: dict[tuple[str, str], list[JobResult]] = {}
    for r in results:
        cells.setdefault((r.instance_slug, r.condition), []).append(r)

    rows = []
    for (slug, condition), cell in sorted(cells.items(), key=lambda kv: (cell_rank(kv[1]), kv[0][1])):
        passes = sum(1 for r in cell if r.passed)
        n = len(cell)
        est = binomial_rate_ci(passes, n)
        rows.append({
            "instance_slug": slug, "condition": condition, "n": n, "passes": passes,
            "pass_rate": est.rate, "ci_low": est.ci_low, "ci_high": est.ci_high,
        })

    summary = {
        "experiment_id": sctx.ctx.experiment_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [asdict(r) for r in results],
        "cells": rows,
    }
    (sctx.results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [f"# SWE-bench Pro campaign {sctx.ctx.experiment_id}", "",
             "| rank | instance | repo | condition | n | pass rate | 95% CI |",
             "|---|---|---|---|---:|---:|---|"]
    by_slug = {}
    for r in results:
        by_slug.setdefault(r.instance_slug, r)
    for row in rows:
        rank = by_slug[row["instance_slug"]].difficulty_rank
        repo = by_slug[row["instance_slug"]].repo
        lines.append(
            f"| {rank} | {row['instance_slug']} | {repo} | {row['condition']} | {row['n']} "
            f"| {row['pass_rate']:.2f} | [{row['ci_low']:.2f}, {row['ci_high']:.2f}] |"
        )
    (sctx.results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n==> wrote {sctx.results_dir / 'summary.json'}")
    print(f"==> wrote {sctx.results_dir / 'summary.md'}")


def cell_rank(cell: list[JobResult]) -> int:
    return cell[0].difficulty_rank if cell else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--instances", default="all")
    parser.add_argument("--conditions", default="cloud")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--cloud-parallelism", type=int, default=2)
    parser.add_argument("--local-parallelism", type=int, default=1)
    parser.add_argument("--claude-model", default=os.environ.get("CLAUDE_MODEL", "sonnet"))
    parser.add_argument("--vllm-url", default=os.environ.get("EDGEPROXY_VLLM_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--upstream", default=os.environ.get("EDGEPROXY_UPSTREAM", "https://lum.id/claude"))
    parser.add_argument("--max-local-tokens", type=int, default=int(os.environ.get("EDGEPROXY_MAX_LOCAL_TOKENS", "100000")))
    parser.add_argument("--local-token-margin", type=float, default=float(os.environ.get("EDGEPROXY_LOCAL_TOKEN_MARGIN", "0.90")))
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--suite-name", default="swebench-pro-15")
    parser.add_argument("--traces-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_dir = args.repo_dir.resolve()
    if not (repo_dir / "edgeproxy").is_dir():
        die(f"expected edgeproxy/ under {repo_dir}")

    loaded = load_dotenv_into_environ(repo_dir / ".env")
    if "ANTHROPIC_AUTH_TOKEN" not in loaded and "ANTHROPIC_AUTH_TOKEN" not in os.environ:
        die("ANTHROPIC_AUTH_TOKEN not available from .env or environment")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITION_POLICY:
            die(f"unknown condition: {c}")

    instances = load_instances(args.instances)
    if not instances:
        die("no instances selected")

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = args.experiment_id or f"swebench-{run_stamp}"
    results_dir = args.results_dir or (SWEBENCH_DIR / "results" / experiment_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_root = args.traces_root or (repo_dir / "traces")
    traces_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_dir))

    run_ctx = RunContext(
        repo_dir=repo_dir,
        results_dir=results_dir,
        traces_root=traces_root,
        suite_name=args.suite_name,
        python_bin=sys.executable,
        claude_bin="claude",
        claude_model=args.claude_model,
        vllm_url=args.vllm_url,
        upstream=args.upstream,
        max_local_tokens=args.max_local_tokens,
        local_token_margin=args.local_token_margin,
        experiment_id=experiment_id,
        run_stamp=run_stamp,
    )
    sctx = SwebenchContext(ctx=run_ctx, claude_model=args.claude_model, results_dir=results_dir)

    jobs = build_jobs(instances, conditions, args.seeds)
    print(f"==> experiment: {experiment_id}")
    print(f"==> instances: {', '.join(i['slug'] for i in instances)}")
    print(f"==> conditions: {', '.join(conditions)}  seeds: {args.seeds}  jobs: {len(jobs)}")
    print(f"==> results: {results_dir}")

    results = run_campaign(sctx, jobs, args.cloud_parallelism, args.local_parallelism)
    write_summary(sctx, results)

    passed = sum(1 for r in results if r.passed)
    print(f"\n==> {passed}/{len(results)} jobs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
