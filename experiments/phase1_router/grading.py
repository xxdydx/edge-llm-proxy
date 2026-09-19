"""Grade a model completion with the OFFICIAL EvalPlus differential-testing
semantics, inside a locked-down Docker container.

**Model code only ever runs in the container** (``--network none
--read-only --cap-drop ALL --pids-limit --memory --cpus --user nobody``,
tmpfs /tmp). The grader script calls ``evalplus.eval.untrusted_check`` -- the
same routine EvalPlus scoring uses -- so special oracles (SET_EQ / NOT_NONE /
float atol / per-task), timing limits, and reliability guards are inherited,
not reimplemented.

Groundtruth (``expected``) and per-input reference times are precomputed on
the host from the trusted canonical solution (``dataset.build_dataset`` ->
``trusted_exec``), each on an independent deepcopy. The grader deepcopies the
inputs again before the candidate runs, so a candidate that mutates its
arguments cannot corrupt the reference or a later grader run.

The container's result is emitted on a sentinel-prefixed line so candidate
stdout cannot corrupt parsing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .config import Phase1Config
from .schema import GradeResult, Problem

_SENTINEL = "@@PHASE1_RESULT@@"

# Runs INSIDE the grader container.
_GRADER_SRC = r'''
import json, os, sys, copy

# Save the REAL stdout fd before anything can redirect it, so the result line
# is written even after candidate stdout/stderr are silenced.
_REAL_OUT = os.dup(1)

_spec = json.load(open("/w/spec.json"))
sys.path.insert(0, "/w")
from _serde import decode

ep = _spec["entry_point"]
base = [decode(x) for x in _spec["base_input"]]
plus = [decode(x) for x in _spec["plus_input"]]
inputs = [copy.deepcopy(x) for x in (base + plus)]        # independent copies
expected = [decode(x) for x in _spec["expected"]]
ref_time = [float(t) for t in _spec["ref_time"]]
atol = _spec.get("atol") or 0.0
n_base = _spec["n_base"]
code = _spec.get("candidate_code") or ""

def emit(d):
    # sentinel-prefixed, written to the SAVED real stdout fd so candidate
    # stdout (redirected below) cannot corrupt or suppress it
    os.write(_REAL_OUT, ("\n%s %s\n" % ("@@PHASE1_RESULT@@", json.dumps(d))).encode())

if not code.strip():
    emit({"status": "NO_CODE", "n_base": n_base, "n_base_ok": 0,
          "n_plus": len(plus), "n_plus_ok": 0, "detail": "empty completion"})
    sys.exit(0)

# silence candidate stdout/stderr at the fd level for the whole check
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 1); os.dup2(devnull, 2)

try:
    from evalplus.eval import untrusted_check, PASS
    stat, details = untrusted_check(
        dataset="mbpp",
        code=code,
        inputs=inputs,
        entry_point=ep,
        expected=expected,
        atol=atol,
        ref_time=ref_time,
        fast_check=False,
        min_time_limit=float(_spec.get("min_time_limit", 1.0)),
        gt_time_limit_factor=float(_spec.get("gt_time_factor", 4.0)),
    )
    det = list(details) + [False] * (len(inputs) - len(details))
    nbo = int(sum(1 for x in det[:n_base] if x))
    npo = int(sum(1 for x in det[n_base:] if x))
    emit({"status": str(stat), "n_base": n_base, "n_base_ok": nbo,
          "n_plus": len(plus), "n_plus_ok": npo, "detail": ""})
except BaseException as e:
    emit({"status": "GRADER_ERROR", "n_base": n_base, "n_base_ok": 0,
          "n_plus": len(plus), "n_plus_ok": 0, "detail": repr(e)[:300]})
'''

_SERDE_FOR_CONTAINER = (Path(__file__).parent / "serde.py").read_text()


def _classify(r: dict) -> GradeResult:
    nb, nbo, npl, nplo = r["n_base"], r["n_base_ok"], r["n_plus"], r["n_plus_ok"]
    base_pass = nb > 0 and nbo == nb
    plus_pass = (nb + npl) > 0 and (nbo + nplo) == (nb + npl)
    raw = r["status"].lower()
    if r["status"] == "NO_CODE":
        status = "NO_CODE"
    elif r["status"] == "GRADER_ERROR":
        status = "ERROR"
    elif "timeout" in raw:
        status = "TIMEOUT"
    elif plus_pass:
        status = "PASS"
    elif (nb + npl) > 0:
        status = "FAIL"
    else:
        status = "SKIPPED"
    return GradeResult(
        status=status, base_pass=base_pass, plus_pass=plus_pass,
        n_base=nb, n_base_ok=nbo, n_plus=npl, n_plus_ok=nplo, detail=r.get("detail", ""),
    )


def grade_completion(problem: Problem, candidate_code: str | None, cfg: Phase1Config) -> GradeResult:
    t0 = time.monotonic()
    if candidate_code is None or not candidate_code.strip():
        return GradeResult("NO_CODE", False, False, problem.n_base, 0, problem.n_plus, 0,
                           "no code extracted", 0.0)

    workdir = Path(tempfile.mkdtemp(prefix="phase1grade_"))
    try:
        (workdir / "spec.json").write_text(json.dumps({
            "entry_point": problem.entry_point,
            "candidate_code": candidate_code,
            "base_input": problem.base_input,
            "plus_input": problem.plus_input,
            "expected": problem.expected,
            "ref_time": problem.ref_time,
            "atol": problem.atol,
            "n_base": problem.n_base,
            "min_time_limit": cfg.grader_min_time_limit,
            "gt_time_factor": cfg.grader_gt_time_factor,
        }))
        (workdir / "grade.py").write_text(_GRADER_SRC)
        (workdir / "_serde.py").write_text(_SERDE_FOR_CONTAINER)
        cmd = [
            "docker", "run", "--rm",
            "--network", "none", "--read-only",
            "--tmpfs", "/tmp:size=128m",
            "--memory", "1g", "--memory-swap", "1g",
            "--pids-limit", "512", "--cpus", "1.0",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "-v", f"{workdir}:/w:ro",
            cfg.grader_image,
            "python", "/w/grade.py",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=cfg.grader_total_timeout_s)
        except subprocess.TimeoutExpired:
            return GradeResult("TIMEOUT", False, False, problem.n_base, 0, problem.n_plus, 0,
                               f"docker run exceeded {cfg.grader_total_timeout_s}s",
                               round(time.monotonic() - t0, 2))
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(_SENTINEL)), None)
        if line is None:
            # rc 137 (SIGKILL, ~always the --memory cgroup OOM killer) or 139
            # (SIGSEGV) with no result is the CANDIDATE exhausting resources /
            # crashing the interpreter -- that is a real correctness failure of
            # the model's code, not a grader/infra fault. ERROR stays reserved
            # for genuine grader malfunction (unparseable output, docker error).
            wall = round(time.monotonic() - t0, 2)
            if proc.returncode in (137, 139):
                return GradeResult(
                    "TIMEOUT", False, False, problem.n_base, 0, problem.n_plus, 0,
                    f"candidate killed (rc={proc.returncode}: "
                    f"{'OOM / memory limit' if proc.returncode == 137 else 'segfault'})",
                    wall,
                )
            return GradeResult("ERROR", False, False, problem.n_base, 0, problem.n_plus, 0,
                               f"no sentinel line; rc={proc.returncode} stderr={proc.stderr[:300]}",
                               wall)
        r = json.loads(line[len(_SENTINEL):].strip())
        g = _classify(r)
        return GradeResult(**{**g.__dict__, "grader_wall_s": round(time.monotonic() - t0, 2)})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
