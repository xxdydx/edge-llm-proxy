"""Shared helpers for eval-suite task checkers.

Every checker.py is invoked as a subprocess with a uniform CLI contract
(--sandbox, --fixture, --report, --out) and imports only from this module.
Keeping checkers dependency-free (stdlib only) means they run identically
whether they are exercised locally against a hand-written gold solution or
against a live Claude Code transcript from either policy condition.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ANSWER_START = "<!-- ANSWER_START -->"
ANSWER_END = "<!-- ANSWER_END -->"
REPORT_START = "<!-- FANOUT_REPORT_START -->"
REPORT_COMPLETE = "<!-- FANOUT_REPORT_COMPLETE -->"


def result(passed: bool, score: float, reason: str, **details: Any) -> dict[str, Any]:
    """Build the one JSON shape every checker returns."""
    return {
        "passed": bool(passed),
        "score": round(float(score), 4),
        "reason": reason,
        "details": details,
    }


def normalize_qualname(value: str) -> str:
    """Collapse a "package.module.function" or "module.function" answer to
    "module.function" by keeping only the last two dot-separated segments.

    Exploration-task answer keys use the bare module.function form, but
    "package.module.function" is equally correct Python terminology (it's
    the real importable path), and models answer with either form
    interchangeably. Comparing normalized values on both sides accepts both
    without the answer key needing to special-case either one.
    """
    parts = value.strip().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else value.strip()


def extract_answer_json(report_text: str) -> dict[str, Any] | None:
    """Pull the JSON object a task's answer block, or first ```json fence."""
    marked = re.search(
        re.escape(ANSWER_START) + r"(.*?)" + re.escape(ANSWER_END), report_text, re.DOTALL
    )
    candidate = marked.group(1).strip() if marked else None
    if candidate is None:
        fence = re.search(r"```json\s*(.*?)```", report_text, re.DOTALL)
        candidate = fence.group(1).strip() if fence else None
    if candidate is None:
        return None
    inner_fence = re.match(r"```(?:json)?\s*(.*?)```\s*$", candidate, re.DOTALL)
    if inner_fence:
        candidate = inner_fence.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def has_required_headings(report_text: str, headings: list[str]) -> list[str]:
    """Return the subset of exact heading lines (e.g. "## Executive Summary") missing."""
    lines = {line.strip() for line in report_text.splitlines()}
    return [heading for heading in headings if heading not in lines]


def has_report_markers(report_text: str) -> bool:
    stripped = report_text.strip()
    return stripped.startswith(REPORT_START) and REPORT_COMPLETE in stripped


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_unchanged(sandbox_dir: Path, fixture_dir: Path, relpaths: list[str]) -> list[str]:
    """Return relpaths whose sandbox content no longer matches the pristine fixture.

    Compares against the task's original fixture rather than a pre-recorded hash,
    so the runner never has to snapshot state before launching a job.
    """
    changed = []
    for rel in relpaths:
        if file_sha256(sandbox_dir / rel) != file_sha256(fixture_dir / rel):
            changed.append(rel)
    return changed


def run_pytest(
    sandbox_dir: Path, python_bin: str | None = None, args: list[str] | None = None
) -> dict[str, Any]:
    """Run pytest inside sandbox_dir and return exit code, JUnit counts, and output tails."""
    python_bin = python_bin or sys.executable
    junit_path = sandbox_dir / ".eval_junit.xml"
    cmd = [
        python_bin,
        "-m",
        "pytest",
        "-q",
        "--continue-on-collection-errors",
        f"--junitxml={junit_path}",
    ]
    if args:
        cmd.extend(args)
    try:
        proc = subprocess.run(
            cmd, cwd=sandbox_dir, capture_output=True, text=True, timeout=120
        )
        timed_out = False
        returncode = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    summary = _parse_junit(junit_path) if junit_path.exists() else None
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
        "summary": summary,
    }


def _parse_junit(junit_path: Path) -> dict[str, int]:
    root = ET.parse(junit_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    attrs = suite.attrib if suite is not None else {}
    return {
        "tests": int(attrs.get("tests", 0)),
        "failures": int(attrs.get("failures", 0)),
        "errors": int(attrs.get("errors", 0)),
        "skipped": int(attrs.get("skipped", 0)),
    }


def pytest_all_passed(pytest_result: dict[str, Any]) -> bool:
    summary = pytest_result.get("summary")
    if pytest_result.get("timed_out") or summary is None:
        return False
    return (
        pytest_result["returncode"] == 0
        and summary["tests"] > 0
        and summary["failures"] == 0
        and summary["errors"] == 0
    )


def grep_count(sandbox_dir: Path, pattern: str, glob: str = "**/*.py", flags: int = 0) -> int:
    regex = re.compile(pattern, flags)
    total = 0
    for path in sorted(sandbox_dir.glob(glob)):
        if path.is_file():
            total += len(regex.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return total


def ast_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def has_top_level_function(path: Path, name: str) -> bool:
    if not path.is_file():
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == name for node in ast_module(path).body
    )


def function_calls_name(path: Path, caller_name: str, callee_name: str) -> bool:
    """True if the body of top-level function `caller_name` contains a call to `callee_name`."""
    if not path.is_file():
        return False
    for node in ast_module(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == caller_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    func = sub.func
                    if isinstance(func, ast.Name) and func.id == callee_name:
                        return True
                    if isinstance(func, ast.Attribute) and func.attr == callee_name:
                        return True
    return False


def import_module_from_sandbox(sandbox_dir: Path, module_name: str):
    """Import a module by name from sandbox_dir without polluting sys.modules permanently."""
    import importlib
    import importlib.util

    sandbox_str = str(sandbox_dir)
    inserted = sandbox_str not in sys.path
    if inserted:
        sys.path.insert(0, sandbox_str)
    sys.modules.pop(module_name, None)
    try:
        return importlib.import_module(module_name)
    finally:
        if inserted:
            sys.path.remove(sandbox_str)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def base_checker_args(description: str):
    """Return an argparse.ArgumentParser with the standard checker CLI contract."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sandbox", type=Path, required=True, help="copy Claude Code worked in")
    parser.add_argument("--fixture", type=Path, required=True, help="pristine original fixture dir")
    parser.add_argument(
        "--report", type=Path, help="captured final-report text; may be absent/empty"
    )
    parser.add_argument("--out", type=Path, required=True, help="where to write the verdict JSON")
    return parser


def edit_task_check(sandbox: Path, fixture: Path, protected_files: list[str]) -> dict[str, Any]:
    """Standard checker body for edit/refactor tasks: tests pass, protected files untouched.

    Shared by every fix-*/refactor-* task since they all reduce to the same
    two conditions; task-specific checkers add any extra structural checks on
    top of this before returning.
    """
    changed = files_unchanged(sandbox, fixture, protected_files)
    if changed:
        return result(
            False, 0.0, f"protected file(s) were modified: {', '.join(changed)}", changed=changed
        )
    pytest_result = run_pytest(sandbox)
    passed = pytest_all_passed(pytest_result)
    summary = pytest_result.get("summary") or {}
    score = 1.0 if passed else 0.0
    reason = (
        "all tests pass"
        if passed
        else f"tests failed or errored: {summary or pytest_result.get('stderr', '')[-300:]}"
    )
    return result(passed, score, reason, pytest=pytest_result)


def run_checker_main(check_fn) -> int:
    """Standard main(): parse args, call check_fn(sandbox, fixture, report_text), write --out."""
    parser = base_checker_args(check_fn.__doc__ or "eval-suite task checker")
    args = parser.parse_args()
    report_text = ""
    if args.report and args.report.exists():
        report_text = args.report.read_text(encoding="utf-8")
    verdict = check_fn(args.sandbox, args.fixture, report_text)
    args.out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict))
    return 0 if verdict["passed"] else 1
