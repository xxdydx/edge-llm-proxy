"""Fail-closed checks for the exact Astropy 13033 image setup pin."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


RUNNER_DIR = Path(__file__).resolve().parent.parent / "eval-suite" / "swebench" / "runner"
sys.path.insert(0, str(RUNNER_DIR))
SPEC = importlib.util.spec_from_file_location(
    "verify_official_grader", RUNNER_DIR / "verify_official_grader.py"
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)

OLD = "32ebe645ce6f2494138180666bfe7816bcbe587c"
NEW = "6ebe80c7a6a250ca25409a6be3a0c15f6c0d4578"
BASE = "298ccb478e6bf092953bca67a3d29dc6c35f6752"


def _raw(path="pyproject.toml", old=OLD, new=NEW):
    return f":100644 100644 {old} {new} M\t{path}\n"


def _check(monkeypatch, raw, *, instance_id="astropy__astropy-13033", base=BASE):
    monkeypatch.setattr(verifier, "docker", lambda _: SimpleNamespace(returncode=0, stdout=raw))
    return verifier._image_source_delta(
        {"instance_id": instance_id, "base_commit": base, "repo_dir": "/testbed"}, "grader"
    )


def test_exact_astropy_13033_setup_pin_allowed(monkeypatch):
    assert _check(monkeypatch, _raw()) == ["pyproject.toml"]


@pytest.mark.parametrize(
    ("raw", "instance_id", "base"),
    [
        (_raw(new="0" * 40), "astropy__astropy-13033", BASE),
        (_raw() + _raw(path="astropy/modeling/core.py"), "astropy__astropy-13033", BASE),
        (_raw(), "astropy__astropy-12907", BASE),
        (_raw(), "astropy__astropy-13033", "0" * 40),
        (_raw() + "malformed\n", "astropy__astropy-13033", BASE),
    ],
)
def test_astropy_13033_adjacent_changes_rejected(monkeypatch, raw, instance_id, base):
    with pytest.raises(RuntimeError):
        _check(monkeypatch, raw, instance_id=instance_id, base=base)
