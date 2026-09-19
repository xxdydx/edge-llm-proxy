#!/usr/bin/env python3
"""Gold/base sanity check for one SWE-bench Verified pilot instance.

Uses a separate ephemeral Docker container; does not touch the agent's live
checkout.  The official eval_script/grading parser must fail FAIL_TO_PASS on
the unpatched base, then pass all tests after the dataset gold patch.  A
machine-readable summary is written without recording the patch itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from run_swebench import (SWEBENCH_DIR, docker, load_instances,
                          run_checker_official_swebench, start_container)


def fingerprint(instance: dict) -> dict:
    inspect = docker(["image", "inspect", "--format", "{{.Id}}", instance["docker_image"]])
    if inspect.returncode != 0:
        raise RuntimeError("Docker image unavailable for grader fingerprint")
    return {
        "base_commit": instance["base_commit"],
        "eval_script_sha256": hashlib.sha256(instance["eval_script"].encode()).hexdigest(),
        "docker_image_id": inspect.stdout.strip(),
    }


def _image_source_delta(instance: dict, container_name: str) -> list[str]:
    """Accept only a known dependency pin atop the instance base tree.

    SWE-bench's Astropy image includes a setup-only `pyproject.toml`
    setuptools pin in a synthetic `SWE-bench` commit, plus many chmod-only
    differences.  A naive `git diff --quiet base HEAD` rejects it despite no
    task-source/test change.  Compare blob IDs (not file modes), then allow
    exactly that known pin; everything else still fails closed.
    """
    repo_dir = instance.get("repo_dir", "/testbed")
    base = instance["base_commit"]
    raw = docker(["exec", container_name, "bash", "-lc",
                  f"cd {repo_dir} && git -c safe.directory={repo_dir} diff --raw --no-abbrev {base} HEAD"])
    if raw.returncode != 0:
        raise RuntimeError("Cannot compare Docker image source tree with instance base")
    content_changes: list[str] = []
    content_blobs: dict[str, tuple[str, str]] = {}
    for line in raw.stdout.splitlines():
        if not line.startswith(":") or "\t" not in line:
            raise RuntimeError("Malformed git raw diff while validating image source")
        meta, path = line.split("\t", 1)
        parts = meta.split()
        if (len(parts) != 5 or len(parts[2]) != 40 or len(parts[3]) != 40
                or not path or any(c not in "0123456789abcdef" for c in parts[2] + parts[3])):
            raise RuntimeError("Malformed git raw diff metadata while validating image source")
        if parts[2] != parts[3]:
            content_changes.append(path)
            content_blobs[path] = (parts[2], parts[3])
    if not content_changes:
        return []
    known_astropy_setup_pin = (
        "3364d307406a06457fba6b129db98c2399e7d624",
        "02dddbe713a3d037b3e5f6ba49af3b7ec1c49c0b",
    )
    known_astropy_13033_setup_pin = (
        "32ebe645ce6f2494138180666bfe7816bcbe587c",
        "6ebe80c7a6a250ca25409a6be3a0c15f6c0d4578",
    )
    known_sphinx_report_flag = (
        "f0afd779b6aaa81d6b4f4979edf07314ca9efa4b",
        "30ca902754f884752095a44e62887134d99595ef",
    )
    # This image's setup commit adds pytest's -rA reporting flag only.  It
    # changes neither the test selection nor the source under evaluation.
    if (instance.get("instance_id") == "sphinx-doc__sphinx-10323"
            and instance.get("base_commit") == "31eba1a76dd485dc633cae48227b46879eda5df4"
            and content_changes == ["tox.ini"]
            and content_blobs["tox.ini"] == known_sphinx_report_flag):
        return content_changes
    if (instance.get("instance_id") == "astropy__astropy-13033"
            and instance.get("base_commit") == "298ccb478e6bf092953bca67a3d29dc6c35f6752"
            and content_changes == ["pyproject.toml"]
            and content_blobs["pyproject.toml"] == known_astropy_13033_setup_pin):
        return content_changes
    if (instance.get("instance_id") != "astropy__astropy-12907"
            or instance.get("base_commit") != "d16bfe05a744909de4b27f5875fe0d4ed41ce607"
            or content_changes != ["pyproject.toml"]
            or content_blobs["pyproject.toml"] != known_astropy_setup_pin):
        raise RuntimeError(f"Unexpected image source changes against base: {content_changes[:5]}")
    return content_changes


def _official_gold_patch(instance_id: str) -> str:
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", streaming=True)
    for row in ds:
        if row["instance_id"] == instance_id:
            return row["patch"]
    raise RuntimeError(f"Instance {instance_id} not found in official dataset")


def verify(slug: str, out_dir: Path) -> dict:
    instance = next(iter(load_instances(slug)))
    if not (instance.get("eval_script") and instance.get("eval_type")):
        raise ValueError("Only official SWE-bench Verified instances are supported")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Deterministic scoped name allows the overnight controller to remove
    # precisely this grader container if its process group is hard-killed
    # before the finally block below can execute.
    name = f"flowmesh-grader-check-{slug}"
    result = {"instance_id": instance["instance_id"], "slug": slug,
              "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    start_container(instance, name)
    try:
        result["fingerprint"] = fingerprint(instance)
        repo_dir = instance.get("repo_dir", "/testbed")
        base = instance["base_commit"]
        source_delta = _image_source_delta(instance, name)
        clean = docker(["exec", name, "bash", "-lc",
                        f"cd {repo_dir} && git -c safe.directory={repo_dir} diff --quiet"])
        if clean.returncode != 0:
            raise RuntimeError("Image checkout dirty before evaluation")
        base_verdict = run_checker_official_swebench(instance, name, out_dir / f"{slug}.base.json")
        gold_patch = _official_gold_patch(instance["instance_id"])
        apply = subprocess.run(
            ["docker", "exec", "-i", name, "bash", "-lc", f"cd {repo_dir} && git apply -"],
            input=gold_patch, capture_output=True, text=True, timeout=60,
        )
        if apply.returncode != 0:
            raise RuntimeError(f"Gold patch did not apply: {apply.stderr[-500:]}")
        gold_verdict = run_checker_official_swebench(instance, name, out_dir / f"{slug}.gold.json")
        result.update({"image_source_delta": source_delta,
                       "base": base_verdict, "gold": gold_verdict,
                       "sanity_pass": (
                           not base_verdict["passed"]
                           and base_verdict["n_fail_to_pass_passed"] < len(instance["fail_to_pass"])
                           and base_verdict["n_pass_to_pass_passed"] == len(instance["pass_to_pass"])
                           and gold_verdict["passed"]
                       )})
    finally:
        docker(["rm", "-f", name])
    (out_dir / f"{slug}.sanity.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug")
    parser.add_argument("--out-dir", type=Path,
                        default=SWEBENCH_DIR / "results" / "agentic-router-pilot-v1" / "grader_sanity")
    args = parser.parse_args()
    result = verify(args.slug, args.out_dir)
    print(json.dumps({"slug": result["slug"], "sanity_pass": result["sanity_pass"],
                      "base": result["base"]["reason"], "gold": result["gold"]["reason"]}))
    return 0 if result["sanity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
