"""Mechanically materialize frozen v7/v8 official Verified metadata, never patches.

Only run against the pinned locally cached Arrow snapshot named by the
selected manifest. Existing differing files fail closed rather than overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.ipc as ipc


ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "eval-suite" / "swebench"
MANIFEST = BENCH / "instances_candidates_verified" / "postpilot_task_manifest_v7.json"


def build_instance(row: dict, slug: str, source_sha: str) -> dict:
    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "repo_dir": "/testbed",
        "docker_image": row["image"],
        "base_commit": row["base_commit"],
        "before_repo_set_cmd": None,
        "problem_statement": row["problem_statement"],
        "fail_to_pass": row["FAIL_TO_PASS"],
        "pass_to_pass": row["PASS_TO_PASS"],
        "issue_categories": [],
        "repo_language": "python",
        "difficulty_rank": row["difficulty"],
        "n_files_changed": None,
        "patch_len_chars": len(row["patch"]),
        "pytest_addopts_override": None,
        "extra_env": {},
        "version": row["version"],
        "eval_script": row["eval_script"],
        "eval_type": row["eval_type"],
        "log_parser": row["log_parser"],
        "source": f"Pinned local SWE-bench Verified Arrow sha256:{source_sha}; official precomputed eval fields; gold source patch omitted",
        "slug": slug,
    }


def materialize(arrow_path: Path, manifest_path: Path = MANIFEST) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    source_sha = hashlib.sha256(arrow_path.read_bytes()).hexdigest()
    if source_sha != manifest["source_arrow_sha256"]:
        raise RuntimeError("Verified Arrow snapshot hash differs from frozen manifest")
    rows = {row["instance_id"]: row for row in ipc.open_stream(arrow_path).read_all().to_pylist()}
    written: list[str] = []
    for task in manifest["tasks"]:
        row = rows[task["instance_id"]]
        slug = "swebench-" + row["repo"].split("/")[-1] + "-" + hashlib.sha256(
            (row["instance_id"] + ":" + row["base_commit"]).encode()
        ).hexdigest()[:10]
        if slug != task["slug"]:
            raise RuntimeError(f"Frozen slug mismatch for {task['instance_id']}")
        instance = build_instance(row, slug, source_sha)
        payload = json.dumps(instance, indent=2, ensure_ascii=False) + "\n"
        for directory in (BENCH / "instances_candidates_verified", BENCH / "instances"):
            path = directory / f"{slug}.json"
            if path.exists() and path.read_text() != payload:
                raise RuntimeError(f"Existing instance metadata differs: {path}")
            if not path.exists():
                path.write_text(payload)
        written.append(slug)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrow_path", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    print(json.dumps(materialize(args.arrow_path, args.manifest)))


if __name__ == "__main__":
    main()
