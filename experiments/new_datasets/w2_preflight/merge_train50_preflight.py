"""Merge the immutable Smith rows, ten isolated gym rows, and Dask rejection rows."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "w2_preflight/preflight.train50.v1.jsonl"
GYM_RESULTS = ROOT / "w2_preflight/train50_instances/gym_results.json"

EXPECTED_GYM = {
    "gym:python__mypy-15413",
    "gym:pandas-dev__pandas-56849",
    "gym:iterative__dvc-5336",
    "gym:conan-io__conan-14177",
    "gym:facebookresearch__hydra-2189",
    "gym:pydantic__pydantic-8500",
    "gym:python__mypy-12222",
    "gym:pandas-dev__pandas-51605",
    "gym:iterative__dvc-9391",
    "gym:conan-io__conan-15422",
}

DASK_REJECTIONS = [
    {
        "task_uid": "gym:dask__dask-7656",
        "status": "fail",
        "selection_used_model_outcome": False,
        "certificate_ref": "experiments/new_datasets/logs/build_images/env/sweb.env.arm64.5c2999da6147421f2209dc__latest/build_image.log",
        "image_digest": None,
        "env_image_build_seconds": 648.21,
        "instance_build_plus_container_wall_seconds": None,
        "warm_grade_seconds": None,
        "fail_to_pass_confirmed_failing": None,
        "fail_to_pass_confirmed_passing": None,
        "fail_to_pass_total": 1,
        "pass_to_pass_confirmed_passing": None,
        "pass_to_pass_total": 48,
        "rejection_reason": "unsupported_dependency",
        "harness": "SWE-Gym/SWE-Bench-Fork (official adapter, local vendor clone at experiments/new_datasets/w2_preflight/_vendor/SWE-Bench-Fork)",
        "arch": "arm64",
        "platform": "linux/arm64/v8",
        "model_patch_applied": "not attempted",
        "notes": "Genuine native-arm64 environment build rejection; 2026-09-19 conda solve log excerpt: Platform: linux-aarch64; PackagesNotFoundError: The following packages are not available from current channels: crick. The solve ran from 09:57:30.283 to 10:08:17.480 (~648.21s). Dask was explicitly excluded from the isolated retry.",
    },
    {
        "task_uid": "gym:dask__dask-9212",
        "status": "fail",
        "selection_used_model_outcome": False,
        "certificate_ref": "experiments/new_datasets/logs/build_images/env/sweb.env.arm64.5c2999da6147421f2209dc__latest/build_image.log",
        "image_digest": None,
        "env_image_build_seconds": 648.21,
        "instance_build_plus_container_wall_seconds": None,
        "warm_grade_seconds": None,
        "fail_to_pass_confirmed_failing": None,
        "fail_to_pass_confirmed_passing": None,
        "fail_to_pass_total": 2,
        "pass_to_pass_confirmed_passing": None,
        "pass_to_pass_total": 103,
        "rejection_reason": "unsupported_dependency",
        "harness": "SWE-Gym/SWE-Bench-Fork (official adapter, local vendor clone at experiments/new_datasets/w2_preflight/_vendor/SWE-Bench-Fork)",
        "arch": "arm64",
        "platform": "linux/arm64/v8",
        "model_patch_applied": "not attempted",
        "notes": "Genuine native-arm64 environment build rejection; 2026-09-19 conda solve log excerpt: Platform: linux-aarch64; PackagesNotFoundError: The following packages are not available from current channels: crick. The solve ran from 09:57:30.283 to 10:08:17.480 (~648.21s). Dask was explicitly excluded from the isolated retry.",
    },
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    existing = read_jsonl(LEDGER)
    if len(existing) != 36 or any(not row["task_uid"].startswith("smith:") for row in existing):
        raise SystemExit("refusing to merge: ledger is not the expected 36 Smith rows")
    if len({row["task_uid"] for row in existing}) != 36:
        raise SystemExit("refusing to merge: duplicate existing task_uid")
    if not GYM_RESULTS.exists():
        raise SystemExit(f"refusing to merge: missing {GYM_RESULTS}")

    gym = json.loads(GYM_RESULTS.read_text())
    gym_ids = [row["task_uid"] for row in gym]
    if set(gym_ids) != EXPECTED_GYM or len(gym_ids) != len(EXPECTED_GYM):
        raise SystemExit("refusing to merge: gym results are not exactly the expected ten non-Dask tasks")
    if any(row.get("rejection_reason") and not row.get("error") for row in gym):
        raise SystemExit("refusing to merge: unexpected gym rejection row lacks error evidence")

    merged = existing + gym + DASK_REJECTIONS
    LEDGER.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in merged))


if __name__ == "__main__":
    main()
