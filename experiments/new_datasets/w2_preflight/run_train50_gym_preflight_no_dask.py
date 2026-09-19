"""Isolated arm64 gym preflight runner for the ten non-Dask tasks.

This deliberately reuses the original runner's source/spec/result helpers and
vendored SWE-Bench-Fork harness.  The only methodological change is that each
task builds its environment image and runs independently, so one task cannot
abort the remaining ten.  Dask task_uids are rejected by the allow-list.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import docker

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_train50_gym_preflight import (  # noqa: E402
    OUT,
    RUN_ID,
    build_env_images,
    image_digest,
    make_test_spec,
    read_catalogue,
    read_pending,
    run_instance,
    runtime_from_log,
    source_row,
    write_instance,
)


GYM_NON_DASK = {
    "gym:python__mypy-12222",
    "gym:python__mypy-15413",
    "gym:pandas-dev__pandas-51605",
    "gym:pandas-dev__pandas-56849",
    "gym:iterative__dvc-5336",
    "gym:iterative__dvc-9391",
    "gym:conan-io__conan-14177",
    "gym:conan-io__conan-15422",
    "gym:facebookresearch__hydra-2189",
    "gym:pydantic__pydantic-8500",
}

ISOLATED_RUN_ID = RUN_ID + ".no_dask"


def result_row(meta, spec, client, env_seconds, wall, result, error=None):
    uid = meta["task_uid"]
    iid = meta["instance_id"]
    instance_dir = OUT / uid.replace(":", "__")
    report_path = Path("logs/run_evaluation") / ISOLATED_RUN_ID / "preflight-baseline-noop" / iid / "report.json"
    report = result[1] if result else None
    if report is not None:
        (instance_dir / "report.json").write_text(json.dumps(report, indent=2))
    try:
        digest = image_digest(client, spec.instance_image_key)
    except Exception:
        digest = None
    detail = report.get(iid, {}) if report else {}
    tests_status = detail.get("tests_status", {})
    ftp = tests_status.get("FAIL_TO_PASS", {})
    ptp = tests_status.get("PASS_TO_PASS", {})
    resolved = bool(detail.get("resolved")) if report else False
    row = {
        "task_uid": uid,
        "status": "pass" if resolved else "fail",
        "selection_used_model_outcome": False,
        "certificate_ref": f"{report_path.as_posix()}" if report else None,
        "image_digest": digest,
        "env_image_build_seconds": env_seconds,
        "instance_build_plus_container_wall_seconds": wall,
        "warm_grade_seconds": runtime_from_log(report_path.parent / "run_instance.log") if report else None,
        "fail_to_pass_confirmed_failing": len(ftp.get("failure", [])),
        "fail_to_pass_confirmed_passing": len(ftp.get("success", [])),
        "fail_to_pass_total": len(spec.FAIL_TO_PASS),
        "pass_to_pass_confirmed_passing": len(ptp.get("success", [])),
        "pass_to_pass_total": len(spec.PASS_TO_PASS),
        "harness": "SWE-Gym/SWE-Bench-Fork (official adapter, local vendor clone at experiments/new_datasets/w2_preflight/_vendor/SWE-Bench-Fork)",
        "arch": spec.arch,
        "platform": spec.platform,
        "test_command": " ".join(spec.eval_script_list[-3:-2]),
        "model_patch_applied": "empty (baseline, no repair patch) - confirms pre-fix buggy state only",
        "notes": "Real native-arm64 container execution against the exact base_commit; no gold/reference patch applied; no model/agent invoked."
        if report
        else f"Per-task harness failure after isolated attempt: {error}",
    }
    if error:
        row["rejection_reason"] = "harness_error"
        row["error"] = error
    return row


def main() -> None:
    pending = read_pending()
    catalogue = read_catalogue()
    gym_meta = [catalogue[x["task_uid"]] for x in pending if x["task_uid"] in GYM_NON_DASK]
    if {x["task_uid"] for x in gym_meta} != GYM_NON_DASK:
        raise SystemExit("refusing to run: pending/catalogue does not contain exactly the ten non-Dask gym tasks")

    OUT.mkdir(parents=True, exist_ok=True)
    client = docker.from_env()
    ledger = []
    for meta in gym_meta:
        uid = meta["task_uid"]
        instance = write_instance(meta, source_row(meta))
        spec = make_test_spec(instance)
        assert spec.arch == "arm64" and spec.platform == "linux/arm64/v8"
        t0 = time.monotonic()
        env_seconds = None
        t1 = None
        error = None
        result = None
        try:
            env_result = build_env_images(client, [spec], max_workers=1)
            env_seconds = round(time.monotonic() - t0, 2)
            if env_result[1]:
                raise RuntimeError(f"environment image build failed: {env_result[1]}")
            pred = {"instance_id": spec.instance_id, "model_name_or_path": "preflight-baseline-noop", "model_patch": ""}
            t1 = time.monotonic()
            result = run_instance(spec, pred, rm_image=False, force_rebuild=False, client=client,
                                  run_id=ISOLATED_RUN_ID, timeout=120)
            if result is None:
                raise RuntimeError("run_instance returned no result; inspect its run_instance.log")
        except Exception as exc:  # isolate this task; continue with the next one
            error = repr(exc)
        if env_seconds is None:
            env_seconds = round(time.monotonic() - t0, 2)
        wall = round(time.monotonic() - t1, 2) if t1 is not None else None
        row = result_row(meta, spec, client, env_seconds, wall, result, error)
        print(json.dumps(row), flush=True)
        ledger.append(row)
    (OUT / "gym_results.json").write_text(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
