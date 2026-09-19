import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_fanout_policy_pair.sh"


def dry_run(*args: str) -> str:
    env = os.environ.copy()
    env.update({"REPO_DIR": str(ROOT), "CLAUDE_BIN": "/definitely/not/claude"})
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", *args],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def test_all_dry_run_selects_four_conditions_and_exact_ablation_flags():
    output = dry_run("--condition", "all", "--mode", "concurrent")

    assert output.count("condition=") == 4
    assert "condition=cloud: --policy cloud-only" in output
    assert (
        "condition=routing: --policy static --cohort-tracking observe "
        "--cohort-window-ms 300 --cohort-barrier-timeout-ms 5000 "
        "--local-cache-tracking observe "
        "--local-cache-salt-scope condition\n"
    ) in output
    assert (
        "condition=cohort: --policy static --cohort-tracking observe "
        "--cohort-window-ms 300 --cohort-barrier-timeout-ms 5000 "
        "--local-cache-tracking observe "
        "--local-cache-salt-scope condition --cohort-parent-placement"
    ) in output
    assert (
        "condition=ablation: --policy static --cohort-tracking observe "
        "--cohort-window-ms 300 --cohort-barrier-timeout-ms 5000 "
        "--local-cache-tracking observe "
        "--local-cache-salt-scope request --cohort-parent-placement"
    ) in output


def test_pair_dry_run_preserves_original_unsalted_cloud_and_routing_pair():
    output = dry_run("--condition", "pair", "--mode", "sequential")

    assert output.count("condition=") == 2
    assert "condition=cohort:" not in output
    assert "condition=ablation:" not in output
    assert "condition=routing: --policy static" in output
    assert "--local-cache-salt-scope off" in output
    assert "--cohort-parent-placement" not in output
