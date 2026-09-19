"""Offline policy invariants for the unattended live collector."""

import json

import pytest

from experiments.agentic_router import pilot_campaign, pilot_live


def test_live_collector_uses_same_drain_and_only_development_tasks():
    assert pilot_live.JOB_TIMEOUT_S == 1200
    assert pilot_campaign.DRAIN.isoformat() == "2026-09-17T16:30:00+08:00"
    assert len(pilot_campaign.DEVELOPMENT_TASKS) == 5
    assert not set(pilot_campaign.DEVELOPMENT_TASKS) & set(pilot_campaign.HOLDOUT_TASKS)


def test_parity_checked_shadow_model_is_observe_only():
    assert pilot_live.SHADOW is not None
    assert pilot_live.SHADOW.name == "quality_model_baseline205_v5_shadow_logreg.json"


def test_container_cleanup_target_is_exact_job_name():
    name = pilot_live._scoped_container_name("swebench-flask-70ca03af28", "cloud")
    assert name.startswith("swebench-agentic-rout-")
    assert "swebench-flask-70ca03af28__cloud" in name
    assert pilot_live._scoped_container_name("swebench-flask-70ca03af28", "grader") == (
        "flowmesh-grader-check-swebench-flask-70ca03af28"
    )


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        ("upstream infrastructure failure: 3 consecutive HTTP 502 responses", True),
        ("claude timed out at configured per-job hard deadline", False),
        ("DockerError: image unavailable", False),
        (None, False),
    ],
)
def test_only_confirmed_upstream_502_is_cloud_retryable(tmp_path, monkeypatch, error, retryable):
    monkeypatch.setattr(pilot_live, "RESULTS", tmp_path)
    path = tmp_path / "verdicts" / "example__cloud__seed1.json"
    path.parent.mkdir()
    assert pilot_live._initial_cloud_launch_needed("example")
    path.write_text(json.dumps({"passed": False, "reason": "job did not complete", "error": error}))
    assert not pilot_live._initial_cloud_launch_needed("example")
    assert pilot_live._retryable_cloud_infra("example") is retryable


def test_resume_skips_persisted_time_budget_and_launches_next_task(tmp_path, monkeypatch):
    requests, seaborn, sklearn = "requests", "seaborn", "sklearn"
    verdicts = tmp_path / "verdicts"
    verdicts.mkdir()
    (verdicts / f"{requests}__cloud__seed1.json").write_text(
        json.dumps({"passed": True, "reason": "all tests pass", "error": None})
    )
    (verdicts / f"{seaborn}__cloud__seed1.json").write_text(
        json.dumps({"passed": False, "reason": "job did not complete",
                    "error": "claude timed out at configured per-job hard deadline"})
    )
    monkeypatch.setattr(pilot_live, "RESULTS", tmp_path)
    monkeypatch.setattr(pilot_live, "LOCK", tmp_path / "pilot_live.lock")
    monkeypatch.setattr(pilot_live, "STATUS", tmp_path / "pilot_live_status.json")
    monkeypatch.setattr(pilot_campaign, "DEVELOPMENT_TASKS", (requests, seaborn, sklearn))
    monkeypatch.setattr(pilot_live, "_remaining", lambda: 1000.0)
    monkeypatch.setattr(pilot_live, "_grade_sanity", lambda _slug: True)
    monkeypatch.setattr(pilot_live, "_wait_for_replay", lambda _slug: True)
    launches = []

    def fake_launch(slug, condition):
        launches.append((slug, condition))
        (verdicts / f"{slug}__{condition}__seed1.json").write_text(
            json.dumps({"passed": True, "reason": "all tests pass", "error": None})
        )
        return 0, "completed"

    monkeypatch.setattr(pilot_live, "_launch", fake_launch)
    pilot_live.run()
    assert launches == [(sklearn, "cloud")]
    assert json.loads((verdicts / f"{seaborn}__cloud__seed1.json").read_text())["error"].startswith(
        "claude timed out"
    )


def test_only_task_does_not_launch_other_preregistered_jobs(tmp_path, monkeypatch):
    tasks = ("first", "target", "later")
    monkeypatch.setattr(pilot_campaign, "DEVELOPMENT_TASKS", tasks)
    monkeypatch.setattr(pilot_live, "RESULTS", tmp_path)
    monkeypatch.setattr(pilot_live, "LOCK", tmp_path / "pilot_live.lock")
    monkeypatch.setattr(pilot_live, "STATUS", tmp_path / "pilot_live_status.json")
    monkeypatch.setattr(pilot_live, "_remaining", lambda: 1000.0)
    checked = []
    monkeypatch.setattr(pilot_live, "_grade_sanity", lambda slug: checked.append(slug) or True)
    monkeypatch.setattr(pilot_live, "_wait_for_replay", lambda _slug: True)
    verdicts = tmp_path / "verdicts"
    verdicts.mkdir()

    def fake_launch(slug, condition):
        (verdicts / f"{slug}__{condition}__seed1.json").write_text(
            json.dumps({"passed": True, "reason": "all tests pass", "error": None}))
        return 0, "completed"

    monkeypatch.setattr(pilot_live, "_launch", fake_launch)
    pilot_live.run(only_task="target")
    assert checked == ["target"]
    assert sorted(path.name for path in verdicts.iterdir()) == ["target__cloud__seed1.json"]
    with pytest.raises(ValueError, match="frozen development"):
        pilot_live.run(only_task="unregistered")
