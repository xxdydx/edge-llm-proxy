"""Synthetic-only checks: never read the development or future v9 snapshots."""

import hashlib
import json

import joblib
import numpy as np
import pytest

from experiments.agentic_router import analysis, frozen_quality_bundle as bundle
from experiments.agentic_router.preregistered_corrected_order_quality import Event, _fit_predict


def _events():
    rows = []
    for i in range(24):
        rows.append(Event(
            call_id=f"synthetic-{i}", task_group=f"synthetic-repo-{i % 4}",
            label="HARM" if i % 4 == 0 else "SAFE",
            features={name: float((i + j) % 7) for j, name in enumerate(analysis.FEATURE_NAMES)},
            request_text=f"synthetic request {i % 5} token_{i % 3}",
            namespace="synthetic", inclusion_probability=0.5,
            supported=True, exclusion_reason=None,
        ))
    return rows


def test_frozen_pipelines_match_preregistered_score_convention():
    train, test = _events()[:20], _events()[20:]
    models = bundle._fit_all(train)
    got = bundle._score(models, test)
    assert list(got) == list(bundle.KINDS)
    for kind in bundle.KINDS:
        assert np.allclose(got[kind], _fit_predict(kind, train, test))
    assert all(list(models[kind].classes_) == [0, 1] for kind in
               ("numeric_logistic", "numeric_ridge", "shallow_forest"))
    assert list(models["word_structural"]["model"].classes_) == [0, 1]
    assert all(len(scores) == 0 for scores in bundle._score(models, []).values())


def test_scoring_fails_closed_on_unsupported_or_missing_inputs():
    models = bundle._fit_all(_events())
    base = _events()[0]
    unsupported = Event(**{**base.__dict__, "supported": False})
    with pytest.raises(ValueError, match="unsupported"):
        bundle._score(models, [unsupported])
    missing = Event(**{**base.__dict__, "features": {}})
    with pytest.raises(ValueError, match="missing predecision"):
        bundle._score(models, [missing])


def test_bundle_hash_and_cloud_all_policy_are_enforced(tmp_path):
    models = bundle._fit_all(_events())
    artifacts = {}
    for kind, filename in bundle.ARTIFACT_NAMES.items():
        path = tmp_path / filename
        joblib.dump(models[kind], path)
        artifacts[kind] = {"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        "judge_protocol": bundle.PROTOCOL,
        "predecision_feature_version": bundle.features.PREDECISION_FEATURE_VERSION,
        "history_support_version": bundle.HISTORY_SUPPORT_VERSION,
        "feature_names": list(analysis.FEATURE_NAMES),
        "fit_code_sha256": bundle._hash(bundle.Path(bundle.__file__)),
        "v1_1_runner_code_sha256": bundle._hash(bundle.Path(bundle.__file__).with_name("preregistered_corrected_order_quality.py")),
        "decision_policy": {"quality_gate_active": False, "placement": "cloud_all", "threshold": None},
        "models": artifacts,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert set(bundle.load_bundle(tmp_path)[1]) == set(bundle.KINDS)
    manifest["decision_policy"]["threshold"] = 0.5
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema or shadow policy"):
        bundle.load_bundle(tmp_path)
    manifest["decision_policy"]["threshold"] = None
    path.write_text(json.dumps(manifest))
    (tmp_path / bundle.ARTIFACT_NAMES["numeric_logistic"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        bundle.load_bundle(tmp_path)
