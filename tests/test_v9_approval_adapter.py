"""Synthetic-only approval-schema safety checks; no pilot files are read."""

import hashlib
import json

import pytest

from experiments.agentic_router import v9_approval_adapter as adapter


def _fixture(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    names = [f"synthetic/file_{i:02d}.json" for i in range(36)]
    hashes = {}
    for i, name in enumerate(names):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(str(i))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = {"version": "terminal-v9-input-approval-2026-09-17-v2",
           "status": adapter.EXPECTED_STATUS, "sha256": hashes,
           "absent_optional_inputs": sorted(adapter.OPTIONAL_ABSENT)}
    manifest = tmp_path / "approval.json"
    manifest.write_text(json.dumps(raw))
    key_sha = hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()
    return root, manifest, raw, key_sha


def _verify(root, manifest, key_sha):
    return adapter.verify_campaign_manifest(
        manifest, root=root,
        expected_outer_sha=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        expected_keyset_sha=key_sha,
    )


def test_complete_synthetic_manifest_and_changed_input(tmp_path):
    root, manifest, _, key_sha = _fixture(tmp_path)
    assert len(_verify(root, manifest, key_sha)) == 39
    (root / "synthetic/file_00.json").write_text("changed")
    with pytest.raises(ValueError, match="hash/missing"):
        _verify(root, manifest, key_sha)


def test_missing_and_unexpected_extra_paths(tmp_path):
    root, manifest, raw, key_sha = _fixture(tmp_path)
    (root / "synthetic/file_00.json").unlink()
    with pytest.raises(ValueError, match="hash/missing"):
        _verify(root, manifest, key_sha)
    (root / "synthetic/file_00.json").write_text("0")
    raw["sha256"]["synthetic/extra.json"] = "a" * 64
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="input count"):
        _verify(root, manifest, key_sha)


def test_traversal_and_symlink_escape(tmp_path):
    root, manifest, raw, key_sha = _fixture(tmp_path)
    digest = raw["sha256"].pop("synthetic/file_00.json")
    raw["../escape.json"] = digest
    # Keep the value under sha256, not at top level.
    raw["sha256"]["../escape.json"] = raw.pop("../escape.json")
    manifest.write_text(json.dumps(raw))
    traversal_key_sha = hashlib.sha256("\n".join(sorted(raw["sha256"])).encode()).hexdigest()
    with pytest.raises(ValueError, match="unsafe manifest path"):
        _verify(root, manifest, traversal_key_sha)

    raw["sha256"].pop("../escape.json")
    raw["sha256"]["synthetic/escape.json"] = digest
    (root / "synthetic/escape.json").symlink_to(tmp_path / "outside.json")
    manifest.write_text(json.dumps(raw))
    symlink_key_sha = hashlib.sha256("\n".join(sorted(raw["sha256"])).encode()).hexdigest()
    with pytest.raises(ValueError, match="escapes repository"):
        _verify(root, manifest, symlink_key_sha)


def test_optional_present_or_outer_hash_changed(tmp_path):
    root, manifest, _, key_sha = _fixture(tmp_path)
    (root / sorted(adapter.OPTIONAL_ABSENT)[0]).parent.mkdir(parents=True, exist_ok=True)
    (root / sorted(adapter.OPTIONAL_ABSENT)[0]).write_text("unexpected")
    with pytest.raises(ValueError, match="unexpectedly present"):
        _verify(root, manifest, key_sha)
    with pytest.raises(ValueError, match="outer SHA"):
        adapter.verify_campaign_manifest(manifest, root=root, expected_outer_sha="0" * 64,
                                         expected_keyset_sha=key_sha)
