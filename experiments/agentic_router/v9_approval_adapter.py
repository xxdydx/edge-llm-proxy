"""Strict one-snapshot bridge from campaign v9 approval to frozen scorer schema.

This module does not read labels or fit/score a model. The immutable campaign
manifest remains the authority; the output merely expresses its checked paths
in the frozen scorer's absolute-path approval format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .frozen_quality_bundle import _external_input_hashes
from .preregistered_corrected_order_quality import REPO, RESULTS, _hash

CAMPAIGN_MANIFEST_SHA256 = "7d410fb4e46acb0476a237ec367aa0bff2d15637f6a5a0f24315a76c2d8deb91"
EXPECTED_KEYSET_SHA256 = "1cb21fbfc75df557c25b3d5d3fdc9096cb4760d9b6eb19188e46bee1d2a61f70"
EXPECTED_STATUS = "terminal_frozen_pending_parent_scoring_go"
OPTIONAL_ABSENT = {
    "experiments/agentic_router/results/postpilot_v9/quarantine_manifest.json",
    "experiments/agentic_router/results/postpilot_v9/pilot_judge_capacity_invalid_quarantine_manifest.json",
    "experiments/agentic_router/results/postpilot_v9/pilot_terminal_partial_manifest.json",
}


def _safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or any(part in (".", "..") for part in candidate.parts):
        raise ValueError(f"unsafe manifest path: {relative}")
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"manifest path escapes repository: {relative}")
    return resolved


def verify_campaign_manifest(
    source: Path, *, root: Path = REPO,
    expected_outer_sha: str = CAMPAIGN_MANIFEST_SHA256,
    expected_keyset_sha: str = EXPECTED_KEYSET_SHA256,
) -> dict[str, str | None]:
    if _hash(source) != expected_outer_sha:
        raise ValueError("campaign approval outer SHA-256 mismatch")
    raw = json.loads(source.read_text())
    if raw.get("status") != EXPECTED_STATUS or raw.get("version") != "terminal-v9-input-approval-2026-09-17-v2":
        raise ValueError("campaign approval status/version mismatch")
    hashes = raw.get("sha256")
    if not isinstance(hashes, dict) or len(hashes) != 36:
        raise ValueError("campaign approval input count mismatch")
    names = sorted(hashes)
    if hashlib.sha256("\n".join(names).encode()).hexdigest() != expected_keyset_sha:
        raise ValueError("campaign approval unexpected or missing input path")
    optional = raw.get("absent_optional_inputs")
    if not isinstance(optional, list) or set(optional) != OPTIONAL_ABSENT or len(optional) != len(OPTIONAL_ABSENT):
        raise ValueError("campaign approval optional-absence set mismatch")
    checked: dict[str, str | None] = {}
    for name in names:
        path = _safe_path(root, name)
        expected = hashes[name]
        if not isinstance(expected, str) or len(expected) != 64 or not path.is_file() or _hash(path) != expected:
            raise ValueError(f"campaign approval input hash/missing mismatch: {name}")
        if str(path) in checked:
            raise ValueError("campaign approval path alias")
        checked[str(path)] = expected
    for name in sorted(OPTIONAL_ABSENT):
        path = _safe_path(root, name)
        if path.exists() or path.is_symlink():
            raise ValueError(f"campaign optional input unexpectedly present: {name}")
        checked[str(path)] = None
    return checked


def adapt(source: Path, output: Path, *, root: Path = REPO, results: Path = RESULTS,
          expected_outer_sha: str = CAMPAIGN_MANIFEST_SHA256,
          expected_keyset_sha: str = EXPECTED_KEYSET_SHA256) -> dict:
    checked = verify_campaign_manifest(source, root=root, expected_outer_sha=expected_outer_sha,
                                       expected_keyset_sha=expected_keyset_sha)
    subset = _external_input_hashes(results)
    if len(subset) != 18 or any(path not in checked or checked[path] != digest
                                for path, digest in subset.items()):
        raise ValueError("frozen scorer input subset not covered by campaign approval")
    if output.exists():
        raise FileExistsError(f"immutable transformed approval already exists: {output}")
    transformed = {
        "terminal": True,
        "parent_approved": True,
        "source_campaign_approval_sha256": _hash(source),
        "adapter_code_sha256": _hash(Path(__file__)),
        "verified_campaign_input_count": len(checked) - len(OPTIONAL_ABSENT),
        "verified_absent_optional_inputs": sorted(OPTIONAL_ABSENT),
        "inputs_sha256": subset,
    }
    with output.open("x") as fh:
        json.dump(transformed, fh, indent=2)
        fh.write("\n")
    return transformed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = adapt(args.source, args.output)
    print(json.dumps({"verified_campaign_inputs": out["verified_campaign_input_count"],
                      "verified_scorer_inputs": len(out["inputs_sha256"]),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
