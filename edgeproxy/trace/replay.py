"""Replay recorded calls through candidate policies. No GPU, no network.

    python -m edgeproxy.trace.replay traces/*.jsonl --policy static
    python -m edgeproxy.trace.replay traces/*.jsonl --sweep

The recorded request holds everything the router looks at, so "what would
policy X have done" is answerable offline over real traffic.

Placement only — latency and cost need a cost model that does not exist yet.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .. import router
from .inspect import load


def calls(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in records
        if r.get("path") == "/v1/messages" and isinstance(r.get("request"), dict)
    ]


def session_gaps(records: list[dict[str, Any]]) -> list[float | None]:
    """Seconds since the previous call in the same session.

    The live proxy tracks this in memory; offline it has to be reconstructed
    from `ts` and the session header, or replayed features would differ from
    recorded ones and the confusion matrix would report spurious mismatches.
    """
    last: dict[str, float] = {}
    out: list[float | None] = []
    for r in sorted(records, key=lambda x: x.get("ts") or 0):
        sid = (r.get("headers") or {}).get("x-claude-code-session-id")
        ts = r.get("ts")
        if not sid or ts is None:
            out.append(None)
            continue
        prev = last.get(sid)
        out.append(round(ts - prev, 1) if prev is not None else None)
        last[sid] = ts
    return out


def features_for(
    record: dict[str, Any], use_recorded: bool, gap: float | None = None
) -> router.CallFeatures:
    """Recompute by default: older records predate the `features` field, and
    recomputing keeps every policy on identical footing."""
    if use_recorded and isinstance(record.get("features"), dict):
        return router.CallFeatures(**record["features"])
    return router.extract_features(record["request"], gap)


def build_policy(name: str, cap: int, clamp: int | None, tools: bool) -> router.Policy:
    if name == "static":
        return router.StaticPolicy(
            max_local_tokens=cap, clamp_max_tokens=clamp, local_can_tool_call=tools
        )
    return router.build(name)


def label(name: str, cap: int, clamp: int | None, tools: bool) -> str:
    if name != "static":
        return name
    return f"static cap={cap // 1000}K clamp={clamp or 'off'} tools={'on' if tools else 'off'}"


# ------------------------------------------------------------------ report ---


def report(name: str, decisions: list[router.Decision], n: int) -> None:
    placements = Counter(d.placement for d in decisions)
    reasons = Counter(d.reason for d in decisions)
    loc = placements["local"]
    print(f"\n{name}")
    print(f"  local  {loc:5d} ({100 * loc / n:5.1f}%)   cloud {placements['cloud']:5d}")
    for reason, count in reasons.most_common():
        print(f"    {reason:20s} {count:5d} ({100 * count / n:5.1f}%)")


def confusion(records: list[dict[str, Any]], decisions: list[router.Decision]) -> None:
    """Recorded placement vs. what the same policy decides now.

    Any mismatch means extract_features drifted from what produced the trace,
    or the decision was not deterministic. This is the harness's own test.
    """
    pairs = Counter()
    for rec, dec in zip(records, decisions):
        actual = rec.get("placement")
        if actual is None:
            continue
        pairs[(actual, dec.placement)] += 1

    if not pairs:
        print("\nconfusion: no records carry a recorded placement (pre-router traces)")
        return

    total = sum(pairs.values())
    mismatched = sum(v for (a, b), v in pairs.items() if a != b)
    print(f"\nconfusion  (recorded -> replayed, {total} annotated records)")
    for (a, b), v in sorted(pairs.items()):
        flag = "" if a == b else "   <- MISMATCH"
        print(f"    {a:6s} -> {b:6s} {v:5d}{flag}")
    print(f"  {mismatched} mismatched")


def sweep(records: list[dict[str, Any]], feats: list[router.CallFeatures]) -> None:
    n = len(feats)
    for name in ("cloud-only", "local-only"):
        pol = router.build(name)
        loc = sum(pol.decide(f).placement == "local" for f in feats)
        print(f"{name:34s} {loc:5d} ({100 * loc / n:5.1f}%)")

    print()
    print(f"{'cap':>6} {'clamp':>7}   {'tools=off':>14} {'tools=on':>14}")
    for cap in (32_000, 65_536, 131_072):
        for clamp in (None, 4096):
            row = []
            for tools in (False, True):
                pol = build_policy("static", cap, clamp, tools)
                loc = sum(pol.decide(f).placement == "local" for f in feats)
                row.append(f"{loc:4d} ({100 * loc / n:5.1f}%)")
            print(f"{cap // 1000:>5}K {str(clamp or 'off'):>7}   {row[0]:>14} {row[1]:>14}")


# --------------------------------------------------------------------- cli ---


def main() -> None:
    ap = argparse.ArgumentParser(prog="edgeproxy.trace.replay")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--policy", action="append", help="repeatable; default: static")
    ap.add_argument("--cap", type=int, default=65_536, help="max_local_tokens for static")
    ap.add_argument("--clamp", default="4096", help="max_tokens clamp, or 'off'")
    ap.add_argument("--tools", default="on", choices=["on", "off"])
    ap.add_argument("--sweep", action="store_true", help="grid over cap x clamp x tools")
    ap.add_argument("--check", action="store_true", help="confusion matrix only")
    ap.add_argument("--out", type=Path, help="write per-call decisions as JSONL")
    ap.add_argument("--use-recorded-features", action="store_true")
    args = ap.parse_args()

    existing = [p for p in args.paths if p.exists()]
    if not existing:
        raise SystemExit("no trace files found")

    records = calls(load(existing))
    if not records:
        raise SystemExit("no /v1/messages records with a recorded request body")
    records.sort(key=lambda r: r.get("ts") or 0)
    gaps = session_gaps(records)
    feats = [
        features_for(r, args.use_recorded_features, g) for r, g in zip(records, gaps)
    ]
    print(f"{len(records)} calls from {len(existing)} file(s)")

    # TTL is per-request, so each call is compared against its own, not a
    # global constant.
    warm = cold = first = uncached = 0
    for f in feats:
        if f.cloud_cache_ttl_s is None:
            uncached += 1
        elif f.seconds_since_last_call is None:
            first += 1
        elif f.seconds_since_last_call <= f.cloud_cache_ttl_s:
            warm += 1
        else:
            cold += 1
    print(
        f"  cloud cache: {warm} within TTL, {cold} past TTL, "
        f"{first} first-in-session, {uncached} no breakpoints"
    )
    ttls = Counter(f.cloud_cache_ttl_s for f in feats if f.cloud_cache_ttl_s)
    if ttls:
        spread = ", ".join(f"{int(t)}s x{c}" for t, c in sorted(ttls.items()))
        print(f"    TTLs requested: {spread}")

    clamp = None if args.clamp == "off" else int(args.clamp)
    tools = args.tools == "on"

    if args.sweep:
        sweep(records, feats)
        return

    names = args.policy or ["static"]
    last: list[router.Decision] = []
    for name in names:
        pol = build_policy(name, args.cap, clamp, tools)
        last = [pol.decide(f) for f in feats]
        if not args.check:
            report(label(name, args.cap, clamp, tools), last, len(feats))

    if args.check or len(names) == 1:
        confusion(records, last)

    if args.out:
        with args.out.open("w", encoding="utf-8") as fh:
            for rec, feat, dec in zip(records, feats, last):
                fh.write(
                    json.dumps(
                        {
                            "id": rec.get("id"),
                            "ts": rec.get("ts"),
                            "session": rec.get("headers", {}).get("x-claude-code-session-id"),
                            "recorded_placement": rec.get("placement"),
                            "placement": dec.placement,
                            "reason": dec.reason,
                            "detail": dec.detail,
                            **feat.__dict__,
                        }
                    )
                    + "\n"
                )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
