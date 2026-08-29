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
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .. import router
from ..cloud_cache import CloudCacheTracker, cache_scope, prefix_chain
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
        # Trace schemas are append-only, while CallFeatures deliberately drops
        # unsafe routing inputs. Ignore historical fields such as the removed
        # character-based est_prompt_tokens rather than resurrecting them.
        allowed = set(router.CallFeatures.__dataclass_fields__)
        payload = {
            key: value
            for key, value in record["features"].items()
            if key in allowed
        }
        return router.CallFeatures(**payload)
    return router.extract_features(record["request"], gap)


def reconstruct_cloud_cache(
    records: list[dict[str, Any]], feats: list[router.CallFeatures]
) -> tuple[list[router.CallFeatures], dict[str, Any]]:
    """Rebuild only the cloud history that actually occurred in the trace."""
    tracker = CloudCacheTracker()
    states: Counter[str] = Counter()
    classified = true_positive = false_positive = actual_warm = 0
    token_errors: list[int] = []
    token_percentage_errors: list[float] = []
    lineages: set[str] = set()
    out: list[router.CallFeatures] = []
    # Redacted traces do not retain the credential, and local-placement rows
    # store the local backend URL. Use one opaque recorded-cloud namespace so a
    # local call can still receive the counterfactual cloud-cache prediction.
    replay_scope = cache_scope("replay://recorded-cloud", {})
    for record, feature in zip(records, feats):
        now = float(record.get("ts") or 0.0)
        chain = prefix_chain(record["request"], replay_scope)
        prediction = tracker.probe(chain, now)
        lineages.add(chain.lineage_key)
        states[prediction.state] += 1
        out.append(
            replace(
                feature,
                cloud_cache_state=prediction.state,
                estimated_cloud_cached_tokens=prediction.estimated_read_tokens,
                estimated_cloud_cached_fraction=prediction.estimated_read_fraction,
                cloud_cache_expires_in_s=prediction.expires_in_s,
                cloud_cache_prediction_confidence=(
                    "confirmed" if prediction.state == "warm" else "conservative"
                ),
            )
        )
        if record.get("placement") != "cloud":
            continue
        usage = record.get("usage") or {}
        if "cache_read_input_tokens" in usage:
            classified += 1
            read = int(usage.get("cache_read_input_tokens") or 0)
            is_warm = read > 0
            actual_warm += is_warm
            true_positive += prediction.state == "warm" and is_warm
            false_positive += prediction.state == "warm" and not is_warm
            if prediction.estimated_read_tokens is not None:
                token_errors.append(abs(prediction.estimated_read_tokens - read))
                if read > 0:
                    token_percentage_errors.append(
                        abs(prediction.estimated_read_tokens - read) / read * 100
                    )
        timing = record.get("timing") or {}
        ttft_s = float(timing.get("ttft_ms") or 0.0) / 1000.0
        tracker.observe_cloud_usage(
            chain,
            prediction,
            request_started_at=now,
            response_started_at=now + ttft_s,
            status=int(record.get("status") or 0),
            usage=usage,
        )
    predicted_warm = true_positive + false_positive
    metrics = {
        "states": states,
        "classified": classified,
        "distinct_lineages": len(lineages),
        "warm_precision": true_positive / predicted_warm if predicted_warm else None,
        "warm_recall": true_positive / actual_warm if actual_warm else None,
        "mean_absolute_token_error": (
            sum(token_errors) / len(token_errors) if token_errors else None
        ),
        "token_error_samples": len(token_errors),
        "mean_absolute_percentage_error": (
            sum(token_percentage_errors) / len(token_percentage_errors)
            if token_percentage_errors
            else None
        ),
    }
    return out, metrics


def build_policy(name: str, cap: int, reserve: int, tools: bool) -> router.Policy:
    if name == "static":
        return router.StaticPolicy(
            max_local_tokens=cap,
            output_reserve_tokens=reserve,
            local_can_tool_call=tools,
        )
    return router.build(name)


def label(name: str, cap: int, reserve: int, tools: bool) -> str:
    if name != "static":
        return name
    return f"static cap={cap // 1000}K reserve={reserve} tools={'on' if tools else 'off'}"


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
    print(f"{'cap':>6} {'reserve':>7}   {'tools=off':>14} {'tools=on':>14}")
    for cap in (32_000, 65_536, 131_072):
        for reserve in (0, 512):
            row = []
            for tools in (False, True):
                pol = build_policy("static", cap, reserve, tools)
                loc = sum(pol.decide(f).placement == "local" for f in feats)
                row.append(f"{loc:4d} ({100 * loc / n:5.1f}%)")
            print(f"{cap // 1000:>5}K {reserve:>7}   {row[0]:>14} {row[1]:>14}")


# --------------------------------------------------------------------- cli ---


def main() -> None:
    ap = argparse.ArgumentParser(prog="edgeproxy.trace.replay")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--policy", action="append", help="repeatable; default: static")
    ap.add_argument("--cap", type=int, default=65_536, help="max_local_tokens for static")
    ap.add_argument("--reserve", type=int, default=0, help="dynamic output reserve")
    ap.add_argument("--tools", default="on", choices=["on", "off"])
    ap.add_argument("--sweep", action="store_true", help="grid over cap x reserve x tools")
    ap.add_argument("--check", action="store_true", help="confusion matrix only")
    ap.add_argument("--out", type=Path, help="write per-call decisions as JSONL")
    ap.add_argument("--use-recorded-features", action="store_true")
    ap.add_argument(
        "--cloud-cache-observe",
        action="store_true",
        help="reconstruct provider-confirmed cloud cache state in timestamp order",
    )
    args = ap.parse_args()
    if args.reserve < 0:
        ap.error("--reserve must be non-negative")

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
    if args.cloud_cache_observe:
        feats, cache_metrics = reconstruct_cloud_cache(records, feats)
        states = cache_metrics["states"]
        print(
            "  cloud cache tracker: "
            + ", ".join(f"{state}={count}" for state, count in sorted(states.items()))
        )
        precision = cache_metrics["warm_precision"]
        recall = cache_metrics["warm_recall"]
        mae = cache_metrics["mean_absolute_token_error"]
        mape = cache_metrics["mean_absolute_percentage_error"]
        precision_label = f"{precision:.3f}" if precision is not None else "n/a"
        recall_label = f"{recall:.3f}" if recall is not None else "n/a"
        mae_label = f"{mae:.1f}" if mae is not None else "n/a"
        print(
            f"    provider-scored={cache_metrics['classified']} "
            f"precision={precision_label} recall={recall_label} "
            f"cached-token-MAE={mae_label} n={cache_metrics['token_error_samples']}"
        )
        if mape is not None:
            print(
                f"    cached-token-MAPE={mape:.2f}% "
                f"distinct-lineages={cache_metrics['distinct_lineages']}"
            )
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

    tools = args.tools == "on"

    if args.sweep:
        sweep(records, feats)
        return

    names = args.policy or ["static"]
    last: list[router.Decision] = []
    for name in names:
        pol = build_policy(name, args.cap, args.reserve, tools)
        last = [pol.decide(f) for f in feats]
        if not args.check:
            report(label(name, args.cap, args.reserve, tools), last, len(feats))

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
