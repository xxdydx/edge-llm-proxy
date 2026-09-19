"""Scoring, baselines, epsilon-constrained threshold selection, and the
rolling-window online sim for the Phase 1 pilot.

`plus_pass` (base MBPP inputs + EvalPlus expanded inputs all correct) is the
correctness label throughout. Quality of a routing rule = mean plus_pass of
whichever arm it chose per problem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .config import Phase1Config
from .policy import (
    always_cloud,
    always_local,
    fit_gbm,
    fit_logreg,
    fit_no_harm_gbm,
    fit_no_harm_logreg,
    oracle,
    static_heuristic,
)
from .cost import (
    COST_ASSUMPTIONS,
    comparable_token_load,
    normalized_total_cost,
    raw_components,
    sensitivity_curve,
    token_load,
)
from .schema import PairedExample
from .switch_cost import cost_curve_over_local_share, sequence_cost


# --- primitives -----------------------------------------------------------

def _quality(examples: list[PairedExample], route_local: list[bool]) -> float:
    if not examples:
        return 0.0
    ok = sum((ex.local_ok if lo else ex.cloud_ok) for ex, lo in zip(examples, route_local))
    return ok / len(examples)


def _local_share(route_local: list[bool]) -> float:
    return sum(route_local) / len(route_local) if route_local else 0.0


def opportunity_2x2(examples: list[PairedExample]) -> dict[str, int]:
    c = {"both": 0, "local_only": 0, "cloud_only": 0, "neither": 0}
    for ex in examples:
        if ex.local_ok and ex.cloud_ok:
            c["both"] += 1
        elif ex.local_ok:
            c["local_only"] += 1
        elif ex.cloud_ok:
            c["cloud_only"] += 1
        else:
            c["neither"] += 1
    c["total"] = len(examples)
    return c


def _token_totals(examples: list[PairedExample], route_local: list[bool]) -> dict:
    """Same bucket semantics as capability_router's fixed aggregator: local
    input = uncached + cache-read + cache-creation; cloud input is unavailable
    (Lumid returns 0) -> None, not a false zero."""
    lu = lcr = lcc = lo = co = ci = 0
    any_local = any_cloud = cloud_reported = False
    for ex, local in zip(examples, route_local):
        u = (ex.local_gen.usage if local else ex.cloud_gen.usage) or {}
        if not u:
            continue
        if local:
            any_local = True
            lu += u.get("input_tokens") or 0
            lcr += u.get("cache_read_input_tokens") or 0
            lcc += u.get("cache_creation_input_tokens") or 0
            lo += u.get("output_tokens") or 0
        else:
            any_cloud = True
            a = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
            if a:
                cloud_reported = True
            ci += a
            co += u.get("output_tokens") or 0
    return {
        "local_input_tokens": (lu + lcr + lcc) if any_local else None,
        "local_input_uncached_tokens": lu if any_local else None,
        "local_input_cache_read_tokens": lcr if any_local else None,
        "local_input_cache_creation_tokens": lcc if any_local else None,
        "local_output_tokens": lo if any_local else None,
        "cloud_input_tokens": ci if (any_cloud and cloud_reported) else None,
        "cloud_input_tokens_note": None if cloud_reported else "gateway reports input_tokens:0; unavailable, not zero",
        "cloud_output_tokens": co if any_cloud else None,
    }


def _latency(examples: list[PairedExample], route_local: list[bool]) -> dict:
    xs = []
    for ex, local in zip(examples, route_local):
        g = ex.local_gen if local else ex.cloud_gen
        v = g.final_attempt_latency_s if g.final_attempt_latency_s is not None else g.latency_s
        if v is not None:
            xs.append(v)
    if not xs:
        return {"mean_s": None, "p50_s": None, "p95_s": None}
    a = np.array(xs)
    return {"mean_s": float(a.mean()), "p50_s": float(np.percentile(a, 50)), "p95_s": float(np.percentile(a, 95))}


@dataclass
class ArmMetrics:
    name: str
    threshold: float | None
    local_share: float
    quality: float
    quality_delta_vs_cloud: float
    within_epsilon: bool
    opportunity: dict
    raw_cost_components: dict  # cloud/local input+output tokens, latency -- NOT summed
    token_load: dict  # provider-reported token split; cloud input unavailable -> flagged
    comparable_token_load: dict  # complete proxy: reference-tokenizer input + reported output
    tokens: dict
    latency: dict
    switch_cost: dict  # sequence_cost keyed by multiplier (0x / 10x)
    norm_cost_per_request: dict  # normalized total cost keyed by multiplier
    norm_cost_sensitivity: dict  # price-ratio sweep keyed by multiplier

    def to_json(self) -> dict:
        return asdict(self)


def _arm(name, examples, route_local, q_cloud, cfg, threshold=None, ref_tokens=None) -> ArmMetrics:
    q = _quality(examples, route_local)
    sc, ncpr, ncs = {}, {}, {}
    for mult in cfg.switch_cost_multipliers:
        sc[str(mult)] = sequence_cost(examples, route_local, mult)
        ncpr[str(mult)] = normalized_total_cost(examples, route_local, mult)
        ncs[str(mult)] = sensitivity_curve(examples, route_local, mult)
    return ArmMetrics(
        name=name,
        threshold=threshold,
        local_share=_local_share(route_local),
        quality=q,
        quality_delta_vs_cloud=q - q_cloud,
        within_epsilon=bool(q >= q_cloud - cfg.epsilon),
        opportunity=opportunity_2x2([e for e, lo in zip(examples, route_local) if lo]) if any(route_local) else {},
        raw_cost_components=raw_components(examples, route_local),
        token_load=token_load(examples, route_local),
        comparable_token_load=comparable_token_load(examples, route_local, ref_tokens or {}),
        tokens=_token_totals(examples, route_local),
        latency=_latency(examples, route_local),
        switch_cost=sc,
        norm_cost_per_request=ncpr,
        norm_cost_sensitivity=ncs,
    )


# --- epsilon-constrained threshold on VAL, reported on TEST -----------------

def _best_threshold(policy, val: list[PairedExample], q_cloud_val: float, cfg: Phase1Config) -> float:
    """Among thresholds whose VAL quality is >= q_cloud_val - epsilon, take the
    one admitting the most local traffic. If none qualify, take the threshold
    with the highest VAL quality (most conservative)."""
    best_ok, best_share = None, -1.0
    fallback, fb_q = 1.01, -1.0
    for thr in cfg.threshold_grid:
        route = policy.route(val, thr)
        q = _quality(val, route)
        share = _local_share(route)
        if q > fb_q:
            fb_q, fallback = q, thr
        if q >= q_cloud_val - cfg.epsilon and share > best_share:
            best_ok, best_share = thr, share
    return best_ok if best_ok is not None else fallback


def epsilon_sensitivity(
    train: list[PairedExample],
    val: list[PairedExample],
    test: list[PairedExample],
    cfg: Phase1Config,
    learned: dict,
    epsilons: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15),
) -> dict:
    """Epsilon is a policy choice, not a fact about the model -- the router's
    operator (the "end user" of this pilot's output) trades quality risk for
    local share, and 0.02 is one point on that curve, not the only defensible
    one. For each learned policy, sweep epsilon, re-select the VAL threshold
    under each (`_best_threshold` already takes epsilon from `cfg`; only the
    threshold selection depends on it, so the classifier itself is fit once),
    and report the achieved TEST local share/quality/violation at each. This
    is the artifact that would back a runtime `--epsilon` knob."""
    q_cloud_val = _quality(val, always_cloud(val))
    q_cloud_test = _quality(test, always_cloud(test))
    out: dict = {}
    for pname, pol in learned.items():
        rows = []
        for eps in epsilons:
            cfg_e = replace(cfg, epsilon=eps)
            thr = _best_threshold(pol, val, q_cloud_val, cfg_e)
            route = pol.route(test, thr)
            q = _quality(test, route)
            rows.append(
                {
                    "epsilon": eps,
                    "threshold": thr,
                    "local_share": _local_share(route),
                    "quality": q,
                    "quality_delta_vs_cloud": q - q_cloud_test,
                    "within_epsilon": bool(q >= q_cloud_test - eps),
                }
            )
        out[pname] = rows
    return out


def run_analysis(
    train: list[PairedExample],
    val: list[PairedExample],
    test: list[PairedExample],
    ordered_all: list[PairedExample],
    cfg: Phase1Config,
    ref_tokens: dict | None = None,
) -> dict:
    q_cloud_test = _quality(test, always_cloud(test))
    q_cloud_val = _quality(val, always_cloud(val))

    report: dict = {
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "q_cloud_test": q_cloud_test,
        "opportunity_2x2_test": opportunity_2x2(test),
        "epsilon": cfg.epsilon,
        "arms_test": {},
    }

    # deterministic baselines on TEST
    for name, fn in (
        ("always_cloud", always_cloud),
        ("always_local", always_local),
        ("static_heuristic", static_heuristic),
        ("oracle", oracle),
    ):
        report["arms_test"][name] = _arm(name, test, fn(test), q_cloud_test, cfg, ref_tokens=ref_tokens).to_json()

    # learned policies: fit on TRAIN, threshold on VAL, report on TEST
    learned = {
        "logreg": fit_logreg(train),
        "no_harm_logreg": fit_no_harm_logreg(train),
    }
    gbm = fit_gbm(train)
    if gbm is not None:
        learned["gbm"] = gbm
    no_harm_gbm = fit_no_harm_gbm(train)
    if no_harm_gbm is not None:
        learned["no_harm_gbm"] = no_harm_gbm
    thresholds = {}
    for pname, pol in learned.items():
        thr = _best_threshold(pol, val, q_cloud_val, cfg)
        thresholds[pname] = thr
        route = pol.route(test, thr)
        report["arms_test"][pname] = _arm(pname, test, route, q_cloud_test, cfg, threshold=thr, ref_tokens=ref_tokens).to_json()
    report["learned_val_thresholds"] = thresholds
    report["epsilon_sensitivity"] = epsilon_sensitivity(train, val, test, cfg, learned)
    report["learned_note"] = (
        "logreg / gbm estimate P(local plus_pass); no_harm_logreg / "
        "no_harm_gbm estimate P(not [cloud passes and local fails]), directly "
        "targeting harmful quadrant B. All are this experiment's own "
        "request-only classifiers, NOT routing-ladder Policy 3/4/5/6."
    )

    # --- required curves ---------------------------------------------------
    report["cost_assumptions"] = COST_ASSUMPTIONS
    report["curves"] = {}
    # 1. quality vs local-traffic share, admitting problems by each learned
    #    policy's P(local) descending, and by the oracle order.
    for pname, pol in learned.items():
        p_local = pol.proba_local(test)
        order_desc = list(np.argsort(-p_local))
        for mult in cfg.switch_cost_multipliers:
            report["curves"][f"quality_vs_local_share__{pname}__switch{mult}x"] = \
                cost_curve_over_local_share(test, order_desc, mult)
    orc_order = sorted(range(len(test)), key=lambda i: not test[i].local_ok)  # local-passing first
    for mult in cfg.switch_cost_multipliers:
        report["curves"][f"quality_vs_local_share__oracle__switch{mult}x"] = \
            cost_curve_over_local_share(test, orc_order, mult)

    # 2. quality-cost Pareto across routing thresholds (cost is normalized
    #    total cost per request -- NOT local share) + per-threshold token-load
    #    split (serving-load proxy, missing usage explicit).
    for pname, pol in learned.items():
        p_local = pol.proba_local(test)
        for mult in cfg.switch_cost_multipliers:
            pts = []
            for thr in cfg.threshold_grid:
                route = [bool(v >= thr) for v in p_local]
                q = _quality(test, route)
                tl = token_load(test, route)
                ctl = comparable_token_load(test, route, ref_tokens or {})
                pts.append({
                    "threshold": thr,
                    "local_request_share": _local_share(route),
                    "quality": q,
                    "quality_delta_vs_cloud": q - q_cloud_test,
                    "within_epsilon": bool(q >= q_cloud_test - cfg.epsilon),
                    "norm_cost_per_request": normalized_total_cost(test, route, mult),
                    "switch_rate": sequence_cost(test, route, mult)["switch_rate"],
                    # complete comparable token-load (main proxy: reference-tokenizer
                    # input + reported output; independent of missing cloud input)
                    "local_comparable_token_load_pct": ctl["local_token_load_pct"],
                    "cloud_comparable_token_load_pct": ctl["cloud_token_load_pct"],
                    "comparable_local_input_token_share": ctl["local_input_token_share"],
                    "comparable_local_output_token_share": ctl["local_output_token_share"],
                    "sel_local_ref_input_tokens": ctl["selected_local_ref_input_tokens"],
                    "sel_local_output_tokens_c": ctl["selected_local_output_tokens"],
                    "sel_cloud_ref_input_tokens": ctl["selected_cloud_ref_input_tokens"],
                    "sel_cloud_output_tokens_c": ctl["selected_cloud_output_tokens"],
                    "ref_input_exact_coverage_pct": ctl["reference_input_method_coverage"]["coverage_exact_pct"],
                    # provider-reported split (cloud input unavailable -> flagged)
                    "local_token_load_pct_output_only": tl["local_token_load_pct_output_only"],
                    "local_token_load_pct_incl_cloud_input": tl["local_token_load_pct_incl_cloud_input"],
                    "cloud_input_unavailable": tl["cloud_input_unavailable"],
                    "provider_local_output_token_share": tl["local_output_token_share"],
                    "provider_cloud_output_token_share": tl["cloud_output_token_share"],
                    "sel_local_input_tokens_provider": tl["selected_local_input_tokens"],
                    "sel_local_output_tokens_provider": tl["selected_local_output_tokens"],
                    "sel_cloud_input_tokens_provider": tl["selected_cloud_input_tokens"],
                    "sel_cloud_output_tokens_provider": tl["selected_cloud_output_tokens"],
                    "n_missing_usage": tl["n_missing_usage"],
                })
            report["curves"][f"quality_cost_pareto__{pname}__switch{mult}x"] = pts

    # 2b. quality vs local TOTAL-TOKEN share (distinct from request share):
    #     x = local_token_load_pct_output_only (both sides measured).
    for pname, pol in learned.items():
        p_local = pol.proba_local(test)
        rows = []
        for thr in cfg.threshold_grid:
            route = [bool(v >= thr) for v in p_local]
            tl = token_load(test, route)
            rows.append({
                "threshold": thr,
                "local_request_share": _local_share(route),
                "local_token_share_output_only": tl["local_token_load_pct_output_only"],
                "local_token_load_pct_incl_cloud_input": tl["local_token_load_pct_incl_cloud_input"],
                "quality": _quality(test, route),
            })
        report["curves"][f"quality_vs_local_token_share__{pname}"] = rows

    # 2c. quality vs the COMPLETE comparable token-load share (reference-
    #     tokenizer input + reported output; independent of missing cloud
    #     input usage). This is the user's main load curve.
    for pname, pol in learned.items():
        p_local = pol.proba_local(test)
        rows = []
        for thr in cfg.threshold_grid:
            route = [bool(v >= thr) for v in p_local]
            ctl = comparable_token_load(test, route, ref_tokens or {})
            rows.append({
                "threshold": thr,
                "local_request_share": _local_share(route),
                "local_comparable_token_load_pct": ctl["local_token_load_pct"],
                "cloud_comparable_token_load_pct": ctl["cloud_token_load_pct"],
                "sel_local_ref_input_tokens": ctl["selected_local_ref_input_tokens"],
                "sel_local_output_tokens": ctl["selected_local_output_tokens"],
                "sel_cloud_ref_input_tokens": ctl["selected_cloud_ref_input_tokens"],
                "sel_cloud_output_tokens": ctl["selected_cloud_output_tokens"],
                "ref_input_exact_coverage_pct": ctl["reference_input_method_coverage"]["coverage_exact_pct"],
                "quality": _quality(test, route),
            })
        report["curves"][f"quality_vs_comparable_token_share__{pname}"] = rows

    report["token_load_note"] = (
        "token-load is a SERVING-LOAD proxy, not monetary cost. The MAIN "
        "number per threshold is `local_comparable_token_load_pct`: selected "
        "local (reference-tokenizer input + reported output) vs selected cloud "
        "(reference-tokenizer input + reported output), where the reference "
        "input count is a SINGLE count of the identical prompt per task -- "
        "from the local vLLM tokenizer where reachable, else the deterministic "
        "character estimate (method + `ref_input_exact_coverage_pct` reported). "
        "This is always computable and never depends on the missing Lumid "
        "cloud input usage. A provider-reported variant is also carried: its "
        "cloud INPUT is None + `cloud_input_unavailable=true` (never zero), so "
        "only its output-only split is a clean measured number. Raw local/cloud "
        "input and output token counts are preserved on every threshold row."
    )

    # 3. cache-switch-cost sensitivity is already carried per-arm at 0x and
    #    10x (switch_cost / norm_cost_per_request / norm_cost_sensitivity).

    # existence verdict from the oracle
    orc = report["arms_test"]["oracle"]
    report["existence_verdict"] = {
        "oracle_local_share": orc["local_share"],
        "oracle_quality": orc["quality"],
        "oracle_quality_delta_vs_cloud": orc["quality_delta_vs_cloud"],
        "meaningful_local_traffic_at_cloud_quality": bool(
            orc["local_share"] >= 0.10 and orc["quality_delta_vs_cloud"] >= -cfg.epsilon
        ),
    }

    # rolling-window online sim over selection order (expanding-only baseline,
    # kept for backward compatibility with existing readers of this key)
    report["rolling_window"] = _rolling_window(ordered_all, cfg)

    # step 4 prep: all four window-update strategies compared side by side
    # (see [[Window-update strategy comparison for the rolling-window
    # router]]). change_point_reset is the evidence-based recommendation --
    # ties the rest stationary, wins on local-share recovery under drift -- so
    # it is flagged, not silently substituted for the expanding baseline above.
    from .window_policies import compare_window_strategies
    report["rolling_window_strategies"] = {
        "strategies": compare_window_strategies(ordered_all, cfg),
        "recommended": "change_point_reset",
        "note": (
            "expanding/sliding/recency_decay/change_point_reset compared "
            "stationary and under a synthetic injected cloud-drift (no live "
            "second cloud backend yet). All four tie stationary at this "
            "corpus size; change_point_reset (threshold tuned to 0.10) "
            "recovers local share fastest after a regime change. See the "
            "linked finding for numbers and caveats."
        ),
    }

    # step 3: joint quality + capacity controller (offline sim over a
    # simulated offered load; no GPU). Quality gate = the learned classifier +
    # its VAL threshold; capacity gate = a GPU-service-second budget anchored
    # to the measured 27B envelope.
    from .capacity import joint_capacity_report
    q_cloud_all = _quality(ordered_all, always_cloud(ordered_all))
    report["joint_capacity"] = joint_capacity_report(
        ordered_all, learned, thresholds, cfg, q_cloud_all, stream_label="ordered_all"
    )
    return report


def _rolling_window(ordered_all: list[PairedExample], cfg: Phase1Config) -> dict:
    """Uninformed policy on window 0; from window k>=1 fit logreg on all
    problems seen so far (windows < k) and evaluate on window k. Compare each
    window's routed quality/cost against always-cloud / always-local / oracle
    on that same window."""
    by_win: dict[int, list[PairedExample]] = {}
    for ex in ordered_all:
        by_win.setdefault(ex.problem.window, []).append(ex)
    windows = sorted(by_win)
    rows = []
    seen: list[PairedExample] = []
    for w in windows:
        cur = by_win[w]
        if not seen or len({e.local_ok for e in seen}) < 2:
            route = always_cloud(cur)  # uninformed / not enough signal yet
            policy_name = "uninformed(always_cloud)"
            thr = None
        else:
            pol = fit_logreg(seen)
            q_cloud_seen = _quality(seen, always_cloud(seen))
            thr = _best_threshold(pol, seen, q_cloud_seen, cfg)
            route = pol.route(cur, thr)
            policy_name = f"logreg_fit_on_windows<{w}"
        q_cloud = _quality(cur, always_cloud(cur))
        row = {
            "window": w,
            "n": len(cur),
            "policy": policy_name,
            "threshold": thr,
            "routed_quality": _quality(cur, route),
            "routed_local_share": _local_share(route),
            "q_always_cloud": q_cloud,
            "q_always_local": _quality(cur, always_local(cur)),
            "q_oracle": _quality(cur, oracle(cur)),
            "oracle_local_share": _local_share(oracle(cur)),
            "within_epsilon": bool(_quality(cur, route) >= q_cloud - cfg.epsilon),
        }
        for mult in cfg.switch_cost_multipliers:
            row[f"routed_mean_cost@{mult}x"] = sequence_cost(cur, route, mult)["mean_cost_per_request"]
        rows.append(row)
        seen = seen + cur
    return {"windows": rows}
