"""Cost model for the Phase 1 pilot.

**Local traffic share is NOT cost.** Cost is computed from measured components:

  raw components (reported separately, never summed by default):
    - cloud_input_tokens   -- unavailable from the Lumid gateway (returns 0) -> None
    - cloud_output_tokens  -- measured
    - local_input_tokens   -- measured (uncached + cache-read + cache-creation)
    - local_output_tokens  -- measured
    - local_latency_s      -- measured GPU/model wall time (final-attempt)
    - cloud_latency_s      -- measured wall time
    - switch_penalty       -- cache-aware cold-prefill units on backend flips

  normalized total cost (declared proxy, explicit assumptions in COST_ASSUMPTIONS):
    total = W_CLOUD_OUT * cloud_output_tokens
          + W_LOCAL_OUT * local_output_tokens
          + W_LOCAL_IN  * local_input_tokens
          + switch_multiplier * cold_prefill_tokens

Assumptions are surfaced in every report. The sensitivity curve varies the
local-vs-cloud token price ratio and shows how the quality/cost frontier and
the epsilon-feasible operating point move.
"""

from __future__ import annotations

import numpy as np

from .schema import PairedExample

# --- explicit, declared assumptions --------------------------------------

COST_ASSUMPTIONS = {
    "unit": "normalized token-equivalents per request (declared proxy, not provider billing)",
    "W_CLOUD_OUT": 1.0,   # cloud output token = 1 unit (reference)
    "W_LOCAL_OUT": 0.20,  # local output token cheaper: GPU is rented, no per-token charge; 0.20 reflects amortised GPU-second vs cloud list price (sensitivity-swept below)
    "W_LOCAL_IN": 0.02,   # local input mostly cache-served; small prefill cost
    "cloud_input_price": "unavailable (gateway returns input_tokens:0) -> excluded from total, reported raw as None",
    "switch_penalty": "cold-prefill token units = prompt_est_tokens/1000 charged on each consecutive backend flip, scaled by switch_multiplier (0x and 10x reported)",
    "local_price_ratio_grid": [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
}


def raw_components(examples: list[PairedExample], route_local: list[bool]) -> dict:
    lu = lcr = lcc = lo = co = 0
    llat = clat = 0.0
    n_local = n_cloud = 0
    cloud_in_reported = False
    ci = 0
    for ex, local in zip(examples, route_local):
        g = ex.local_gen if local else ex.cloud_gen
        u = g.usage or {}
        lat = g.final_attempt_latency_s if g.final_attempt_latency_s is not None else (g.latency_s or 0.0)
        if local:
            n_local += 1
            lu += u.get("input_tokens") or 0
            lcr += u.get("cache_read_input_tokens") or 0
            lcc += u.get("cache_creation_input_tokens") or 0
            lo += u.get("output_tokens") or 0
            llat += lat or 0.0
        else:
            n_cloud += 1
            a = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
            if a:
                cloud_in_reported = True
            ci += a
            co += u.get("output_tokens") or 0
            clat += lat or 0.0
    return {
        "n_local": n_local, "n_cloud": n_cloud,
        "cloud_input_tokens": ci if cloud_in_reported else None,
        "cloud_output_tokens": co if n_cloud else None,
        "local_input_tokens": (lu + lcr + lcc) if n_local else None,
        "local_input_uncached_tokens": lu if n_local else None,
        "local_input_cache_read_tokens": lcr if n_local else None,
        "local_input_cache_creation_tokens": lcc if n_local else None,
        "local_output_tokens": lo if n_local else None,
        "local_latency_s_total": round(llat, 2) if n_local else None,
        "cloud_latency_s_total": round(clat, 2) if n_cloud else None,
    }


def token_load(examples: list[PairedExample], route_local: list[bool]) -> dict:
    """Token-*load* split for the tokens actually incurred by the routing
    decision (local tokens for local-routed problems, cloud tokens for
    cloud-routed). A serving-load proxy -- NOT monetary cost.

    Missing usage is explicit, never zero:
      * cloud INPUT tokens are unavailable (Lumid returns input_tokens:0) ->
        reported as None; any share/percent that depends on cloud input is
        ``*_incl_cloud_input = None`` and flagged. An output-only variant
        (both sides measured) is provided as the clean number.
      * a routed problem whose chosen arm's generation was not OK contributes
        nothing and is counted in ``n_missing_usage``.
    """
    li = lo = ci = co = 0
    n_local = n_cloud = n_missing = 0
    cloud_input_seen = False
    for ex, local in zip(examples, route_local):
        g = ex.local_gen if local else ex.cloud_gen
        u = g.usage or {}
        if g.status != "OK" or not u:
            n_missing += 1
        if local:
            n_local += 1
            li += u.get("input_tokens") or 0
            li += u.get("cache_read_input_tokens") or 0
            li += u.get("cache_creation_input_tokens") or 0
            lo += u.get("output_tokens") or 0
        else:
            n_cloud += 1
            cin = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
            if cin:
                cloud_input_seen = True
            ci += cin
            co += u.get("output_tokens") or 0

    def _pct(a, b):
        return (a / b) if b else None

    total_out = lo + co
    total_in_measured = li + (ci if cloud_input_seen else 0)
    total_all_incl_cloud_in = li + lo + ci + co if cloud_input_seen else None
    total_all_output_plus_local_in = li + lo + co  # cloud input excluded (unavailable)

    return {
        "proxy": "serving token-load, not monetary cost",
        # raw counts (preserved)
        "selected_local_input_tokens": li,
        "selected_local_output_tokens": lo,
        "selected_cloud_input_tokens": ci if cloud_input_seen else None,
        "selected_cloud_output_tokens": co,
        "n_local": n_local, "n_cloud": n_cloud, "n_missing_usage": n_missing,
        # local token-load % (local in+out over all selected in+out)
        "local_token_load_pct_incl_cloud_input": (
            _pct(li + lo, total_all_incl_cloud_in) if total_all_incl_cloud_in else None
        ),
        "cloud_token_load_pct_incl_cloud_input": (
            _pct(ci + co, total_all_incl_cloud_in) if total_all_incl_cloud_in else None
        ),
        "cloud_input_unavailable": not cloud_input_seen,
        # output-only variant: both sides measured -> the clean number
        "local_token_load_pct_output_only": _pct(lo, total_out),
        "cloud_token_load_pct_output_only": _pct(co, total_out),
        # local-in-plus-output vs (that + cloud output): cloud input dropped, flagged
        "local_token_load_pct_local_in_plus_all_out": _pct(li + lo, total_all_output_plus_local_in),
        # separate input / output shares. input shares need cloud input, which
        # is unavailable -> None + flag, never silently 1.0.
        "local_input_token_share": _pct(li, li + ci) if cloud_input_seen else None,
        "cloud_input_token_share": _pct(ci, li + ci) if cloud_input_seen else None,
        "input_share_unavailable": not cloud_input_seen,
        "local_output_token_share": _pct(lo, total_out),
        "cloud_output_token_share": _pct(co, total_out),
    }


def comparable_token_load(
    examples: list[PairedExample],
    route_local: list[bool],
    ref_tokens: dict[str, dict],
) -> dict:
    """A COMPLETE, comparable serving-load proxy that does not depend on the
    (missing) Lumid cloud input usage.

    Input side uses a single reference count per task -- the SAME for both
    backends, since the prompt is identical -- from the local vLLM tokenizer
    where reachable, else the deterministic character estimate. Output side
    uses each backend's actually-reported output tokens. Then per policy:

        selected_local  = sum(ref_input[t] for t routed local) + sum(local reported output)
        selected_cloud  = sum(ref_input[t] for t routed cloud) + sum(cloud reported output)
        local_pct       = selected_local / (selected_local + selected_cloud)

    Still a load proxy, not monetary cost.
    """
    lin = lout = cin = cout = 0
    n_local = n_cloud = n_est = n_exact = n_missing_ref = n_missing_out = 0
    for ex, local in zip(examples, route_local):
        tid = ex.problem.task_id
        rt = ref_tokens.get(tid)
        if rt is None:
            n_missing_ref += 1
            rin = int(ex.features.get("prompt_est_tokens", 0.0))
            method = "prompt_est_tokens"
        else:
            rin = int(rt["ref_input_tokens"])
            method = rt.get("method", "prompt_est_tokens")
        n_exact += method == "vllm_tokenizer"
        n_est += method != "vllm_tokenizer"
        g = ex.local_gen if local else ex.cloud_gen
        out = (g.usage or {}).get("output_tokens")
        if out is None:
            n_missing_out += 1
            out = 0
        if local:
            n_local += 1
            lin += rin
            lout += out
        else:
            n_cloud += 1
            cin += rin
            cout += out
    sel_local = lin + lout
    sel_cloud = cin + cout
    tot = sel_local + sel_cloud
    def _p(a, b):
        return (a / b) if b else None
    return {
        "proxy": "complete comparable serving token-load (reference-tokenizer input + reported output); NOT monetary cost",
        "reference_input_method_coverage": {
            "vllm_tokenizer": n_exact, "estimated": n_est,
            "coverage_exact_pct": _p(n_exact, n_exact + n_est),
            "n_tasks_missing_ref_entry": n_missing_ref,
            "n_routed_calls_missing_output_usage": n_missing_out,
        },
        # raw counts (preserved)
        "selected_local_ref_input_tokens": lin,
        "selected_local_output_tokens": lout,
        "selected_cloud_ref_input_tokens": cin,
        "selected_cloud_output_tokens": cout,
        "selected_local_total_tokens": sel_local,
        "selected_cloud_total_tokens": sel_cloud,
        "n_local": n_local, "n_cloud": n_cloud,
        # percentages
        "local_token_load_pct": _p(sel_local, tot),
        "cloud_token_load_pct": _p(sel_cloud, tot),
        "local_input_token_share": _p(lin, lin + cin),
        "cloud_input_token_share": _p(cin, lin + cin),
        "local_output_token_share": _p(lout, lout + cout),
        "cloud_output_token_share": _p(cout, lout + cout),
    }


def _cold_prefill_units(examples: list[PairedExample], route_local: list[bool]) -> float:
    prev = None
    units = 0.0
    for ex, local in zip(examples, route_local):
        if prev is not None and local != prev:
            units += float(ex.features.get("prompt_est_tokens", 0.0)) / 1000.0
        prev = local
    return units


def normalized_total_cost(
    examples: list[PairedExample],
    route_local: list[bool],
    switch_multiplier: float,
    w_local_out: float | None = None,
    w_local_in: float | None = None,
) -> float:
    c = raw_components(examples, route_local)
    w_lo = COST_ASSUMPTIONS["W_LOCAL_OUT"] if w_local_out is None else w_local_out
    w_li = COST_ASSUMPTIONS["W_LOCAL_IN"] if w_local_in is None else w_local_in
    total = COST_ASSUMPTIONS["W_CLOUD_OUT"] * (c["cloud_output_tokens"] or 0)
    total += w_lo * (c["local_output_tokens"] or 0)
    total += w_li * (c["local_input_tokens"] or 0)
    total += switch_multiplier * _cold_prefill_units(examples, route_local)
    n = len(examples) or 1
    return total / n  # per-request


def sensitivity_curve(
    examples: list[PairedExample],
    route_local: list[bool],
    switch_multiplier: float,
) -> list[dict]:
    """Vary the local-output token price ratio (W_LOCAL_OUT) over the declared
    grid; report per-request normalized cost at each. Quality and local share
    are fixed by the routing decision, so this isolates the price assumption."""
    out = []
    for ratio in COST_ASSUMPTIONS["local_price_ratio_grid"]:
        out.append({
            "local_output_price_ratio": ratio,
            "norm_cost_per_request": normalized_total_cost(
                examples, route_local, switch_multiplier,
                w_local_out=ratio, w_local_in=ratio * 0.1,
            ),
        })
    return out
