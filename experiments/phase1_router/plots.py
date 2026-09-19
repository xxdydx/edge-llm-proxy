"""Required Phase 1 plots (+ CSVs):

  1. quality vs local-traffic share (per learned policy + oracle, at 0x/10x switch)
  2. quality-cost Pareto across routing thresholds (cost = normalized total
     cost per request, NOT local share; epsilon-feasible points marked)
  3. cache-switch-cost sensitivity: normalized cost per request vs the
     local-output price ratio, at switch multiplier 0x and 10x
  + rolling-window: routed vs always-cloud vs oracle quality over windows

Every plot also writes its underlying CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path_noext: Path):
    path_noext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_noext}.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{path_noext}.pdf", bbox_inches="tight")
    plt.close(fig)


def _csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_all(report: dict, out_dir: Path) -> dict[str, str]:
    pdir = out_dir / "plots"
    made: dict[str, str] = {}
    curves = report.get("curves", {})

    # 1. quality vs local share
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, rows in sorted(curves.items()):
        if not key.startswith("quality_vs_local_share__"):
            continue
        label = key.replace("quality_vs_local_share__", "")
        ax.plot([r["local_share"] for r in rows], [r["quality"] for r in rows],
                marker="o", ms=3, label=label)
        _csv(rows, pdir / f"{key}.csv")
    qc = report.get("q_cloud_test")
    if qc is not None:
        ax.axhline(qc, ls="--", c="gray", lw=1, label=f"always-cloud q={qc:.3f}")
        ax.axhline(qc - report.get("epsilon", 0.02), ls=":", c="red", lw=1, label="q_cloud - eps")
    ax.set_xlabel("local traffic share"); ax.set_ylabel("quality (plus_pass rate)")
    ax.set_title("Quality vs local-traffic share"); ax.legend(fontsize=7)
    _save(fig, pdir / "quality_vs_local_share")
    made["quality_vs_local_share"] = str(pdir / "quality_vs_local_share")

    # 1b. quality vs local TOTAL-TOKEN share (distinct from request share)
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for key, rows in sorted(curves.items()):
        if not key.startswith("quality_vs_local_token_share__"):
            continue
        label = key.replace("quality_vs_local_token_share__", "")
        xs = [r["local_token_share_output_only"] for r in rows if r["local_token_share_output_only"] is not None]
        ys = [r["quality"] for r in rows if r["local_token_share_output_only"] is not None]
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, label=f"{label} (output-token share)")
            plotted = True
        _csv(rows, pdir / f"{key}.csv")
    if qc is not None:
        ax.axhline(qc, ls="--", c="gray", lw=1, label=f"always-cloud q={qc:.3f}")
        ax.axhline(qc - report.get("epsilon", 0.02), ls=":", c="red", lw=1)
    ax.set_xlabel("local share of selected OUTPUT tokens  (load proxy, not cost)")
    ax.set_ylabel("quality (plus_pass rate)")
    ax.set_title("Quality vs local token-load share\n(cloud input tokens unavailable; output-token share shown)")
    if plotted:
        ax.legend(fontsize=7)
    _save(fig, pdir / "quality_vs_local_token_share")
    made["quality_vs_local_token_share"] = str(pdir / "quality_vs_local_token_share")

    # 1c. quality vs the COMPLETE comparable token-load share (main load curve)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, rows in sorted(curves.items()):
        if not key.startswith("quality_vs_comparable_token_share__"):
            continue
        label = key.replace("quality_vs_comparable_token_share__", "")
        xs = [r["local_comparable_token_load_pct"] for r in rows if r["local_comparable_token_load_pct"] is not None]
        ys = [r["quality"] for r in rows if r["local_comparable_token_load_pct"] is not None]
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, label=label)
        _csv(rows, pdir / f"{key}.csv")
    if qc is not None:
        ax.axhline(qc, ls="--", c="gray", lw=1, label=f"always-cloud q={qc:.3f}")
        ax.axhline(qc - report.get("epsilon", 0.02), ls=":", c="red", lw=1)
    cov = None
    for k, rows in curves.items():
        if k.startswith("quality_vs_comparable_token_share__") and rows:
            cov = rows[0].get("ref_input_exact_coverage_pct")
            break
    ax.set_xlabel("local share of comparable total tokens\n(reference-tokenizer input + reported output; load proxy, not cost)")
    ax.set_ylabel("quality (plus_pass rate)")
    ttl = "Quality vs local comparable token-load share"
    if cov is not None:
        ttl += f"\n(reference input: {cov*100:.0f}% exact vLLM tokenizer, rest estimated)"
    ax.set_title(ttl)
    ax.legend(fontsize=7)
    _save(fig, pdir / "quality_vs_comparable_token_share")
    made["quality_vs_comparable_token_share"] = str(pdir / "quality_vs_comparable_token_share")

    # 2. quality-cost Pareto across thresholds
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, rows in sorted(curves.items()):
        if not key.startswith("quality_cost_pareto__"):
            continue
        label = key.replace("quality_cost_pareto__", "")
        xs = [r["norm_cost_per_request"] for r in rows]
        ys = [r["quality"] for r in rows]
        ax.plot(xs, ys, marker=".", ms=4, lw=0.8, label=label, alpha=0.8)
        feas = [(r["norm_cost_per_request"], r["quality"]) for r in rows if r["within_epsilon"]]
        if feas:
            ax.scatter([p[0] for p in feas], [p[1] for p in feas], s=18, c="green", zorder=5)
        _csv(rows, pdir / f"{key}.csv")
    if qc is not None:
        ax.axhline(qc, ls="--", c="gray", lw=1)
        ax.axhline(qc - report.get("epsilon", 0.02), ls=":", c="red", lw=1)
    ax.set_xlabel("normalized total cost per request (declared proxy)")
    ax.set_ylabel("quality (plus_pass rate)")
    ax.set_title("Quality-cost Pareto across routing thresholds\n(green = within epsilon of always-cloud)")
    ax.legend(fontsize=7)
    _save(fig, pdir / "quality_cost_pareto")
    made["quality_cost_pareto"] = str(pdir / "quality_cost_pareto")

    # 3. cache-switch-cost sensitivity (norm cost vs price ratio, 0x vs 10x)
    arms = report.get("arms_test", {})
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm_name in ("oracle", "logreg", "gbm", "always_local"):
        arm = arms.get(arm_name)
        if not arm:
            continue
        for mult, rows in sorted(arm.get("norm_cost_sensitivity", {}).items()):
            ax.plot([r["local_output_price_ratio"] for r in rows],
                    [r["norm_cost_per_request"] for r in rows],
                    marker="o", ms=3, label=f"{arm_name} @ switch {mult}x")
            _csv(rows, pdir / f"switch_sensitivity__{arm_name}__{mult}x.csv")
    ax.set_xlabel("local-output token price ratio (vs cloud output = 1.0)")
    ax.set_ylabel("normalized total cost per request")
    ax.set_title("Cache-switch-cost sensitivity (0x vs 10x cold-prefill penalty)")
    ax.legend(fontsize=7)
    _save(fig, pdir / "cache_switch_cost_sensitivity")
    made["cache_switch_cost_sensitivity"] = str(pdir / "cache_switch_cost_sensitivity")

    # + rolling window
    rw = report.get("rolling_window", {}).get("windows", [])
    if rw:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["window"] for r in rw], [r["routed_quality"] for r in rw], marker="o", label="routed (learned)")
        ax.plot([r["window"] for r in rw], [r["q_always_cloud"] for r in rw], ls="--", label="always-cloud")
        ax.plot([r["window"] for r in rw], [r["q_oracle"] for r in rw], ls=":", label="oracle")
        ax.set_xlabel("window (chronological)"); ax.set_ylabel("quality")
        ax.set_title("Rolling-window online sim: routed vs baselines"); ax.legend(fontsize=8)
        _save(fig, pdir / "rolling_window_quality")
        _csv(rw, pdir / "rolling_window.csv")
        made["rolling_window_quality"] = str(pdir / "rolling_window_quality")

    # + epsilon sensitivity: epsilon is an operator-tunable knob, not a fixed
    # constant -- show the local-share/quality tradeoff it buys across a grid.
    eps_sens = report.get("epsilon_sensitivity", {})
    if eps_sens:
        fig, ax = plt.subplots(figsize=(6, 4))
        for pname, rows in sorted(eps_sens.items()):
            xs = [r["epsilon"] for r in rows]
            ax.plot(xs, [r["local_share"] for r in rows], marker="o", label=f"{pname} local share")
        ax.set_xlabel("epsilon (max quality drop below cloud)")
        ax.set_ylabel("achieved TEST local share")
        ax.set_title("Epsilon sensitivity: how much local share each epsilon buys")
        ax.legend(fontsize=8)
        _save(fig, pdir / "epsilon_sensitivity")
        flat = [
            {"policy": pname, **row} for pname, rows in eps_sens.items() for row in rows
        ]
        _csv(flat, pdir / "epsilon_sensitivity.csv")
        made["epsilon_sensitivity"] = str(pdir / "epsilon_sensitivity")

    # + joint quality/capacity controller: local share admitted vs offered load
    jc = report.get("joint_capacity", {})
    jc_logreg = jc.get("logreg", {}) if isinstance(jc, dict) else {}
    sweep = jc_logreg.get("sweep", [])
    if sweep:
        eps = report.get("epsilon", 0.02)
        xs = [r["offered_load_req_s"] for r in sweep]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, [r["quality_only"]["local_request_share"] for r in sweep],
                marker="s", ls="--", label="quality gate only (local share)")
        ax.plot(xs, [r["joint"]["local_request_share"] for r in sweep],
                marker="o", label="joint quality+capacity (local share)")
        ax.plot(xs, [r["oracle_cap"]["local_request_share"] for r in sweep],
                marker="^", ls=":", label="oracle within capacity")
        ax.set_xlabel("offered load (req/s)")
        ax.set_ylabel("local request share")
        ax.set_ylim(0, 1)
        ax2 = ax.twinx()
        ax2.plot(xs, [r["joint"]["quality"] for r in sweep],
                 color="tab:red", marker="o", label="joint routed quality")
        q_cloud = report.get("q_cloud_test")
        if q_cloud is not None:
            ax2.axhline(q_cloud - eps, color="tab:red", ls=":", lw=1,
                        label=f"cloud quality - eps ({eps})")
        ax2.set_ylabel("quality", color="tab:red")
        ax.set_title("Joint controller: local admission vs offered load")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
        _save(fig, pdir / "joint_capacity_admission")
        flat = [
            {
                "offered_load_req_s": r["offered_load_req_s"],
                "joint_local_share": r["joint"]["local_request_share"],
                "joint_local_service_share": r["joint"]["local_service_share"],
                "joint_quality": r["joint"]["quality"],
                "joint_within_epsilon": r["joint"]["within_epsilon"],
                "joint_n_shed_by_capacity": r["joint"]["n_shed_by_capacity"],
                "quality_only_local_share": r["quality_only"]["local_request_share"],
                "quality_only_peak_util": r["quality_only"]["peak_window_util"],
                "quality_only_capacity_violated": r["quality_only"]["capacity_violated"],
                "capacity_only_local_share": r["capacity_only"]["local_request_share"],
                "oracle_cap_local_share": r["oracle_cap"]["local_request_share"],
                "oracle_cap_quality": r["oracle_cap"]["quality"],
            }
            for r in sweep
        ]
        _csv(flat, pdir / "joint_capacity_admission.csv")
        made["joint_capacity_admission"] = str(pdir / "joint_capacity_admission")

    return made
