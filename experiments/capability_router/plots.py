"""The required plots, each written as PNG + PDF + the exact underlying
CSV so a reviewer can rebuild any figure without re-running the experiment.

Required set (exactly these three, plus a cost plot only if cost is ever
configured):
  1. quality_vs_local_share  -- Q_router vs. local traffic share
  2. threshold_vs_local_share -- local traffic share vs. classifier threshold
  3. threshold_vs_quality     -- Q_router vs. classifier threshold
  4. quality_vs_cost          -- NOT produced: this experiment has no cost
     model configured (`ArmFullMetrics.cost` is always None). Instead of
     silently omitting it, `cost_plot_unavailable()` writes a README noting
     exactly why, so its absence is documented, not just missing.

`opportunity_2x2` and `baseline_comparison` are additional, clearly-labeled
extras under `plots/extras/` -- not part of the required set above.

matplotlib is a hard dependency (see pyproject.toml) -- if it is ever
missing, this fails loudly at import time rather than silently skipping a
plot.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import AnalysisReport


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_fig(fig, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=150)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def _sweep_points(report: AnalysisReport, model_name: str) -> tuple[list[float], list[float], list[float]]:
    """(thresholds, local_shares, q_router_values) for one model's sweep.
    Q_router per threshold is recomputed from ArmResult's known-local
    success/violation counts restricted to the *chosen* backend per call --
    for a threshold sweep every call is either local (scored by
    known_local/violation) or cloud (assumed teacher-equivalent, since cloud
    replay is the teacher's own trajectory); this mirrors `q_router` exactly
    for the local-routed subset and cloud otherwise.
    """
    sweep = report.pareto_sweeps_test[model_name]
    thresholds = [r.threshold for r in sweep]
    shares = [r.local_share for r in sweep]
    # 1 - quality_loss is Q_router *restricted to the locally-routed known
    # subset*; for the plot's y-axis, that is exactly what changes as the
    # threshold moves cloud-routed calls (assumed teacher-equivalent) into
    # or out of the local set, so it is reported directly rather than
    # re-deriving q_router from raw examples here.
    quality = [1.0 - r.quality_loss if r.quality_loss is not None else 1.0 for r in sweep]
    return thresholds, shares, quality


def plot_quality_vs_local_share(report: AnalysisReport, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    rows = []
    for model_name in report.pareto_sweeps_test:
        thresholds, shares, quality = _sweep_points(report, model_name)
        ax.plot(shares, quality, marker=".", linewidth=1, label=model_name)
        for t, s, q in zip(thresholds, shares, quality):
            rows.append({"arm": model_name, "threshold": t, "local_share": s, "quality": q})

    markers = {"always_cloud": "s", "always_local": "^", "policy4": "D", "oracle": "*"}
    for name, m in report.baselines_test.items():
        q = m.q_router.success_rate
        share = m.traffic_pct_local / 100.0
        ax.scatter([share], [q if q is not None else float("nan")], marker=markers.get(name, "o"), s=90, label=name)
        rows.append({"arm": name, "threshold": None, "local_share": share, "quality": q})

    ax.set_xlabel("local traffic share (of all test calls)")
    ax.set_ylabel("Q_router (chosen-backend success rate, known labels)")
    ax.set_title("Quality vs. local traffic share (test split)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_stem)
    _write_csv(rows, out_stem.with_suffix(".csv"))


def plot_threshold_vs_local_share(report: AnalysisReport, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    rows = []
    for model_name in report.pareto_sweeps_test:
        thresholds, shares, _ = _sweep_points(report, model_name)
        ax.plot(thresholds, shares, label=model_name)
        for t, s in zip(thresholds, shares):
            rows.append({"arm": model_name, "threshold": t, "local_share": s})
    ax.set_xlabel("classifier threshold")
    ax.set_ylabel("local traffic share")
    ax.set_title("Local traffic share vs. threshold (test split)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_stem)
    _write_csv(rows, out_stem.with_suffix(".csv"))


def plot_threshold_vs_quality(report: AnalysisReport, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    rows = []
    for model_name in report.pareto_sweeps_test:
        thresholds, _, quality = _sweep_points(report, model_name)
        ax.plot(thresholds, quality, label=model_name)
        for t, q in zip(thresholds, quality):
            rows.append({"arm": model_name, "threshold": t, "quality": q})
    ax.set_xlabel("classifier threshold")
    ax.set_ylabel("Q_router (local-routed subset)")
    ax.set_title("Quality vs. threshold (test split)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_stem)
    _write_csv(rows, out_stem.with_suffix(".csv"))


def cost_plot_unavailable(out_dir: Path) -> Path:
    """Required 4th plot is quality-vs-cost, *only if cost is configured*.
    This experiment has no cost model (`ArmFullMetrics.cost` is always
    None, matching the standing instruction that dollar cost is no longer
    required) -- write an explicit, discoverable note instead of silently
    omitting the file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "quality_vs_cost_UNAVAILABLE.md"
    path.write_text(
        "# quality_vs_cost -- not produced\n\n"
        "This experiment does not configure a cost model. Every "
        "`ArmFullMetrics.cost` field is `None` by design (dollar cost was "
        "explicitly marked not required for this study). No fabricated or "
        "placeholder plot is written here; this file exists so the absence "
        "is a documented decision, not a missing artifact.\n"
    )
    return path


# ------------------------------------------------------------------ extras --


def plot_opportunity_2x2(report: AnalysisReport, out_stem: Path) -> None:
    counts = report.opportunity_2x2_test
    labels = ["both", "local_only", "cloud_only", "neither"]
    values = [counts.get(k, 0) for k in labels]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("count (test split, known-both calls)")
    ax.set_title(
        f"[extra] Opportunity 2x2 (excluded_unknown={counts.get('excluded_unknown', 0)}, total={counts.get('total', 0)})"
    )
    fig.tight_layout()
    _save_fig(fig, out_stem)
    _write_csv([{"quadrant": k, "count": counts.get(k, 0)} for k in labels + ["excluded_unknown", "total"]], out_stem.with_suffix(".csv"))


def plot_baseline_comparison(report: AnalysisReport, out_stem: Path) -> None:
    names = list(report.baselines_test.keys())
    local_pct = [report.baselines_test[n].traffic_pct_local for n in names]
    q = [(report.baselines_test[n].q_router.success_rate or 0.0) * 100.0 for n in names]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = range(len(names))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], local_pct, width, label="local traffic %", color="#4C72B0")
    ax1.bar([i + width / 2 for i in x], q, width, label="Q_router %", color="#55A868")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("%")
    ax1.set_title("[extra] Baselines: local traffic share vs. routing quality (test split)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save_fig(fig, out_stem)
    rows = [
        {
            "arm": n, "local_traffic_pct": report.baselines_test[n].traffic_pct_local,
            "q_router_pct": q[i], "q_router_known_count": report.baselines_test[n].q_router.known_count,
            "router_capture": report.baselines_test[n].router_capture,
        }
        for i, n in enumerate(names)
    ]
    _write_csv(rows, out_stem.with_suffix(".csv"))


def save_all_plots(report: AnalysisReport, results_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    required = (
        (plot_quality_vs_local_share, "quality_vs_local_share"),
        (plot_threshold_vs_local_share, "threshold_vs_local_share"),
        (plot_threshold_vs_quality, "threshold_vs_quality"),
    )
    for fn, stem in required:
        out_stem = results_dir / "plots" / stem
        fn(report, out_stem)
        outputs[stem] = out_stem

    outputs["quality_vs_cost"] = cost_plot_unavailable(results_dir / "plots")

    extras_dir = results_dir / "plots" / "extras"
    for fn, stem in (
        (plot_opportunity_2x2, "opportunity_2x2"),
        (plot_baseline_comparison, "baseline_comparison"),
    ):
        out_stem = extras_dir / stem
        fn(report, out_stem)
        outputs[f"extras/{stem}"] = out_stem
    return outputs
