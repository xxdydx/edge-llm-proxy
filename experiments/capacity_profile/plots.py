"""Service curves for the capacity profile (+ each plot's CSV).

  1. throughput vs offered load       - input / output / total tok/s and req/s vs concurrency
  2. latency vs offered load          - p50/p95/p99 e2e and p95 TTFT vs concurrency
  3. open-loop latency vs arrival rate - p50/p95 e2e vs target rps, SLO line
  4. KV + queue vs offered load       - mean gpu_cache_usage & max waiting vs concurrency
  5. cost vs offered load             - GPU-active-equivalent s/request vs concurrency
  6. queue wait vs offered load       - mean queue wait (histogram or named proxy)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path_noext: Path) -> None:
    path_noext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_noext}.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{path_noext}.pdf", bbox_inches="tight")
    plt.close(fig)


def _csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_all(stage_rows: list[dict], envelope: dict, out_dir: Path, slo: dict) -> dict[str, str]:
    pdir = out_dir / "plots"
    made: dict[str, str] = {}

    closed = sorted(
        (r for r in stage_rows if r["mode"] == "closed_loop"),
        key=lambda r: r["offered_load"],
    )
    openl = sorted(
        (r for r in stage_rows if r["mode"] == "open_loop"),
        key=lambda r: r["offered_load"],
    )

    def _agg(rows, x_key, y_key):
        by_x: dict[float, list[float]] = {}
        for r in rows:
            y = r.get(y_key)
            if y in (None, ""):
                continue
            by_x.setdefault(r[x_key], []).append(float(y))
        return [(x, sum(v) / len(v)) for x, v in sorted(by_x.items())]

    # 1. throughput vs concurrency (input / output / total tok/s, + req/s)
    if closed:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        for y_key, style, colour in (
            ("input_token_tps", "^-", "tab:cyan"),
            ("output_token_tps", "o-", "tab:blue"),
            ("total_token_tps", "D-", "tab:green"),
        ):
            pts = _agg(closed, "offered_load", y_key)
            if pts:
                ax1.plot([x for x, _ in pts], [y for _, y in pts], style, ms=4,
                         color=colour, label=y_key)
        rps = _agg(closed, "offered_load", "realized_rps")
        ax1.set_xlabel("offered concurrency")
        ax1.set_ylabel("realized tok/s")
        ax2 = ax1.twinx()
        ax2.plot([x for x, _ in rps], [y for _, y in rps], "s--", color="tab:red", label="req/s")
        ax2.set_ylabel("realized req/s", color="tab:red")
        knee = envelope.get("throughput_knee_concurrency")
        if knee:
            ax1.axvline(knee, color="grey", ls=":", label=f"knee c={knee}")
        ax1.set_title("Throughput vs offered load (closed loop)")
        ax1.legend(loc="upper left", fontsize=8)
        _save(fig, pdir / "throughput_vs_concurrency")
        _csv(closed, pdir / "throughput_vs_concurrency.csv")
        made["throughput_vs_concurrency"] = str(pdir / "throughput_vs_concurrency.png")

    # 2. latency vs concurrency
    if closed:
        fig, ax = plt.subplots(figsize=(6, 4))
        for y_key, style in (("p50_e2e_s", "o-"), ("p95_e2e_s", "s-"), ("p99_e2e_s", "^-"), ("p95_ttft_s", "d--")):
            pts = _agg(closed, "offered_load", y_key)
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], style, ms=4, label=y_key)
        ax.axhline(slo["p95_e2e_s"], color="red", ls=":", label=f"SLO p95 e2e {slo['p95_e2e_s']}s")
        ax.set_xlabel("offered concurrency")
        ax.set_ylabel("latency (s)")
        ax.set_title("Latency vs offered load (closed loop)")
        ax.legend(fontsize=8)
        _save(fig, pdir / "latency_vs_concurrency")
        made["latency_vs_concurrency"] = str(pdir / "latency_vs_concurrency.png")

    # 3. open-loop latency vs arrival rate
    if openl:
        fig, ax = plt.subplots(figsize=(6, 4))
        for y_key, style in (("p50_e2e_s", "o-"), ("p95_e2e_s", "s-")):
            pts = _agg(openl, "offered_load", y_key)
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], style, ms=4, label=y_key)
        ax.axhline(slo["p95_e2e_s"], color="red", ls=":", label=f"SLO p95 e2e {slo['p95_e2e_s']}s")
        ax.set_xlabel("target arrival rate (req/s)")
        ax.set_ylabel("end-to-end latency (s)")
        ax.set_title("Latency vs arrival rate (open loop, Poisson)")
        ax.legend(fontsize=8)
        _save(fig, pdir / "latency_vs_arrival_rate")
        _csv(openl, pdir / "latency_vs_arrival_rate.csv")
        made["latency_vs_arrival_rate"] = str(pdir / "latency_vs_arrival_rate.png")

    # 4. KV + queue vs concurrency
    if closed:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        kv = _agg(closed, "offered_load", "mean_gpu_cache_usage")
        wq = _agg(closed, "offered_load", "max_num_waiting")
        ax1.plot([x for x, _ in kv], [y for _, y in kv], "o-", color="tab:green", label="mean KV usage")
        ax1.set_xlabel("offered concurrency")
        ax1.set_ylabel("gpu_cache_usage_perc", color="tab:green")
        ax1.set_ylim(0, 1.05)
        ax2 = ax1.twinx()
        ax2.plot([x for x, _ in wq], [y for _, y in wq], "s--", color="tab:purple", label="max waiting")
        ax2.set_ylabel("max num_requests_waiting", color="tab:purple")
        ax1.set_title("KV pool + scheduler queue vs offered load")
        _save(fig, pdir / "kv_queue_vs_concurrency")
        made["kv_queue_vs_concurrency"] = str(pdir / "kv_queue_vs_concurrency.png")

    # 5. cost vs concurrency
    if closed:
        fig, ax = plt.subplots(figsize=(6, 4))
        pts = _agg(closed, "offered_load", "gpu_active_equiv_s_per_request")
        ax.plot([x for x, _ in pts], [y for _, y in pts], "o-", color="tab:brown")
        ax.set_xlabel("offered concurrency")
        ax.set_ylabel("GPU-active-equiv s / request")
        ax.set_title("Cost proxy vs offered load")
        _save(fig, pdir / "cost_vs_concurrency")
        made["cost_vs_concurrency"] = str(pdir / "cost_vs_concurrency.png")

    # 6b. SLO sensitivity: feasible frontier at stricter/default/looser cutoffs
    sens = envelope.get("slo_sensitivity") or []
    if sens:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        xs = [f"{v['p95_e2e_s']}s\n{v['label']}" for v in sens]
        conc = [v.get("max_concurrency_under_slo") or 0 for v in sens]
        rps = [v.get("max_offered_rps_under_slo") or 0 for v in sens]
        import numpy as np

        idx = np.arange(len(xs))
        ax1.bar(idx - 0.2, conc, width=0.4, color="tab:blue", label="max concurrency")
        ax1.set_ylabel("max concurrency under SLO", color="tab:blue")
        ax1.set_xticks(idx)
        ax1.set_xticklabels(xs)
        ax2 = ax1.twinx()
        ax2.bar(idx + 0.2, rps, width=0.4, color="tab:red", label="max offered req/s")
        ax2.set_ylabel("max offered req/s under SLO", color="tab:red")
        ax1.set_title("Capacity envelope vs p95 e2e SLO cutoff")
        _save(fig, pdir / "slo_sensitivity")
        _csv(sens, pdir / "slo_sensitivity.csv")
        made["slo_sensitivity"] = str(pdir / "slo_sensitivity.png")

    # 6. queue wait vs offered load (both modes; label the source)
    for rows, xlabel, tag in (
        (closed, "offered concurrency", "concurrency"),
        (openl, "target arrival rate (req/s)", "arrival_rate"),
    ):
        if not rows:
            continue
        pts = _agg(rows, "offered_load", "mean_queue_wait_s")
        if not pts:
            continue
        src = rows[-1].get("queue_wait_source", "unknown")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([x for x, _ in pts], [y for _, y in pts], "o-", color="tab:orange")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"mean queue wait (s) [{src}]")
        ax.set_title(f"Queue wait vs offered load ({tag})")
        _save(fig, pdir / f"queue_wait_vs_{tag}")
        made[f"queue_wait_vs_{tag}"] = str(pdir / f"queue_wait_vs_{tag}.png")

    return made
