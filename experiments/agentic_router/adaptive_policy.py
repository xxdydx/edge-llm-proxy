"""Small prequential diagnostic for paired-call routing policies.

Every evaluated task group is predicted using only *earlier* groups.  The same
groups and thresholds are used for frozen, sliding, and recency-weighted
models.  This measures judge-label proxy harm, not end-to-end task success or
a two-percentage-point quality guarantee.  UNKNOWN calls stay in the served
share denominator and are reported, never relabeled as SAFE.

Callers must supply features reconstructed from the original full trajectory
at dispatch time (not stale replay-row ``derived`` dictionaries).  The module
does no network calls and never changes live placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Literal, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Label = Literal["SAFE", "HARM", "UNKNOWN"]


@dataclass(frozen=True)
class Observation:
    call_id: str
    trajectory_id: str
    task_group: str
    timestamp: float
    features: tuple[float, ...]
    label: Label
    static_eligible: bool = True
    # Optional *comparable* selected-load estimate, supplied by the caller.
    # None is unknown, not zero. Provider input/output usage should be
    # reported separately by the data pipeline rather than normalized here.
    comparable_local_tokens: float | None = None
    comparable_cloud_tokens: float | None = None


def _fit(rows: Sequence[Observation], weights: Sequence[float] | None = None):
    labeled = [(r, i) for i, r in enumerate(rows) if r.label != "UNKNOWN"]
    if not labeled or len({r.label for r, _ in labeled}) < 2:
        return None
    x = np.asarray([r.features for r, _ in labeled], dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("all predecision features must be finite and have equal width")
    y = np.asarray([int(r.label == "HARM") for r, _ in labeled])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000),
    )
    fit_kwargs = {}
    if weights is not None:
        fit_kwargs["logisticregression__sample_weight"] = np.asarray(
            [weights[i] for _, i in labeled], dtype=float
        )
    model.fit(x, y, **fit_kwargs)
    return model


def _score(model, rows: Sequence[Observation]) -> list[float | None]:
    if model is None:
        return [None] * len(rows)
    x = np.asarray([r.features for r in rows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("all predecision features must be finite")
    return [float(v) for v in model.predict_proba(x)[:, 1]]


def _summary(records: list[tuple[Observation, bool]]) -> dict:
    n = len(records)
    known = [(r, local) for r, local in records if r.label != "UNKNOWN"]
    local_known = [r for r, local in known if local]
    harm_all = sum(r.label == "HARM" for r, _ in known)
    harm_local = sum(r.label == "HARM" for r in local_known)
    local_tokens = [r.comparable_local_tokens for r, local in records if local]
    cloud_tokens = [r.comparable_cloud_tokens for r, local in records if not local]
    tokens_complete = all(v is not None for v in local_tokens + cloud_tokens)
    return {
        "n_calls": n,
        "n_known": len(known),
        "n_unknown": n - len(known),
        "unknown_fraction": (n - len(known)) / n if n else None,
        "local_share_all_calls": sum(local for _, local in records) / n if n else None,
        "harm_fraction_all_known": harm_all / len(known) if known else None,
        "harm_fraction_local_known": harm_local / len(local_known) if local_known else None,
        "n_local_known": len(local_known),
        "n_harm_local": harm_local,
        "comparable_selected_token_share_local": (
            sum(local_tokens) / (sum(local_tokens) + sum(cloud_tokens))
            if tokens_complete and local_tokens + cloud_tokens
            and sum(local_tokens) + sum(cloud_tokens) > 0 else None
        ),
        "token_coverage_complete": tokens_complete,
    }


def compare(
    observations: Sequence[Observation], *, initial_groups: int = 2,
    sliding_groups: int = 3, return_local_threshold: float = 0.10,
    cloud_threshold: float = 0.25,
) -> dict:
    """Evaluate fixed policies on identical future groups, no current labels.

    The policy thresholds are illustrative fixed inputs, *not* selected from
    these held-out labels.  Recency weighting gives each preceding group half
    the weight of its immediate successor.  Same-trajectory placement state
    is reset on every policy run and never crosses trajectory IDs.
    """
    if initial_groups < 1 or sliding_groups < 1:
        raise ValueError("group counts must be positive")
    if not 0 <= return_local_threshold < cloud_threshold <= 1:
        raise ValueError("invalid hysteresis thresholds")
    ordered = sorted(observations, key=lambda r: (r.timestamp, r.call_id))
    if len({r.call_id for r in ordered}) != len(ordered):
        raise ValueError("call_id must be unique")
    widths = {len(r.features) for r in ordered}
    if len(widths) > 1:
        raise ValueError("feature width mismatch")
    by_group: dict[str, list[Observation]] = {}
    for r in ordered:
        by_group.setdefault(r.task_group, []).append(r)
    groups = sorted(by_group.items(), key=lambda item: (item[1][0].timestamp, item[0]))
    for (name_a, block_a), (name_b, block_b) in zip(groups, groups[1:]):
        if block_a[-1].timestamp >= block_b[0].timestamp:
            raise ValueError(
                f"overlapping task groups {name_a} and {name_b}; "
                "cannot use later labels before earlier-group predictions"
            )
    if len(groups) <= initial_groups:
        return {"status": "insufficient_future_groups", "n_groups": len(groups)}

    names = ("always_local", "always_cloud", "static", "frozen", "sliding", "recency")
    outputs: dict[str, list[tuple[Observation, bool]]] = {name: [] for name in names}
    per_group: dict[str, dict[str, dict]] = {}
    frozen_rows = [r for _, block in groups[:initial_groups] for r in block]
    frozen = _fit(frozen_rows)
    for at in range(initial_groups, len(groups)):
        name, current = groups[at]
        prior = groups[:at]
        sliding = _fit([r for _, block in prior[-sliding_groups:] for r in block])
        recency_rows = [r for _, block in prior for r in block]
        weights = [0.5 ** (at - 1 - index) for index, (_, block) in enumerate(prior) for _ in block]
        recency = _fit(recency_rows, weights)
        predictions = {
            "frozen": _score(frozen, current),
            "sliding": _score(sliding, current),
            "recency": _score(recency, current),
        }
        group_records: dict[str, list[tuple[Observation, bool]]] = {n: [] for n in names}
        for policy in names:
            previous: dict[str, bool] = {}
            for index, row in enumerate(current):
                if policy == "always_local":
                    local = True
                elif policy == "always_cloud":
                    local = False
                elif policy == "static":
                    local = row.static_eligible
                else:
                    risk = predictions[policy][index]
                    prior_local = previous.get(row.trajectory_id, False)
                    local = bool(
                        row.static_eligible and risk is not None and
                        (risk <= return_local_threshold or
                         (prior_local and risk < cloud_threshold))
                    )
                previous[row.trajectory_id] = local
                group_records[policy].append((row, local))
                outputs[policy].append((row, local))
        per_group[name] = {policy: _summary(records) for policy, records in group_records.items()}
    return {
        "status": "diagnostic_proxy_only",
        "n_groups": len(groups),
        "n_future_groups": len(groups) - initial_groups,
        "future_groups": [name for name, _ in groups[initial_groups:]],
        "thresholds": {"return_local": return_local_threshold, "cloud": cloud_threshold},
        "summary": {name: _summary(records) for name, records in outputs.items()},
        "per_group": per_group,
        "caveat": "Paired judge HARM is a call-level proxy, not task success; tiny group counts and correlated calls do not establish an epsilon guarantee. UNKNOWN calls remain in local-share and coverage denominators. This group-disjoint test is stricter than chronological per-call updating, which can be valid if each label is actually available before the later decision.",
    }


def compare_snapshot(*, initial_groups: int = 2, sliding_groups: int = 3) -> dict:
    """Load the frozen paired snapshot with fresh causal feature derivation."""
    from . import analysis, model_comparison

    events, meta = model_comparison.load_events()
    rows = [
        Observation(
            call_id=e.call_id,
            trajectory_id=e.trajectory_id,
            task_group=e.task_group,
            timestamp=e.timestamp_unix_s,
            features=tuple(float(e.features[name]) for name in analysis.FEATURE_NAMES),
            label=e.label,
        )
        for e in events
    ]
    try:
        output = compare(rows, initial_groups=initial_groups, sliding_groups=sliding_groups)
    except ValueError as exc:
        if not str(exc).startswith("overlapping task groups"):
            raise
        output = {
            "status": "not_identifiable_from_snapshot",
            "reason": str(exc),
            "n_calls": len(rows),
            "n_unknown": sum(row.label == "UNKNOWN" for row in rows),
            "n_task_groups": len({row.task_group for row in rows}),
            "caveat": "All current task groups overlap in wall time, so a chronological group-disjoint frozen/sliding/recency generalization comparison cannot be formed. Chronological per-call updates are separately possible when feedback arrival times are real and strictly prior to each decision; this snapshot has no such arrival telemetry. Collect new independent groups in successive batches before comparing adaptation generalization.",
        }
    output["source_meta"] = meta
    return output


def compare_bandits(
    observations: Sequence[Observation], *, delay_calls: int = 1,
    local_bonus: float = 0.10, alpha: float = 0.1,
    thompson_seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict:
    """Chosen-arm-only feedback simulation on chronological paired labels.

    The objective is an explicit *proxy utility*: cloud=1 on known labels;
    local=1+local_bonus on SAFE, 0 on HARM. This encodes a preference for
    useful local service without claiming cloud's SAFE answer is wrong. UNKNOWN
    is served but supplies no reward/update. Delayed feedback is a simulation
    assumption: real paired labels need shadow replay/judging and are not
    automatically observed in production.
    """
    if delay_calls < 1 or not 0 <= local_bonus <= 1 or alpha < 0:
        raise ValueError("invalid bandit parameters")
    ordered = sorted(observations, key=lambda r: (r.timestamp, r.call_id))
    if not ordered:
        return {"status": "empty"}
    if len({r.call_id for r in ordered}) != len(ordered):
        raise ValueError("call_id must be unique")
    width = len(ordered[0].features)
    if any(len(r.features) != width for r in ordered):
        raise ValueError("feature width mismatch")

    def vector(row: Observation) -> np.ndarray:
        raw = np.asarray(row.features, dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("nonfinite feature")
        return np.concatenate(([1.0], raw / np.sqrt(1.0 + raw * raw)))

    class Arm:
        def __init__(self):
            self.a = np.eye(width + 1)
            self.b = np.zeros(width + 1)

        def mean_and_variance(self, x):
            mean = np.linalg.solve(self.a, self.b)
            variance = float(x @ np.linalg.solve(self.a, x))
            return float(mean @ x), max(variance, 0.0)

        def update(self, x, reward):
            self.a += np.outer(x, x)
            self.b += reward * x

    def reward(row: Observation, local: bool) -> float | None:
        if row.label == "UNKNOWN":
            return None
        if not local:
            return 1.0
        return 1.0 + local_bonus if row.label == "SAFE" else 0.0

    def run(kind: str, seed: int | None = None) -> dict:
        arms = {True: Arm(), False: Arm()}
        rng = np.random.default_rng(seed)
        # (release decision index, selected arm, x, observed reward)
        pending: deque[tuple[int, bool, np.ndarray, float]] = deque()
        records = []
        observed_rewards = []
        released_before_decision = []
        for i, row in enumerate(ordered):
            released = 0
            while pending and pending[0][0] <= i:
                _, chosen, prior_x, prior_reward = pending.popleft()
                arms[chosen].update(prior_x, prior_reward)
                released += 1
            released_before_decision.append(released)
            x = vector(row)
            if kind in ("always_local", "always_cloud", "static"):
                local = (kind == "always_local" or (kind == "static" and row.static_eligible))
            else:
                scores = {}
                for chosen in (False, True):
                    mean, variance = arms[chosen].mean_and_variance(x)
                    scores[chosen] = (
                        mean + alpha * np.sqrt(variance)
                        if kind == "linucb"
                        else float(rng.normal(mean, alpha * np.sqrt(variance)))
                    )
                local = bool(row.static_eligible and scores[True] > scores[False])
            got = reward(row, local)
            if got is not None:
                observed_rewards.append(got)
                if kind in ("linucb", "thompson"):
                    pending.append((i + delay_calls, local, x, got))
            records.append((row, local))
        return {
            **_summary(records),
            "mean_proxy_utility_known": sum(observed_rewards) / len(observed_rewards) if observed_rewards else None,
            "n_updates_released_before_decision": sum(released_before_decision),
            "n_updates_pending_after_last_decision": len(pending),
        }

    policies = {name: run(name) for name in ("always_local", "always_cloud", "static", "linucb")}
    policies["thompson"] = [run("thompson", seed) for seed in thompson_seeds]
    return {
        "status": "offline_selected_arm_proxy_simulation",
        "delay_calls": delay_calls,
        "local_bonus": local_bonus,
        "alpha": alpha,
        "n_thompson_seeds": len(thompson_seeds),
        "policies": policies,
        "caveat": "Judge SAFE/HARM is call-level paired proxy feedback, not task success. Only the selected arm is updated, after the specified decision delay; in live service this feedback requires separate sampling and judging. Seed variation is not task-group uncertainty.",
    }


def compare_bandits_snapshot(*, delay_calls: int = 1, local_bonus: float = 0.10) -> dict:
    """Mechanism-only simulation on the frozen 205-call chronology.

    The snapshot has no actual judge-feedback arrival timestamps; therefore
    ``delay_calls`` is a synthetic assumption, not an empirical online result.
    Same-task calls may inform later same-task calls, unlike the stricter
    group-disjoint generalization test above.
    """
    from . import analysis, model_comparison

    events, meta = model_comparison.load_events()
    if any(e.timestamp_unix_s is None for e in events):
        return {"status": "missing_event_timestamps", "source_meta": meta}
    rows = [Observation(
        call_id=e.call_id, trajectory_id=e.trajectory_id,
        task_group=e.task_group, timestamp=e.timestamp_unix_s,
        features=tuple(float(e.features[name]) for name in analysis.FEATURE_NAMES),
        label=e.label,
    ) for e in events]
    result = compare_bandits(rows, delay_calls=delay_calls, local_bonus=local_bonus)
    result["source_meta"] = meta
    result["feedback_timing_status"] = "synthetic_call_delay_no_observed_arrival_time"
    result["paired_feedback_budget"] = {
        "n_calls_with_paired_known_label": sum(r.label != "UNKNOWN" for r in rows),
        "n_calls_needing_paired_feedback_to_reproduce_simulation": len(rows),
        "deployment_note": "This generous offline simulation reads paired labels for every call. Live use must budget sampled shadow replay and judgment separately; unpaired calls cannot update the selected arm this way.",
    }
    return result
