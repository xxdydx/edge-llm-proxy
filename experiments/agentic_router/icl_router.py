"""Optional shadow in-context routing probe; prompt construction only.

This module makes no model calls.  It builds compact, request-time-only
examples from already labeled development calls.  An eventual DeepSeek-V4-
Flash probe may score at most 20 development calls and compare latency/tokens
against the frozen classifier.  No reserved final-holdout call or exemplar may
enter that pilot.  The numeric nearest-neighbor baseline lives in
``model_comparison.py``; this is a distinct prompt-based hypothesis.
"""

from __future__ import annotations

import json
import math
from typing import Any

from . import analysis
from .model_comparison import LabeledEvent, RESERVED_PILOT_HOLDOUT_TASKS

MAX_EXEMPLARS = 4


def _distance(a: LabeledEvent, b: LabeledEvent, scales: dict[str, float]) -> float:
    return math.sqrt(sum(((a.features[name] - b.features[name]) / scales[name]) ** 2
                         for name in analysis.FEATURE_NAMES))


def choose_exemplars(target: LabeledEvent, training: list[LabeledEvent],
                     max_exemplars: int = MAX_EXEMPLARS) -> list[LabeledEvent]:
    """Nearest different-task exemplars, with both classes represented.

    No current label is used.  UNKNOWN training examples are not treated as
    negative or positive evidence.  The target task and both reserved tasks
    are excluded even if the caller accidentally includes them in training.
    """
    if target.task_group in RESERVED_PILOT_HOLDOUT_TASKS:
        raise ValueError("reserved final holdout cannot enter the ICL pilot")
    pool = [e for e in training if e.label in ("SAFE", "HARM")
            and e.task_group != target.task_group
            and e.task_group not in RESERVED_PILOT_HOLDOUT_TASKS
            and e.call_id != target.call_id]
    if not pool:
        return []
    scales = {}
    for name in analysis.FEATURE_NAMES:
        values = [e.features[name] for e in pool]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        scales[name] = math.sqrt(variance) or 1.0
    ordered = sorted(pool, key=lambda e: (_distance(target, e, scales), e.call_id))
    selected: list[LabeledEvent] = []
    for label in ("HARM", "SAFE"):
        match = next((e for e in ordered if e.label == label), None)
        if match is not None and len(selected) < max_exemplars:
            selected.append(match)
    for e in ordered:
        if len(selected) >= max_exemplars:
            break
        if e not in selected:
            selected.append(e)
    return selected


def _summary(event: LabeledEvent) -> dict[str, Any]:
    # These feature names are the audited pre-dispatch contract.  No raw
    # prompt, response, judge reason, backend identity, or task outcome enters.
    return {name: event.features[name] for name in analysis.FEATURE_NAMES}


def build_prompt(target: LabeledEvent, exemplars: list[LabeledEvent]) -> str:
    if target.task_group in RESERVED_PILOT_HOLDOUT_TASKS:
        raise ValueError("reserved final holdout cannot enter the ICL pilot")
    if any(e.label not in ("SAFE", "HARM") or e.task_group == target.task_group
           or e.task_group in RESERVED_PILOT_HOLDOUT_TASKS for e in exemplars):
        raise ValueError("exemplars must be labeled training calls from other development tasks")
    examples = [{"state": _summary(e), "outcome": e.label} for e in exemplars]
    return (
        "You are a conservative router risk estimator. Each state is known BEFORE a model "
        "is chosen. HARM means the local model was judged materially worse than the cloud "
        "model for the next agent action; SAFE means local was at least as good. The labels "
        "are noisy proxy judgments. Infer from the examples only; if support is weak or "
        "contradictory, abstain. Return one JSON object with keys verdict (HARM, SAFE, or "
        "UNKNOWN), harm_probability (number 0..1 or null for UNKNOWN), and reason "
        "(at most 30 words). Do not choose a backend or invent test outcomes.\n"
        f"Examples: {json.dumps(examples, sort_keys=True)}\n"
        f"Target: {json.dumps(_summary(target), sort_keys=True)}"
    )


def parse_response(text: str) -> tuple[str, float | None]:
    """Strict parser: malformed, overconfident, or non-JSON output abstains."""
    try:
        raw = json.loads(text)
    except (TypeError, ValueError):
        return "UNKNOWN", None
    if not isinstance(raw, dict) or raw.get("verdict") not in ("HARM", "SAFE", "UNKNOWN"):
        return "UNKNOWN", None
    verdict = raw["verdict"]
    prob = raw.get("harm_probability")
    if verdict == "UNKNOWN":
        return "UNKNOWN", None
    if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not math.isfinite(prob) or not 0 <= prob <= 1:
        return "UNKNOWN", None
    return verdict, float(prob)
