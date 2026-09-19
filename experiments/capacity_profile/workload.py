"""Load a fixed, seed-shuffled subset of the Phase 1 MBPP+ prompts.

The prompt bytes are built the same way Phase 1's local arm built them
(``experiments.phase1_router.prompting.build_prompt``) so the token-length
distribution the box sees here matches the routing workload this capacity
envelope bounds. This is a load generator: correctness of the generated code is
never checked.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

# byte-identical to experiments.phase1_router.prompting._INSTRUCTION
_INSTRUCTION = (
    "You are given a Python programming problem as a docstring, including one "
    "or more example assertions. Write a single, self-contained Python "
    "solution: the required function (named exactly as in the examples) plus "
    "any helper code it needs. Return ONLY a Python code block -- no prose, no "
    "explanation, no test code."
)


@dataclass(frozen=True)
class WorkloadItem:
    task_id: str
    entry_point: str
    prompt: str  # the full user-message content sent to the model
    prompt_chars: int


def _build_prompt(raw_prompt: str) -> str:
    return f"{_INSTRUCTION}\n\n{raw_prompt.strip()}\n"


def load_workload(dataset_gz: Path, size: int, seed: int) -> tuple[list[WorkloadItem], dict]:
    """Return ``size`` workload items plus a provenance dict.

    Selection: read every task in file order, take a ``random.Random(seed)``
    shuffle, keep the first ``size``. Deterministic for a given (file, size,
    seed).
    """
    raw_bytes = dataset_gz.read_bytes()
    rows: list[dict] = []
    with gzip.open(dataset_gz, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if size > len(rows):
        raise ValueError(f"workload size {size} exceeds dataset ({len(rows)} tasks)")

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    picked = [rows[i] for i in order[:size]]

    items: list[WorkloadItem] = []
    for r in picked:
        prompt = _build_prompt(r["prompt"])
        items.append(
            WorkloadItem(
                task_id=r["task_id"],
                entry_point=r["entry_point"],
                prompt=prompt,
                prompt_chars=len(prompt),
            )
        )

    lengths = sorted(it.prompt_chars for it in items)
    provenance = {
        "dataset_gz": str(dataset_gz),
        "dataset_gz_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "dataset_total_tasks": len(rows),
        "workload_size": size,
        "workload_seed": seed,
        "selected_task_ids": [it.task_id for it in items],
        "prompt_chars_min": lengths[0],
        "prompt_chars_median": lengths[len(lengths) // 2],
        "prompt_chars_max": lengths[-1],
    }
    return items, provenance
