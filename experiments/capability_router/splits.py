"""Deterministic grouped train/validation/test split by SWE-bench task
group. All of one group's calls land in exactly one partition -- never
split within a group -- so no split can leak a task's request shapes across
train and test. Persisted so a rerun (or a reviewer) can verify the exact
same partition was used, not just check the code once.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .schema import TeacherCall


@dataclass(frozen=True)
class SplitDefinition:
    seed: int
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


def make_split(
    task_groups: list[str], seed: int, train_frac: float, val_frac: float
) -> SplitDefinition:
    groups = sorted(set(task_groups))
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, round(n * train_frac)) if n >= 3 else max(1, n - 2)
    n_val = max(1, round(n * val_frac)) if n - n_train >= 2 else max(0, n - n_train - 1)
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = groups[:n_train]
    val = groups[n_train : n_train + n_val]
    test = groups[n_train + n_val :]
    return SplitDefinition(
        seed=seed,
        train_groups=tuple(train),
        val_groups=tuple(val),
        test_groups=tuple(test),
    )


def apply_split(examples: list, split: SplitDefinition, group_of) -> tuple[list, list, list]:
    """`group_of(example) -> str`; kept generic so it works on TeacherCall or
    LabeledExample rows alike."""
    train = [e for e in examples if group_of(e) in split.train_groups]
    val = [e for e in examples if group_of(e) in split.val_groups]
    test = [e for e in examples if group_of(e) in split.test_groups]
    return train, val, test


def save_split(split: SplitDefinition, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(split), indent=2) + "\n")


def load_split(path: Path) -> SplitDefinition:
    data = json.loads(path.read_text())
    return SplitDefinition(
        seed=data["seed"],
        train_groups=tuple(data["train_groups"]),
        val_groups=tuple(data["val_groups"]),
        test_groups=tuple(data["test_groups"]),
    )


def repo_of(task_group: str) -> str:
    """"swebench-openlibrary-c05ccf2c" -> "openlibrary". Used only for the
    optional repo-held-out variant, never for the primary grouped split."""
    stripped = task_group.removeprefix("swebench-")
    return stripped.rsplit("-", 1)[0] if "-" in stripped else stripped


def distinct_repos(task_groups: list[str]) -> set[str]:
    return {repo_of(g) for g in task_groups}
