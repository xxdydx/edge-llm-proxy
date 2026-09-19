"""Deterministic, non-executing Dataset C controlled profile plan."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class Condition(str, Enum):
    COLD_LOW = "cold_low"
    COLD_HIGH = "cold_high"
    WARM_LOW = "warm_low"
    WARM_HIGH = "warm_high"


@dataclass(frozen=True)
class ProfileInvocation:
    ordinal: int
    request_ref: str
    condition: Condition
    output_cap: int
    priming_count: int
    declared_background_load: str


@dataclass(frozen=True)
class ControlledProfilePlan:
    protocol_version: str
    seed: int
    output_cap: int
    requests: tuple[str, ...]
    invocations: tuple[ProfileInvocation, ...]

    @property
    def foreground_count(self) -> int:
        return len(self.invocations)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(asdict(row), sort_keys=True, default=lambda x: x.value) for row in self.invocations)


def build_profile_plan(request_refs: Iterable[str], *, seed: int, output_cap: int = 256) -> ControlledProfilePlan:
    refs = tuple(request_refs)
    if not 1 <= len(refs) <= 10 or len(set(refs)) != len(refs) or any(not ref for ref in refs):
        raise ValueError("provide 1 to 10 distinct saved development request refs")
    if output_cap <= 0:
        raise ValueError("output_cap must be positive")
    conditions = [Condition.COLD_LOW, Condition.COLD_HIGH, Condition.WARM_LOW, Condition.WARM_HIGH]
    rows = [(ref, condition) for ref in refs for condition in conditions]
    random.Random(seed).shuffle(rows)
    invocations = tuple(ProfileInvocation(i, ref, condition, output_cap, int(condition.value.startswith("warm")), "high" if condition.value.endswith("high") else "low") for i, (ref, condition) in enumerate(rows))
    return ControlledProfilePlan("w5-controlled-profile-v1", seed, output_cap, refs, invocations)
