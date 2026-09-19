"""Build the plain completion prompt for one MBPP+ problem and extract the
Python solution from a model response. No tools, no system prompt tricks --
identical bytes go to both backends.
"""

from __future__ import annotations

import re

from .schema import Problem

_INSTRUCTION = (
    "You are given a Python programming problem as a docstring, including one "
    "or more example assertions. Write a single, self-contained Python "
    "solution: the required function (named exactly as in the examples) plus "
    "any helper code it needs. Return ONLY a Python code block -- no prose, no "
    "explanation, no test code."
)


def build_prompt(p: Problem) -> str:
    return f"{_INSTRUCTION}\n\n{p.prompt.strip()}\n"


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:\n```|\Z)", re.DOTALL | re.IGNORECASE)


def extract_code(text: str | None, entry_point: str) -> str | None:
    """Pull the solution code out of a model response.

    Preference order: the LAST fenced block that defines ``entry_point``; else
    the last fenced block; else, if the raw text itself defines
    ``entry_point``, the raw text. Returns None when nothing plausibly
    contains the function.
    """
    if not text:
        return None
    blocks = [m.group(1).strip() for m in _FENCE.finditer(text)]
    defining = [b for b in blocks if re.search(rf"^\s*def\s+{re.escape(entry_point)}\b", b, re.MULTILINE)]
    if defining:
        return defining[-1]
    if blocks:
        return blocks[-1]
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\b", text, re.MULTILINE):
        return text.strip()
    return None
