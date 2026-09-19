"""Pre-dispatch, request-only features for the routing policy.

Everything here is knowable BEFORE either model is called -- no generation
text, no grades, no post-hoc signals. This is what a deployable router would
actually see.
"""

from __future__ import annotations

import re

from .prompting import build_prompt
from .schema import Problem

_WORD = re.compile(r"\w+")


def extract_features(p: Problem) -> dict[str, float]:
    prompt = build_prompt(p)
    doc = p.prompt
    # crude token estimate: MBPP prompts are short English + a code line
    est_tokens = len(prompt) / 4.0
    example_asserts = doc.count("assert")
    # arity of the target function, read from the first example assertion
    m = re.search(rf"{re.escape(p.entry_point)}\s*\(", doc)
    arity = 0.0
    if m:
        after = doc[m.end():]
        depth, arg_chars, args = 1, 0, 1
        for ch in after:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            elif ch == "," and depth == 1:
                args += 1
            arg_chars += 1
        arity = float(args) if arg_chars else 0.0
    words = _WORD.findall(doc.lower())
    return {
        "prompt_chars": float(len(prompt)),
        "prompt_est_tokens": est_tokens,
        "doc_chars": float(len(doc)),
        "doc_words": float(len(words)),
        "n_example_asserts": float(example_asserts),
        "entry_point_arity": arity,
        "entry_point_len": float(len(p.entry_point)),
        # lexical difficulty hints
        "mentions_string": float(any(w in words for w in ("string", "char", "substring", "vowel"))),
        "mentions_math": float(any(w in words for w in ("sum", "product", "prime", "factorial", "gcd", "lcm", "digit"))),
        "mentions_list": float(any(w in words for w in ("list", "array", "tuple", "element", "sublist"))),
        "mentions_sort": float(any(w in words for w in ("sort", "sorted", "order", "largest", "smallest"))),
        "mentions_regex": float(any(w in words for w in ("regex", "pattern", "match"))),
        "mentions_dp": float(any(w in words for w in ("minimum", "maximum", "longest", "count", "ways", "number"))),
    }


FEATURE_NAMES = (
    "prompt_chars", "prompt_est_tokens", "doc_chars", "doc_words",
    "n_example_asserts", "entry_point_arity", "entry_point_len",
    "mentions_string", "mentions_math", "mentions_list", "mentions_sort",
    "mentions_regex", "mentions_dp",
)
