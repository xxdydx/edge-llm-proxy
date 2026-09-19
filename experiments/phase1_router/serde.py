"""Tuple/set-preserving JSON codec.

MBPP+ inputs and outputs contain tuples (and occasionally sets) that plain
JSON flattens to lists. The dataset builder deserialises inputs with
EvalPlus's own ``mbpp_deserialize_inputs`` on the host, then encodes them with
this codec so the Docker grader can reconstruct the exact Python objects.
"""

from __future__ import annotations

from typing import Any


def encode(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return {"__t__": [encode(x) for x in obj]}
    if isinstance(obj, set):
        return {"__s__": [encode(x) for x in obj]}
    if isinstance(obj, frozenset):
        return {"__fs__": [encode(x) for x in obj]}
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        return {"__d__": [[encode(k), encode(v)] for k, v in obj.items()]}
    if isinstance(obj, complex):
        return {"__c__": [obj.real, obj.imag]}
    return obj


def decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__t__" in obj:
            return tuple(decode(x) for x in obj["__t__"])
        if "__s__" in obj:
            return set(decode(x) for x in obj["__s__"])
        if "__fs__" in obj:
            return frozenset(decode(x) for x in obj["__fs__"])
        if "__c__" in obj:
            return complex(obj["__c__"][0], obj["__c__"][1])
        if "__d__" in obj:
            return {decode(k): decode(v) for k, v in obj["__d__"]}
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(x) for x in obj]
    return obj
