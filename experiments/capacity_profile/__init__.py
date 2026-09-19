"""27B edge capacity/throughput profile.

A serving-performance experiment, NOT a quality experiment: it drives the local
Qwen3.8-27B vLLM endpoint (relay ``:18004``) with the real Phase 1 MBPP+ prompt
distribution under a closed-loop concurrency ladder and an open-loop Poisson
arrival ramp, scrapes vLLM ``/metrics`` throughout, evaluates a provisional SLO,
stops at saturation, and rolls the raw per-request rows up into a capacity
envelope (max sustainable requests/s and concurrent requests under SLO, the
throughput knee, and the limiting resource).
"""
