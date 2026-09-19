"""Phase 1 of the continuously-learned edge/cloud routing project.

Controlled two-model existence-case pilot on a standalone, non-agent coding
workload (MBPP+ v0.2.0). Pairs Qwen3.8-27B-NVFP4 (local vLLM, 100K context)
against DeepSeek-V4-Flash (Lumid), runs every problem through BOTH with
deterministic decoding, grades each completion with objective isolated unit
tests (Docker only -- model code never runs on the host), and asks:

  * offline oracle -- is there meaningful local traffic at cloud-level quality?
  * pre-dispatch classifier -- can a request-only policy beat random / simple
    heuristics on held-out task groups within epsilon = 0.02 of always-cloud?
  * cache-aware switching cost -- how does the frontier move at 0x vs 10x
    switch penalty?

Reuses the hardened checkpoint / identity / bounded-fail-fast / token
utilities from ``experiments.capability_router.executor``.
"""
