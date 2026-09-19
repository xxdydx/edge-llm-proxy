# Edge LLM — onboarding (Student B, cloud + edge)

**One line:** A local small/quantized model + KV/prefix cache that serves
latency-sensitive calls itself and offloads heavy ones to the cloud.

**Setup.** Everyone builds against **one shared local harness** (provided day
one — the local agent loop with plug points for memory / inference /
analytics-agent, a common workload, and measurement hooks). You own the
**inference plug point**; the memory and agent seams are stubbed for you.
**Standalone**: your module + a stub cloud endpoint, nothing blocks you.

## Goal

- **Local serving:** a local model + local KV/prefix cache + batching, behind an
  Anthropic-API-compatible endpoint the harness hits via `ANTHROPIC_BASE_URL`.
- **Routing:** decide per call — serve local vs. offload to cloud — by
  difficulty, latency, cost, and local capacity.
- **Cache reuse:** share prefix cache across a fan-out cohort's siblings (shared
  ancestor prefix).

## What you build (pluggable)

```
route(call) -> {local|cloud}      # per-call placement
serve_local(call) -> completion   # local model + KV/prefix cache
prefix_key(call) -> hash          # cohort prefix sharing
```

I will be using Claude Code as the harness for now.

## Plan

1. **wk 1–2:** register your local endpoint as the shared harness's inference
   plug point (+ stub cloud) over a throttled link (`tc/netem`); drive it with
   the harness's fan-out workload.
2. **wk 3–5:** local serving + prefix cache + batching.
3. **wk 4–6:** local↔cloud routing vs. cloud-only / local-only.
4. **wk 6+ (optional):** draft-local + verify-cloud.

## Success metrics

- **Local serve rate** (fraction handled at the edge).
- **Prefix-cache reuse** (recompute avoided across a cohort).
- **Routing quality:** latency/$ at matched quality vs. cloud-only and
  local-only.
- **Throughput/cost:** tokens/sec + $ per turn.

## Ramp up

Read: vLLM/SGLang RadixAttention (2312.07104); speculative decoding
(2211.17192); quantization (llama.cpp, GPTQ/AWQ); FlowMesh v1 (2510.26913). Env:
local GPU box + stub cloud endpoint + `netem`. **First milestone (wk 2):** the
harness hits your local endpoint via `ANTHROPIC_BASE_URL`, and you route one
call local-vs-cloud over a throttled link, measured.


