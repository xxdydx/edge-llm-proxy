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

Tier 0 — Low-hanging fruit (get this working first, not really "research")
Rule-based router (prompt length + prefix-cache-hit + queue depth threshold)
Basic prefix caching via vLLM's built-in feature, measured Local-only /
cloud-only baselines for comparison

Tier 1 — Solid systems contribution (OSDI-appropriate, your core plan)
Cache-aware routing policy — router uses live cache-hotness as a first-class
signal, not just a caching side effect Oracle/upper-bound comparison — run every
call both ways, report how close your policy gets to the theoretical best
Cost/quality Pareto sweep — vary routing weights, plot the tradeoff curve
instead of one static number

Tier 2 — The genuinely novel systems idea (still OSDI, but the "one real idea"
reviewers want) Cohort-sequential cache-routing coupling — show that routing
decisions early in a fan-out cohort change the optimal decision for later
siblings (warming cache changes the calculus). The counterintuitive claim:
sometimes deliberately routing sibling #1 to local, even though it looks worse
in isolation, is globally optimal for the cohort. This is your strongest
concrete OSDI-shaped contribution — a real, non-obvious interaction effect
nobody's characterized. A live cache-hotness signal exposed from vLLM to an
external scheduler — vLLM doesn't currently expose this cheaply; building a
lightweight interface for it is a small, real, potentially-upstreamable systems
contribution on its own.

Tier 3 — ML-flavored contribution (MLSys / workshop-tier, blends learning +
systems) Confidence-calibrated cascade routing — instead of a hand-tuned
threshold, use the local model's own output confidence (token log-probs /
entropy) to decide whether to escalate to cloud, in the style of cascade
inference / RouteLLM. The research question becomes: how well-calibrated does a
small model's confidence need to be for cascade routing to actually work, and
does that calibration quality change under quantization?
Distillation-specialized local model + routing co-design — instead of a generic
small model, distill the cloud model's behavior on this agent's specific task
distribution (Panorama-style QLoRA), then study how routing policy interacts
with a specialized local model vs. a generic one. Two levers instead of one —
does specialization shrink the routing problem, or just shift where the
threshold sits? Learned routing policy (small classifier/bandit) trained online
from routing outcomes, compared against the hand-tuned utility function — does a
learned policy actually beat a well-tuned rule, and by how much, and does it
generalize across task types?

Tier 4 — NeurIPS/ICLR-level moonshot (genuine ML research, much higher risk,
probably beyond 13 weeks alone but worth knowing exists) Theoretical
characterization of the cache-routing coupling as a sequential decision/bandit
problem with state-dependent rewards — formalize the cohort-routing problem
(Tier 2's idea) as a proper online learning problem with regret bounds. Prove
(or empirically demonstrate with strong evidence) that a cache-state-aware
policy achieves provably better regret than a cache-unaware one under realistic
cohort-arrival distributions. This is the kind of theory-meets-systems paper
that could genuinely target a learning-theory-adjacent venue, but it requires
real theoretical machinery (regret analysis, online convex optimization or
contextual bandits), which is a big lift on top of your systems build.
Meta-learned/self-improving routing across many agents and task types — a
routing policy that transfers across different agents/tasks without retraining
from scratch, learning a shared representation of "task difficulty → local
suitability" that generalizes. This is basically a mini-research-program on its
own (multi-task routing generalization), not a semester project. Co-training the
local model and the router jointly — instead of treating distillation and
routing as separate stages, jointly optimize: train the small model specifically
to be good at the subset of calls the router would send it, and train the router
specifically around the small model's actual (evolving) capability boundary,
iterating between the two. This is a genuinely novel co-design idea nobody's
fully explored in this specific local/cloud LLM-serving context — closer to a
"new training paradigm" paper than a systems paper, and would need serious ML
modeling chops, not just systems engineering.
