# Edge LLM Client — build & research plan

Target: RTX 5080 (16 GB, sm_120) FlowMesh SSH box, disposable 8h sessions.
Harness: Claude Code via `ANTHROPIC_BASE_URL`.
Horizon: ~13 weeks. Tier 0 → Tier 2 is the realistic scope; Tier 3 is a stretch.

---

## 0. The one thing you build

Everything is one process: **`edgeproxy`** — an Anthropic-Messages-compatible
server that sits between Claude Code and two backends.

```
Claude Code ──ANTHROPIC_BASE_URL──▶ edgeproxy ──┬──▶ vLLM (local, OpenAI API)
                                    │           └──▶ Anthropic API (cloud)
                                    │                  ▲ shaped link
                                    ├─ router(call) -> {local|cloud}
                                    ├─ prefix_key(call) -> hash
                                    ├─ cache probe -> resident prefix len
                                    └─ trace recorder -> JSONL
```

The three plug points from the brief map cleanly:

| Brief | Where it lives |
| --- | --- |
| `route(call) -> {local\|cloud}` | `edgeproxy/router/` — swappable policy classes |
| `serve_local(call)` | `edgeproxy/backends/local.py` → vLLM |
| `prefix_key(call)` | `edgeproxy/cache/keying.py` |

Keep the router a pure function of a `CallFeatures` struct. Every policy
(static rule, utility, oracle, learned) implements the same interface. This is
what makes the Tier 1 sweep and Tier 2 lookahead cheap to add later.

---

## 1. Three decisions that shape everything else

### 1.1 Record and replay traces from week 1

Claude Code is non-deterministic and interactive. You cannot run an oracle
comparison ("serve every call both ways") against it — the second run diverges
after the first differing token, and the conversation state is gone.

So: `edgeproxy` records every request it sees to JSONL (full messages, tool
defs, `cache_control` markers, arrival timestamp, parent session id, and the
cloud reference completion). Then build `replay.py`, which feeds recorded calls
back through the router with synthetic arrival timing.

This unlocks:
- **Oracle**: exhaustive/DP over placements for a recorded cohort.
- **Policy sweeps**: same trace, N router configs, apples-to-apples.
- **Fast iteration**: no GPU needed to test router logic, only to calibrate the
  cost model.

Live Claude Code is then reserved for the demo and for end-to-end sanity, not
for the numbers in the paper. **Do this first. It is the highest-leverage item
in the plan.**

### 1.2 Route call *classes*, not sessions

A quantized 7B model cannot reliably drive Claude Code's main agent loop —
tool-call schema drift and repair loops will eat you. Don't fight it.

Claude Code's traffic is naturally bimodal:

- **Main-loop calls** — long context, tool definitions, must produce valid
  `tool_use` blocks. Hard. Mostly cloud, at least at first.
- **Sidecalls** — conversation titles, "is this a new topic" classification,
  file/output summarization, and other short-output utility calls. Short,
  schema-light, latency-visible-but-not-critical. A 7B model handles these fine.

That split is a working Tier 0 system in week 2 with a non-trivial local-serve
rate, and it gives the router a real difficulty signal instead of a synthetic
one. Widen the local class over time as quality gating proves out; the *shape of
that boundary* is itself a result worth plotting.

### 1.3 Cache hotness must be queryable *before* you commit a request

The Tier 2 idea needs the router to know "how many tokens of this prospective
prefix are resident in vLLM right now" — cheaply, without scheduling the
request. vLLM does not expose this today (it reports an aggregate hit-rate
metric after the fact, which is useless for a decision).

Build both, in this order:

1. **Shadow radix tree in the proxy** (no vLLM patch, ~200 LOC). You know every
   request you sent local; hash token blocks with the same `block_size`, keep an
   LRU sized to the engine's `num_gpu_blocks`. Approximate, but zero-risk and
   available immediately.
2. **Real probe endpoint** (vLLM patch). In vLLM V1, `KVCacheManager` already
   hashes request tokens into `BlockHash`es and holds a
   `cached_block_hash_to_block` map. A probe is: tokenize → block-hash the
   prefix → walk the map → return longest resident prefix length. Dict lookups,
   microseconds, no scheduler involvement. Expose it as an API-server route.

Validate (1) against (2). The delta between them is itself a nice figure ("how
well can an external scheduler estimate cache state without engine access?"),
and (2) is the small upstreamable contribution the brief points at.

---

## 2. Hardware budget

16 GB is enough, but only if you don't waste it on weights. KV cache headroom
*is* the experiment — a big model with no room for cache makes prefix reuse
unmeasurable.

- **Model**: 4-bit AWQ 7–8B. `Qwen2.5-Coder-7B-Instruct-AWQ` matches the
  workload; `Qwen3-8B-AWQ` is the general-purpose alternative. ~5 GB weights.
- **KV**: ~8–9 GB at `--gpu-memory-utilization 0.90`. For a 7B GQA model
  (28 layers, 4 KV heads, head_dim 128) that's ~56 KB/token fp16 → ~150K tokens
  resident. `--kv-cache-dtype fp8` roughly doubles it.
- **Context**: Claude Code's system+tools prefix is ~10–15K tokens. So you hold
  on the order of ten full contexts. That pressure is *good* — it makes eviction
  real and the cohort effect measurable. Don't tune it away.
- **Prefix cache**: `--enable-prefix-caching` (default on in V1).

**Day-one risk check**: sm_120 (Blackwell consumer) needs a recent vLLM on a
CUDA 12.8+ build, and AWQ-Marlin / attention-backend kernel coverage on sm_120
has historically lagged. Verify a 10-token generation works *before* writing any
proxy code. If vLLM is broken on sm_120, fall back to SGLang (also has a radix
cache) or llama.cpp (loses continuous batching — a real loss, but the routing
and cohort work survives).

---

## 3. Session workflow (disposable containers)

8h TTL, filesystem wiped, so:

- `bootstrap.sh` in the repo: clone → `uv sync` → pull model weights → launch
  vLLM → launch proxy. Target under 10 minutes cold.
- Model weights re-download each session (~5 GB, fine). Set `HF_HOME` to scratch.
- **Never** leave results only on the box. `results/` is committed, or rsync'd
  down at end of session.
- Once `bootstrap.sh` stabilizes (~week 3), bake it into a Docker image off the
  FlowMesh GPU SSH base image and add `image:` to the task spec.
- Router development and replay experiments need no GPU — do them locally, and
  use box time only for calibration and cache-dependent runs.

---

## 4. Link shaping

The brief says `tc/netem`. Try it, but expect `NET_ADMIN` to be unavailable in
the container. Fallback is better anyway: **inject delay in the proxy's cloud
backend path** — a configurable `(base_rtt, jitter, bandwidth)` model applied to
the request/response. Reasons it's preferable:

- Reproducible across sessions and machines.
- Sweepable as an experiment parameter without root.
- Applies only to the cloud path, which is what you actually want to vary.

Keep a `--shaping=netem|proxy|none` flag so you can show they agree if netem
does work.

---

## 5. Quality measurement (the part that's usually underspecified)

"Latency/$ at matched quality" needs a quality number that isn't an LLM judge's
mood. Avoid an LLM judge as the primary metric — its variance will swamp your
routing deltas.

### 5.1 What counts as "the same answer"

Not textual match. **Action equivalence**: same tool name, same
semantically-relevant arguments. In an agent loop the tool call is what
propagates to the next request; the prose around it doesn't. This is
programmatically checkable and it's exactly the property that determines whether
a replayed trace is still valid.

| Call type | Comparison | Noise |
| --- | --- | --- |
| Classification / title / yes-no | exact or normalized match | low |
| Tool-calling turns | tool name + args structural equality | low |
| Free-text summaries | embedding similarity | high |

### 5.2 Three rungs, one trace

All three read the same recorded JSONL. Cost rises ~1000× per rung and fidelity
rises with it. **Tune on rung 1, characterize with rung 2, validate with
rung 3.** Never tune against rung 3 — too slow and too noisy to iterate on.

**Rung 1 — Teacher forcing** (run constantly: every router tweak, every sweep)

Score local's output against the recording, then *discard it* and feed the
recorded cloud output forward, so the next call's input is still the one on
disk. Every call gets scored; the replay never leaves the rails. Name and
mechanism borrowed from seq2seq training.

Answers: *which weight settings are better?*

Known bias — **optimistic**. A bad local edit at call 2 would, in a real run,
fail the test at call 3 and cost several recovery calls. Teacher forcing erases
that. It measures per-call accuracy, not session outcome. State this explicitly
in the paper rather than letting a reviewer find it.

**Rung 2 — Divergence horizon** (run per candidate policy)

Replay until local's action first differs from the recording, then stop and
record *when*. One number per session; run over ~50 sessions and plot the
fraction still on-trace after k calls — a survival curve.

Answers: *how aggressively can I route local before sessions leave the map?*

More honest than any averaged number, and directly actionable: if the curve is
flat for the first few calls then collapses, that's a routing feature (session
depth) waiting to be implemented. Known bias — **pessimistic**: it scores
"diverged but equally good" as death.

Rungs 1 and 2 fail in opposite directions, which is why reporting both is
defensible. Each is ~40 lines of Python over the trace file.

**Rung 3 — Live end-to-end** (final policies only)

A suite of ~20–30 deterministic agentic tasks with *programmatic* checkers
(unit test passes, file contains string, command exits 0), run through the full
Claude Code loop under each policy. Expensive and sparse.

Answers: *did the task actually get done?* This is the only rung that can credit
a divergence that was fine, and the number that makes the paper credible.

---

## 6. Schedule

### Weeks 1–2 — milestone: harness hits local endpoint, one call routed

- [ ] Verify vLLM on sm_120 (generation works, prefix caching on). **Do first.**
- [ ] `POST /v1/messages`: non-streaming, then SSE with the full Anthropic event
      sequence (`message_start` / `content_block_start` / `content_block_delta` /
      `content_block_stop` / `message_delta` / `message_stop`). Claude Code needs
      streaming; getting the event grammar exactly right is most of the work.
- [ ] `POST /v1/messages/count_tokens`, `/v1/models`.
- [ ] Anthropic↔OpenAI translation: system prompt, multi-part content, `tool_use`
      / `tool_result` ↔ OpenAI tool calls, stop reasons.
- [ ] Cloud passthrough backend + proxy-side link shaping.
- [ ] Router v0: static call-class rule (§1.2).
- [ ] Trace recorder → JSONL.
- [ ] Measurement hooks: TTFT, ITL, e2e, tokens in/out, placement, $ estimate.
- **Deliverable**: Claude Code session driven end-to-end, sidecalls served local,
  main loop to cloud, with a latency/cost table.

### Weeks 3–5 — local serving, prefix cache, batching

- [ ] Model bake-off on the 16 GB budget (quality vs. KV headroom).
- [ ] `prefix_key`: derive from Claude Code's `cache_control` breakpoints + a
      canonicalized message-prefix hash. Verify vLLM actually hits on it.
- [ ] Characterize prefix caching: TTFT vs. resident-prefix-length curve. This
      curve *is* your router's cost model — fit it and store the coefficients.
- [ ] Shadow radix tree (§1.3 step 1).
- [ ] vLLM cache-probe patch (§1.3 step 2) + validation vs. shadow.
- [ ] Batching: concurrency sweep, find the knee where queueing delay overtakes
      throughput gain. The router needs this as its capacity term.
- [ ] `replay.py` working against recorded traces.
- [ ] Scorers: action-equivalence check (§5.1), then teacher-forcing and
      divergence-horizon modes (§5.2). ~40 LOC each once replay exists.
- **Deliverable**: prefix-cache reuse numbers, calibrated local cost model,
  local-only and cloud-only baselines.

### Weeks 4–6 — routing (overlaps above)

- [ ] Utility router: `score = w_lat·E[latency] + w_cost·$ + w_qual·E[quality]`,
      with `E[latency]` fed by the fitted cost model **and live cache hotness**.
- [ ] Oracle via replay: exhaustive over placements for small cohorts, DP or
      beam for larger. Report policy-vs-oracle gap.
- [ ] Pareto sweep over `(w_lat, w_cost, w_qual)` → cost/quality frontier, not a
      single operating point.
- [ ] Ablation: same router with the cache-hotness term removed. This is the
      Tier 1 claim — cache-awareness as a first-class routing signal.
- **Deliverable**: Pareto plot vs. cloud-only / local-only / oracle, plus
  divergence-horizon survival curves per policy.

### Weeks 7–10 — Tier 2: cohort-sequential cache-routing coupling

The core idea, stated precisely:

> For a fan-out cohort of N siblings sharing ancestor prefix P, myopic per-call
> routing is globally suboptimal. Serving sibling #1 locally pays a one-time
> `|P|`-token prefill that all N−1 later siblings amortize. A myopic router
> prices that prefill against sibling #1 alone, judges local too slow, and sends
> it to cloud — after which sibling #2 faces the identical cold cache, and the
> whole cohort leaks to cloud.

It's an investment problem: the action changes the state, and the reward is
state-dependent. That framing is also the on-ramp to Tier 4 if you ever want it.

**Getting fan-out**: Claude Code's subagent/Task tool spawns siblings off a
shared ancestor context — that's a real cohort, and the natural source. Record
those traces, then synthesize controlled cohorts from them for the sweep.

**Experiment**: sweep cohort width `N` × shared prefix length `|P|` × local load
× link RTT. Compare four policies:

| Policy | Description |
| --- | --- |
| cloud-only / local-only | baselines |
| myopic | per-call optimal given current cache state |
| cohort-aware | lookahead over remaining siblings |
| oracle | exhaustive over the cohort's placements |

**Headline figure**: heatmap over `(N, |P|)` of cohort-aware improvement over
myopic, with the region where myopic collapses to cloud-only marked. Expect an
interior optimum — routing *all* siblings local saturates the engine, so the
best policy is a mix, which makes the curve interesting rather than monotone.

**Confounds to control explicitly:**
- Eviction between siblings — measure residency lifetime under load; the effect
  vanishes if P is evicted before sibling #2 arrives, and *that boundary* is a
  result.
- Sibling suffix heterogeneity — vary independently of `|P|`.
- Arrival timing — simultaneous vs. staggered fan-out change the answer.

**Ablation worth doing**: derive a closed-form threshold from the analytical
model (when does amortized prefill beat N cloud calls?) and show the empirical
crossover matches. A predictive "when does this matter" rule is more useful to a
reviewer than a single improvement percentage.

### Weeks 10–13 — writeup, plus one Tier 3 stretch if ahead

Pick **one**, only if Tier 2 is fully in hand:

- **Confidence-calibrated cascade** (cheapest — vLLM already returns logprobs).
  Replace the hand-tuned quality threshold with local-model entropy/logprob, and
  ask the real question: *how well-calibrated must a small model be for cascade
  routing to pay off, and does 4-bit quantization degrade that calibration?* The
  quantization-vs-calibration angle is the novel part and you get it nearly free
  since you're quantized anyway.
- **Learned router** (bandit over the same features) vs. the tuned utility
  function. Honest negative results are fine here.
- Distillation co-design is a semester on its own. Don't.

---

## 7. Risk register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| vLLM broken on sm_120 | Blocks everything | Verify day 1. Fall back to SGLang, then llama.cpp |
| Anthropic SSE grammar mismatch | Claude Code hangs/errors | Capture real API streams and diff event-by-event; write a conformance test |
| Local model can't do tool calls | Kills naive "local serves everything" | §1.2 — route call classes; never a project-level dependency |
| 8h TTL / wiped disk | Lost work, slow iteration | `bootstrap.sh` → Docker image; results committed; router dev off-GPU |
| `NET_ADMIN` unavailable | No netem | Proxy-side shaping (§4), which is better anyway |
| Cohort effect is small | Weakens Tier 2 | The *boundary* (when it appears/vanishes) is the result. Sweep hard enough to find where it's large |
| Quality metric too noisy | No "matched quality" claim | Programmatic checkers, not LLM judges (§5) |

---

## 8. Repo layout

```
edgeproxy/
  server.py            # /v1/messages, SSE, count_tokens
  translate.py         # Anthropic <-> OpenAI
  backends/
    local.py           # vLLM client
    cloud.py           # Anthropic passthrough + shaping
  router/
    base.py            # CallFeatures, Policy interface
    static.py utility.py cohort.py oracle.py
  cache/
    keying.py          # prefix_key
    shadow.py          # shadow radix tree
    probe.py           # vLLM probe client
  trace/
    record.py replay.py
  metrics.py
bench/
  tasks/               # deterministic agentic tasks + checkers
  sweep.py
vllm-patch/            # cache probe endpoint
bootstrap.sh
results/
```

## 9. First three days

1. vLLM up on the box, 10-token generation, prefix caching confirmed on. Record
   `nvidia-smi`, engine startup log (`num_gpu_blocks`), and a TTFT baseline.
2. Non-streaming `/v1/messages` that Claude Code can complete one turn against.
3. SSE streaming + tool-call translation. This is the real work of week 1 —
   budget for it.

Everything else in weeks 1–2 is downstream of these three.
