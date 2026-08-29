# Paired trace quality review: cloud vs edge (28 August)

## Bottom line

Neither run completed the requested task: *explore the repository with several
agents and return a detailed consolidated report*. The edge run made more
usable partial progress—it returned an interim report from three completed
workstreams—but never closed the task. The cloud run also stopped at an interim
report, waiting for its trace/data worker.

So this is **not evidence that edge matches cloud on agent quality**, and it is
not a valid end-to-end winner/loser benchmark. It is strong evidence that the
local server, prefix cache, and tool protocol worked. It also isolates a
separate Claude Code security-monitor sidecall that the local model repeatedly
truncates at its 64-token output limit. It does **not** establish that the
agent main loop itself must be excluded from local placement.

## Scope and comparison caveats

I analysed:

- `traces/cloud-trace-28-08.jsonl` — 243 records; 235 Messages calls.
- `traces/edge-trace-28-08.jsonl` — 179 records; 175 Messages calls.

They are the same intended workload and use the same Claude Code session
shape, but they are not a deterministic A/B replay:

1. The initial instructions differ by one character (`nexplore` versus
   `explore`).
2. Once either model emits a different action, all later prompts, tool results,
   agent timing, and cache state can diverge.
3. Both captures end while the parent task is still waiting for workers.

Therefore a turn-by-turn tool-argument equality score would be misleading.
The appropriate quality comparison here is task outcome, structural action
behaviour, and failure mode—not text similarity or a claimed per-call accuracy
rate.

## Outcome quality

| Criterion | Cloud | Edge | Assessment |
| --- | --- | --- | --- |
| Requested final detailed report delivered | No | No | Both fail the task-level success criterion. |
| Parallel work launched | 4 agents across core, measurement/tests, trace/results, ops/docs | 5 intended workstreams; 7 `Agent` calls including re-launches | Both use the requested fan-out pattern. |
| Completed work available to parent | Interim report covering 3 of 4 sections | Interim report covering 3 of 5 sections | Edge has slightly broader partial exploration, but neither is final. |
| Parent failure mode | Waited for the trace/results worker; three cloud HTTP failures also occurred | Waited for purpose and architecture workers | Both are incomplete; this is not a completed-task model ranking. |
| Best defensible outcome | Incomplete partial report | Incomplete partial report, with more immediately useful content | **Edge narrowly wins partial-progress usefulness; neither wins task quality.** |

The trace endings are especially important. The final response from each
subagent is a progress utterance such as “Reading…” or “Synthesizing…”, not a
completed hand-back. The cloud parent eventually assembled a three-section
interim report. The edge parent assembled a three-section report with explicit
placeholders, then continued to state that two workers had not returned.

## Tool-call feasibility and disagreement

### What both models did successfully

Both trajectories made feasible read-only exploration actions. Their tool use
was limited to `Read`, `Bash`, and `Agent`; no destructive shell action was
observed. The local model emitted structurally usable tool calls and the proxy
served all local Messages requests successfully.

| Tool calls emitted | Cloud | Edge |
| --- | ---: | ---: |
| `Agent` | 4 | 7 |
| `Bash` | 43 | 25 |
| `Read` | 91 | 64 |

Both sides also made ordinary exploration mistakes: oversized reads, missing
paths, shell failures, and policy/classifier interruptions. These are agent
workflow failures, not evidence that the Anthropic-compatible local API could
not carry tools.

### Where the trajectories diverged

The cloud run launched four broad slices. The edge run initially launched five
more granular slices—purpose, core architecture, benchmarks, operations, and
tests—then re-issued two `Agent` calls for the unfinished operations/tests
slices. That is an early structural divergence, so later tool sequences are
not comparable action-for-action.

The decisive measured disagreement is in Claude Code's separate security
monitor, not in the fan-out workers themselves:

- Edge has 98 local calls whose system prompt identifies them as the security
  monitor. All request exactly 64 output tokens; 80/98 stop at `max_tokens`.
- Cloud has 97 equivalent security-monitor calls: 91 finish with `end_turn`
  and none stop at `max_tokens` (six are cloud failures without a stop reason).
- The route trace labels these calls as headerless “main” because they lack an
  agent ID, but their system prompt proves they are not the repository-agent
  main loop. Treating all headerless calls as main was an analysis error.

At the whole-trace level, 80/175 edge Messages calls (45.7%) ended in
`max_tokens`, versus 5/235 cloud calls (2.1%). The local worker calls returned
HTTP 200 and their last captured turns end normally. The pathology is
concentrated in the low-output security-monitor class, not basic tool transport
or proven main-loop reasoning.

## Edge server, cache, and GPU health

This trace ran on an **NVIDIA RTX 6000 Ada Generation**, not the current RTX
5090. It is therefore evidence for the 28-August RTX 6000 Ada configuration
only.

| Edge metric | Result | Interpretation |
| --- | ---: | --- |
| Local Messages / HTTP success | 174 / 174 (100%) | vLLM and the proxy remained available for the workload. |
| Local cache-probe availability | 174 / 174 | No `local-probe-unavailable` event in this run. |
| Probe state | 166 warm, 8 cold | The fan-out workload quickly established reusable prefixes. |
| Actual cache-use accounting | 173 available; 5.029M cache-read tokens; 1.226M created tokens | Prefix caching materially worked. |
| Predicted vs actual local cached tokens | 163/173 within 5% of total input; 163 exact token matches | The live probe was accurate for this captured request shape. |
| Token-weighted cache-read fraction | 78.7% | Most input prefill was avoided after warm-up. |
| KV-cache utilisation metric | 19.23% p50, 28.97% p90, 43.08% max | No confirmed capacity crisis, but this metric is not a retained-prefix-residency proof. |
| vRAM use | 92.22% p50; 3.734 GiB minimum free | High but stable model residency; no OOM occurred. |
| Engine concurrency / waiting | 3 running p50, 5 p90, 7 max; waiting 0 p90, 2 max | Some scheduling pressure, but not sustained queue saturation. |

Conclusion: **the local model was able to run properly as a server**. It had
healthy HTTP availability, working cache reuse, no confirmed active KV crisis, and no GPU
OOM. That does not mean it ran properly as the primary reasoning agent for
this task.

## Fan-out workloads and cache behaviour

The trace contains two overlapping fan-out patterns:

1. The parent launches repository-exploration workers, each with a shared
   Claude Code/agent scaffold but a distinct assignment: purpose/direction,
   core architecture, benchmarks/traces, operations/deployment, and tests.
2. The parent repeatedly asks for worker status and then attempts to synthesize
   the returned findings. Those parent turns share a much larger 25--27-tool
   scaffold and a growing worker-result history.

The expected cache pattern is: the first member of a shared-scaffold cohort
creates blocks; concurrently or subsequently launched siblings reuse that
common prefix, but not the task-specific suffix. That is what happens early:

| Fan-out example (edge timestamps) | Time gap | Expected reuse | Observed reuse | Reading |
| --- | ---: | --- | --- | --- |
| First operations/deployment worker: 14,192-token prompt, 15 tools | baseline | Cold first worker | 0 read; 14,112 created | Expected cold cohort seed. |
| First core-architecture worker: 14,224-token prompt, 15 tools | 40.6 s later | Reuse common agent scaffold, create its distinct instruction/suffix | 12,544 read; 1,568 created | Clear cross-worker fan-out reuse. |
| First benchmarks/traces worker: 14,263-token prompt, 15 tools | 43.4 s after that | Same shared scaffold | 12,544 read; 1,568 created | A second sibling reuses the same base. |
| Operations worker continuation: 35,808-token prompt | 68.3 s after its seed | Reuse nearly all prior worker context | 34,496 read; 0 created | Fully warm continuation. |
| Architecture continuation: 60,522-token prompt | 231.8 s after its seed | Reuse its own accumulated history | 59,584 read; 0 created | Fully warm continuation despite concurrent siblings. |

This is real fan-out benefit, not merely repeated retries: distinct workers
with closely matching 15-tool scaffolds reused the shared 12,544-token prefix
within about a minute. Their later continuations reached 96--99% reuse. The
most common warm state in the whole local run was 166/174 calls; the
token-weighted cache-read fraction was 78.7%.

### The unexpected cache loss

There is one strong cache discontinuity. The parent had a 49,024-token,
27-tool, 19-message status/synthesis prompt at edge timestamp
`1787933499.314`. It read 48,608 tokens from cache, created none, and had
413.8 ms TTFT. **357.4 seconds later**, the immediately evolved 49,064-token,
27-tool, 20-message version was reported cold: 0 read, 48,608 created, and
24.71 s TTFT.

The 40-token growth and one extra message cannot by themselves explain losing
every shared block; a common prefix should still have survived. The sampled KV
field was 0% at both calls, and never exceeded 43.08% in the capture. That
*looks* unlike pressure eviction, but it is not conclusive: the same 0% sample
on the warm call coexists with the live probe proving 48,608 resident cached
tokens. In this vLLM metric, `kv_cache_usage_perc` evidently does not provide a
reliable count of retained prefix blocks at that instant (or has different
sampling semantics).

The trace cannot prove the underlying process event, but it supports a narrow
conclusion: **the local prefix-cache state disappeared or was reset between
these calls.** Possible causes include a vLLM cache/engine reset, eviction, or
another lifecycle event. The current telemetry cannot distinguish them, so it
would be wrong to label this either a confirmed unintended eviction or a
confirmed reset. This cold restart alone added roughly 24.3 seconds to TTFT
for that parent turn.

There are other legitimate cold starts: they occur when a new worker or a
materially different parent scaffold begins. They should not be called
unexpected eviction. Conversely, no example was found of two clearly matching
fan-out siblings failing to reuse their common prefix. The one apparent loss is
the near-identical parent pair above and should be investigated with vLLM
lifecycle/reset logs and a prefix-block-residency counter in the next run.

## Latency, network, and jitter

| Metric | Cloud-only trace | Edge local trace | What can be concluded |
| --- | ---: | ---: | --- |
| TTFT p50 / p90 | 2.84 s / 15.68 s | 11.95 s / 33.81 s | Local was about 4.2x slower at p50 and 2.2x slower at p90 for streamed calls. |
| Total latency p50 / p90 | 4.74 s / 54.02 s | 7.88 s / 75.55 s | Edge was slower end-to-end on this high-concurrency agent workload. |
| Decode time per token p50 / p90 | 11.99 / 152.27 ms | 46.53 / 215.05 ms | Local decode was roughly 3.9x slower at p50. |
| Output rate p50 | 74.9 tok/s | 22.6 tok/s | Local throughput was materially lower. |

The cloud path is highly variable: its measured network component has 527 ms
p50, 14.39 s p90, and 114.33 s maximum; its link-jitter EWMA is 5.24 s p50 and
13.32 s p90. That is real cloud-path variability in this capture, and it
contributes to cloud tail latency and the six cloud 503/504 outcomes.

Do **not** read the edge link EWMAs as a local-network jitter comparison. There
is only one cloud-routed edge call, so there is one cloud network sample
(237 ms). For the 174 local calls, `network_ms` includes local proxy/backend
work rather than a comparable WAN path, and the cloud-link monitor has no new
local-link observations. This run cannot establish that the edge had better or
worse WAN jitter; it establishes that local compute/queue time dominated the
edge latency here.

## Cost and cache savings

The trace's counterfactual Anthropic list-price accounting reports:

| Quantity | Cloud trace | Edge trace |
| --- | ---: | ---: |
| Priced Messages | 229 | 175 |
| Recorded cloud bill | $5.527 | $0.000080 (one fallback call) |
| Estimated avoided cloud cost | $0 | $16.482 | 

The edge saving is a **counterfactual cloud-token price**, not profit: it does
not include the GPU, container, or engineering cost, and it is conditional on
the local output being acceptable. Since the local task did not finish, this
$16.482 must not be presented as a quality-matched saving.

## Why edge diverged

The proven placement mismatch is the security-monitor call class, not cache
failure or GPU memory exhaustion:

1. Claude Code sends a long transcript to a security-monitor model and grants
   it only 64 output tokens. The local model commonly emits reasoning before
   its concise allow/block decision, exhausting that budget.
2. The current router sees the request as a feasible, tool-free 27--30K-token
   request, so it routes it local. It has no call-class feature for this
   special low-output monitor.
3. Cloud completes 91 comparable monitor calls normally; edge truncates 80.
   This is a high-confidence, narrow routing signal. It is unrelated to the
   local output clamp: the request itself asks for `max_tokens: 64`, and these
   records have no proxy clamp applied.
4. Moderate concurrency compounded the latency: three running requests at the
   median and five at p90, combined with eager-mode/sequence-cap overhead,
   made local turns slower. It does not explain the monitor's 64-token
   truncation.

The cloud trace has its own incomplete outcome and WAN failures, so it is not a
clean gold reference for repository-agent quality. It is, however, a useful
reference for this exact security-monitor call class.

### Concrete trajectory examples

The divergence is visible in the responses, not just aggregates:

- Edge successfully obtained a detailed benchmark worker hand-back and then
  summarized its useful finding: high prefix reuse but approximately 12-second
  median TTFT. That is a good local subtask outcome and shows the model can
  read, calculate, and report a bounded slice.
- The edge parent later produced a three-section interim report, explicitly
  leaving **Purpose & Direction** and **Core Architecture** as placeholders.
  Its final captured user-facing response was still a status explanation that
  those two workers had not returned. It never converted its partial findings
  into the requested complete report.
- Cloud followed a similar partial-success pattern: it produced an interim
  report from three of four workers and waited for the trace/results worker.
  It also never delivered a completed synthesis before capture end. This is why
  the report does not claim cloud task success.
- The asymmetric structural signal is that edge truncated 80 of 98
  64-token security-monitor calls, whereas cloud completed 91 analogous calls.
  This is a sidecar/control-plane failure. The incomplete fan-out reports do
  not, on their own, prove a local orchestration-quality failure.

This distinction matters for routing: several individual local tool actions
were feasible and cache-efficient, while the *sequence* failed. Evaluating only
the JSON validity of one `Read` or `Bash` call would incorrectly label this
workload a success.

## Recommended routing policy after this study

Keep the existing hard feasibility gates and add one narrow call-class gate;
otherwise retain local-first placement for experimentation:

```text
server-side tool / unsupported local tool / probe unavailable / token budget fail
    -> cloud

security-monitor system prompt with a 64-token output allowance
    -> cloud

otherwise
    -> local when the existing feasibility rules allow it
```

Concretely, this trace supports keeping Qwen3.8 enabled for the agent workload
while forcing only the failing security-monitor class to cloud. Record
per-call-class outcomes and add a local-to-cloud retry/cooldown only after a
non-monitor class demonstrates a repeated failure. Cache warmth remains a
preference among feasible calls, never proof of output quality.

For a valid next comparison, run the exact same initial prompt and environment
under cloud-only and local-policy conditions, keep the trace until both runs
end, and score a predeclared completion rubric plus tool-action divergence.

## Reproducibility

Metrics were calculated from the two JSONL traces with `jq`, using `/v1/messages`
records only for request, timing, placement, cache, tool, and resource counts.
No prompt or source contents are reproduced in this report.
