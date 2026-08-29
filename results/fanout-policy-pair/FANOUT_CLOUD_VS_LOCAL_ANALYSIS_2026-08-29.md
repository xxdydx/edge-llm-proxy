# Paired Fan-Out Evaluation: Cloud Claude vs Local Qwen3.8

**Run:** `20260829T054236Z`  
**Date analysed:** 2026-08-29  
**Workload:** one read-only, five-agent exploration of `edgeproxy/`, launched concurrently through a cloud-only proxy and a static-routing proxy  
**Cloud backend:** `claude-sonnet-5` through the configured Lumid gateway  
**Local backend:** `Inferact/Qwen3.8-27B-NVFP4` through vLLM on an NVIDIA RTX 6000 Ada Generation  
**Purpose:** determine when local inference is operationally healthy and qualitatively good enough, when cloud is safer, and what routing policy this pair supports.

## Executive conclusion

The local GPU path was operationally healthy. All **50/50 local message calls returned HTTP 200**, all were admitted as `fits`, the live cache probe was available on every call, and its predicted cached-token count matched the post-request actual count exactly on **50/50 calls**. The run reused **1,197,952 of 2,051,488 input tokens (58.4%)**, with cache hits on **48/50 requests (96.0%)**. Peak sampled KV-cache use was only **18.46%**, peak sampled running requests was three, queued requests stayed at zero, host RAM stayed below 38%, and no OOM, vLLM crash, or local-probe failure occurred. There is no evidence that GPU capacity or cache eviction harmed this run.

Local quality was mixed but useful. Qwen successfully performed all ordinary file operations: **51 Read + 24 Bash calls, all successful**. It eventually completed all five delegated investigations and reconstructed a detailed, mostly accurate report. Its weak point was agent orchestration: **2 of 7 attempted `Agent` calls (28.6%) contained malformed JSON**, versus **0 of 5 cloud Agent calls**. Claude Code recovered by retrying, but the errors added calls, latency, and complexity. Qwen was also much more verbose and slower to decode: it emitted **84,750 output tokens**, 2.85 times the cloud trace's 29,748, at a median **28.0 tokens/s** versus cloud's **138.8 tokens/s**.

The current 4,096-token local output clamp was too small for this workload. Local hit `max_tokens` on **11/50 calls (22%)**, including ten worker-report turns and the parent synthesis. Those truncations caused repeated “I was cut off” continuation turns. Cloud was not clean either: three cloud calls recorded `max_tokens`, including two approximately 301-second synthesis calls, and another approximately 337-second parent call had no stop reason. Therefore this pair does **not** show that cloud always solves long-form synthesis. It does show that the local 4K clamp predictably truncates Qwen's verbose reports.

Most importantly, **neither saved Markdown file is a complete final report**. The runner captured only Claude Code's last assistant continuation. The cloud file begins `## Open Questions (continued)` and contains roughly **18%** of the report text recoverable from the last three parent responses. The routing file begins in the middle of the telemetry section and contains roughly **49%** of the report text recoverable from the last two parent responses. Both end with the completion marker, so the marker proves that the last continuation finished, not that the beginning was captured. As delivered, the local file is far more useful—14.2 KB versus 3.1 KB—but neither passes the requested end-to-end outcome.

The strongest routing design supported by this run is a **hybrid parent/worker split**:

- Keep read-heavy worker/subagent turns local when the live probe succeeds and the request fits. This is where Qwen was reliable and where prefix reuse materially reduced TTFT.
- Prefer cloud for top-level orchestration and final synthesis, especially turns that can launch or collect `Agent` work. In this Claude Code build, top-level turns can be recognized before placement by the presence of the `TaskOutput`, `ScheduleWakeup`, or `Workflow` tools; worker turns had a smaller tool suite without those three tools.
- Raise the local clamp from 4,096 to a headroom-aware 8,192 for experiments before concluding that Qwen cannot finish worker reports. The largest observed input was 71,377 tokens, so 71,377 + 8,192 = 79,569, still below the configured 90,000-token effective budget.
- Add failure-aware state: after a malformed local tool call or a local `max_tokens` stop, route the recovery/continuation turn cloud under a conservative policy. Do not rely on HTTP 200 or valid top-level response JSON as a quality signal.

Applied counterfactually to the observed local trajectory, routing only the seven top-level 22-tool calls cloud and retaining the other 43 calls locally would give an **86% local request share**. That is not a guaranteed replay result—the trajectory would change after mixed placement—but it shows that conservative control-plane routing need not sacrifice most local experimentation.

## 1. Scope, files, and comparison method

Only the new matched pair was analysed.

### Inputs

- Cloud trace: `traces/fanout/cloud_20260829T054236Z.jsonl`
- Routing/local trace: `traces/fanout/routing_20260829T054236Z.jsonl`
- Cloud saved output: `results/fanout-policy-pair/cloud_claude_20260829T054236Z.md`
- Routing saved output: `results/fanout-policy-pair/routing_claude_20260829T054236Z.md`
- Proxy logs with the same timestamp in `results/fanout-policy-pair/`

The directory-form trace copies are byte-identical to the flat copies, so they were not double-counted. SHA-256 checks confirmed each duplicate pair.

### What “paired” means here

Both conditions began from the same prompt and ran concurrently, but this is a **task-level pair**, not 50 identical per-call A/B samples. As soon as the first model chose different tools or wording, its future conversation body diverged. Request 20 in one trace is not necessarily comparable to request 20 in the other. The valid comparisons are therefore:

1. task outcome and adherence;
2. aggregate operational metrics;
3. tool reliability by class;
4. fan-out structure and recovery behaviour;
5. factual accuracy of the eventual reports against the checked-out source;
6. concrete divergence points.

### Git diff interpretation

A raw line diff of JSONL is not a semantic comparison because every request ID, timestamp, session ID, message history, and response differs after divergence. The raw command nevertheless shows the trajectories are structurally different:

```text
git diff --no-index --stat cloud_trace routing_trace
1 file changed, 55 insertions(+), 41 deletions(-)
```

The saved report diff is similarly dominated by missing continuations:

```text
git diff --no-index --stat cloud_report routing_report
1 file changed, 108 insertions(+), 37 deletions(-)
```

The saved outputs have only 1.9% word-sequence similarity. That does not mean the underlying findings disagree by 98.1%; it mainly means the cloud file contains only the tail of its report and the routing file contains a different, longer tail. This report therefore uses a normalized structural comparison rather than interpreting raw JSONL line changes as quality.

### Repository mutation check

The workload was intended to be read-only. The trace contains only `Read`, `Bash`, and `Agent` tool calls. The Bash commands are searches and inspection (`find`, `grep`, `ls`, `wc`); no source-writing command was found. The current Git diff contains no tracked source changes attributable to either run. The downloaded result directory is untracked, as expected. Both conditions respected the folder scope and read-only requirement.

### Important experimental confounds

- The conditions ran concurrently. Wall-clock comparisons include different provider/GPU contention schedules.
- Both proxies had local cache observation enabled. The cloud-only proxy queried the same vLLM cache that the routing run was warming. Its probe is read-only, but cloud-side *predicted local* warmth is contaminated by the concurrent local run and must not be treated as an independent counterfactual.
- The cloud gateway supplied output-token usage but no usable input/cache-token detail. Cloud cache reuse and full cloud cost cannot be measured from this trace.
- The prompt required five concurrent foreground agents, but neither model launched all five in one wave. Each initially emitted three Agent calls, then launched the remaining work later. Maximum overlapping `/v1/messages` intervals was four in both traces; sampled vLLM running requests peaked at three.

## 2. Headline quantitative comparison

| Metric | Cloud condition | Routing/local condition | Interpretation |
| --- | ---: | ---: | --- |
| `/v1/messages` calls | 36 | 50 | Local required more recovery and continuation turns |
| HTTP 200 | 36/36 | 50/50 | Both protocol paths were available |
| Placement | 36 cloud | 50 local | Static routing admitted every local call as `fits` |
| Task wall time | 1,967.1 s (32.8 min) | 2,617.7 s (43.6 min) | Local was 650.6 s / 33.1% longer |
| Output tokens recorded | 29,748 | 84,750 | Local generated 2.85× more; cloud accounting is incomplete on stalled calls |
| Median output tokens/call | 194 | 395 | Qwen was more verbose |
| Median TTFT | 3.23 s | 9.58 s | Overall cloud faster to first token |
| P90 TTFT | 31.61 s | 21.49 s | Cloud tail worse due parent/provider stalls |
| Median total latency | 25.97 s | 34.94 s | Cloud faster in the middle of the distribution |
| P90 total latency | 292.07 s | 173.35 s | Cloud's approximately five-minute stalls dominate its tail |
| Median decode rate | 138.8 tok/s | 28.0 tok/s | Cloud approximately 5× faster |
| Tool uses | 56 | 82 | Local did 46% more tool work |
| Read/Bash | 42/9 | 51/24 | Local explored more but was less efficient |
| Agent attempts | 5 | 7 | Local needed two retries |
| Unique tool errors | 0 | 2 | Both local errors were malformed Agent JSON |
| `max_tokens` stops | 3/36 (8.3%) | 11/50 (22.0%) | Local 4K clamp is binding; cloud also stalled/truncated |
| Saved final output | 3.1 KB | 14.2 KB | Local tail more useful; both incomplete |

The latency table needs nuance. Cloud's typical tool turn was much faster: its tool-use calls had a 11.89-second median total latency versus 25.59 seconds locally. But cloud parent/control-plane calls were extremely slow: its eight 22-tool top-level calls had a 292.07-second median total latency. Local top-level calls had a 141.79-second median. Thus “cloud is faster” is true for ordinary short outputs but false for this run's orchestration tail.

## 3. Local cache analysis

### 3.1 Accounting and probe correctness

Local input accounting balances exactly:

```text
2,051,488 total input tokens
= 1,197,952 cache-read
+   816,928 cache-created
+    36,608 uncached
```

This gives:

- request-level cache hits: **48/50 (96.0%)**;
- token-weighted cache reuse: **58.4%**;
- cache creation: **39.8%** of input;
- uncached remainder outside read/write blocks: **1.8%**.

The live probe was unusually strong in this run:

- 50/50 predictions available;
- 50/50 predicted warm/cold state correct;
- 50/50 predicted cached-token counts exactly equal to actual response accounting;
- 50/50 within the recorded 5% agreement threshold;
- median probe cost 59.7 ms, P90 88.4 ms, maximum 236.9 ms;
- total probe time across all calls approximately 3.43 seconds, negligible against a 43.6-minute task.

This directly validates the `high` to `xhigh` reasoning-effort translation for the request shape used here. There are no `local-probe-unavailable` decisions and no evidence that the probe rendered a different cache prefix than generation.

### 3.2 Cache warmth materially improved TTFT

The relationship between uncached input and local server TTFT is extremely strong. Excluding the 981-token title sidecall, Pearson correlation between `(total input - cache read)` and `server_ttft_ms` is **0.996** over 49 calls.

| Pre-request cached fraction | Calls | Median input | Median cached fraction | Median TTFT | Median total latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Below 25% | 4 | 22,970 | 8.4% | 9.91 s | 21.61 s |
| 25–75% | 29 | 44,506 | 43.8% | 14.52 s | 32.91 s |
| Above 75% | 17 | 40,693 | 91.0% | 2.15 s | 124.86 s |

The warm group's total latency remains high because these were often long report-generation turns. Prefix caching removes prefill cost; it does not accelerate Qwen's roughly 28 tok/s decode.

Concrete same-shape examples make the effect clearer:

- A 55,609-token call with only 17,248 cached tokens (31.0%) had **24.64 s TTFT**.
- A 55,397-token call with 51,744 cached tokens (93.4%) had **2.92 s TTFT**.
- A 67,184-token call with 17,248 cached tokens (25.7%) had **30.61 s TTFT**.
- Its 67,230-token warm continuation reused 64,288 tokens (95.6%) and had **2.42 s TTFT**.

This is exactly the fan-out reuse mechanism the router is intended to exploit. Sibling workers repeatedly shared a base prefix of 17,248 tokens, while within-branch continuations grew to 34,496, 51,744, 53,312, 54,880, 64,288, and 65,856 cached tokens.

### 3.3 Eviction assessment

There is **no evidence of an unexpected prefix eviction** in the local trace:

- only the initial 27,050-token main request and independent 981-token title sidecall were cold;
- all 48 later local calls had a non-zero live-probed prefix;
- every live-probed cached-token count appeared identically in actual usage;
- no warm prediction became an actual miss.

Two resource snapshots reported `kv_cache_usage_pct=0.0` while the same calls actually read 17,248 and 64,288 cached tokens. This proves that the sampled vLLM KV-usage gauge is **not a reliable prefix-residency/eviction indicator** in this setup. It appears to fall to zero when no request is actively holding blocks even though reusable prefix blocks remain discoverable by the live probe. Routing should trust the direct probe for prefix residency, not infer eviction from the telemetry gauge.

The sampled cumulative counters are directionally consistent. Between the first and last snapshots, `prompt_tokens_cached_total` rose by 1,146,208. The trace recorded 1,197,952 cache-read tokens; the 51,744-token difference equals the final call's cache read because its resource snapshot was captured before that request completed.

## 4. GPU and host health

### 4.1 VRAM

Every local request snapshot reported:

- GPU: NVIDIA RTX 6000 Ada Generation;
- total VRAM: 47.988 GiB;
- used VRAM: 44.272 GiB;
- free VRAM: 3.716 GiB;
- used: 92.26%.

This high percentage is expected for vLLM because the server was launched with `--gpu-memory-utilization 0.90` and reserves most of the device for weights and KV-cache. It does not mean the device was continually about to OOM. The useful pressure signals are KV occupancy, queueing, failed allocations, and engine health. Those were healthy.

### 4.2 KV cache and concurrency

- configured KV-cache capacity: 558,571 tokens;
- sampled KV usage: median 10.77%, P90 15.46%, maximum 18.46%;
- peak estimated occupied capacity from the gauge: approximately 103,112 tokens;
- sampled running requests: median two, maximum three;
- sampled waiting requests: zero on all 50 calls;
- vLLM `max_num_seqs`: eight, so sampled concurrency stayed well below the configured sequence limit.

The engine had ample KV capacity during this experiment. Even if the gauge were a complete residency measure—which the zero-gauge/cache-hit examples show it is not—the peak was below one fifth of capacity. There is no basis here for routing cloud because KV is “full.”

### 4.3 Host RAM and telemetry freshness

- host RAM used: 37.40–37.82%, median 37.67%;
- available RAM remained approximately 76.8–77.3 GiB;
- telemetry age: median 525 ms, P90 893 ms, maximum 996 ms.

The one-second sampler was fresh enough to establish absence of sustained queue/memory pressure. The trace does not record GPU SM utilization, temperature, power, or clock rate, so it cannot support an average-utilization claim. Two out-of-band live checks during the run observed 100% GPU utilization, but those are spot checks, not a distribution.

### 4.4 Capacity headroom

The static policy's effective budget was 100,000 × 0.9 = 90,000 input-plus-reserved-output tokens. The largest observed local prompt was 71,377 tokens:

```text
71,377 + current 4,096 clamp = 75,473 (83.9% of budget)
71,377 + proposed 8,192 cap = 79,569 (88.4% of budget)
```

An 8,192 cap would have fit every observed call under the policy's token budget. The run does not prove how much extra decode concurrency or runtime that cap would create, so it should be tested rather than assumed free.

## 5. Quality and divergence analysis

### 5.1 What both models did well

Both conditions:

- stayed inside `edgeproxy/`;
- used only read-only file inspection;
- eventually launched five logically distinct investigations: routing, request handling, cache/trace, telemetry/config/timing/shaping/cost, and tests/testability;
- found the central request flow correctly;
- identified several real risks, including lack of local-to-cloud transport failover, full request buffering, stream accumulation, observe-only cloud cache state, and the importance of policy short-circuit order;
- produced no tracked repository mutation.

This is a non-trivial positive result for local Qwen. It maintained scope over a 43-minute, 50-call trajectory, executed 75 ordinary file/search tools without error, recovered from its own malformed Agent calls, and eventually consolidated all five branches.

### 5.2 First structural divergence: five-way fan-out became waves of three and two

The prompt explicitly required five foreground agents emitted together. Neither model achieved that.

Cloud said:

> “Three agents have returned. I still have two pending... I launched only three in the first...”

It first emitted three Agent calls, then later emitted two. Local also first emitted three. Its next parent turn attempted three calls—one retry plus the remaining two—and later needed another one-agent retry. Maximum overlapping message requests was four in both traces, not five.

The likely harness-level cause is the size of the Agent tool inputs. Each delegation repeats a long constraint/task prompt. Generating five large tool calls in one assistant response stresses output limits before the agents even begin. A cleaner fan-out benchmark should predefine compact agents with `--agents` or drastically shorten each Agent prompt so all five calls fit in one orchestration response.

### 5.3 Local-only divergence: malformed Agent JSON

Cloud Agent execution:

- five attempts;
- five valid tool inputs;
- five returned results;
- zero tool errors.

Local Agent execution:

- seven attempts;
- five valid returned results;
- two malformed inputs represented as `_unparsed`;
- two Claude Code `InputValidationError` results;
- successful recovery through retries.

The trace records the exact failure:

> `<tool_use_error>InputValidationError: Agent was called with input that could not be parsed as JSON.`

One local parent response recognized it directly:

> “the third (cache and trace) failed to parse as JSON”

The second malformed call was the tests/testability delegation. This is a model/tool-decoding quality failure, not a GPU failure. Strict schemas were injected locally, but the parser still surfaced `_unparsed` content for these long Agent arguments. Ordinary tools did not show this problem: all 51 Read and 24 Bash calls succeeded.

Routing implication: the risky unit is not “all tool use.” It is **large, nested Agent-call argument generation on the parent/control plane**. Blanket-cloud routing for every tool-bearing call would discard the strongest local result—the reliable worker tool loop.

### 5.4 Output truncation and recovery

#### Local

Every local request was clamped from the client's 64,000 reservation to 4,096. Eleven responses reached exactly 4,096 output tokens and stopped `max_tokens`. Ten were worker/subagent report turns; one was the parent synthesis.

The consequences are visible in the trace:

> “I was cut off mid-sentence... Let me continue from where I left off”

and later:

> “I was mid-report. I need to continue...”

Claude Code recovered by sending continuation turns, but this increased the local call count, output volume, wall time, and chance that the runner would save only the last fragment.

#### Cloud

Cloud had three `max_tokens` records despite the request retaining `max_tokens=64000`. Two parent synthesis calls lasted approximately 301 seconds and reported zero output tokens even though response text was reconstructed; another parent call lasted approximately 337 seconds with no stop reason. These look more like provider/gateway duration or accounting anomalies than a simple 64K visible-output exhaustion. The trace is insufficient to isolate the cause.

Cloud therefore had fewer truncations, and none in its worker reports, but a worse parent-call latency tail.

### 5.5 Final output assembly failure

The complete parent synthesis is spread across multiple response records.

Cloud parent synthesis fragments:

- fragment 1: 8,711 characters, `max_tokens`;
- fragment 2: 5,306 characters, `max_tokens`;
- fragment 3: 3,106 characters, `end_turn` and completion marker.

The saved cloud file contains only fragment 3 and literally begins:

> `## Open Questions (continued)`

Local parent synthesis fragments:

- fragment 1: 14,462 characters, `max_tokens`;
- fragment 2: 14,528 characters, `end_turn` and completion marker.

The saved routing file contains essentially the second fragment and begins mid-section with the `shaping.py` bullet. It omits the executive summary and routing/request/cache findings from fragment 1.

The current completion check therefore gives a false positive. It validates only that the last continuation is long and contains the marker. The runner must either capture all assistant messages from `--output-format stream-json` and concatenate the parent synthesis turns, or require a final self-contained restatement that fits in one response. Until that is fixed, “report exists” is not a valid end-to-end success criterion.

### 5.6 Detail, efficiency, and usefulness

The reconstructed local report is approximately 29.0K characters / 3,549 words. The reconstructed cloud report is approximately 17.1K characters / 2,261 words. Local is more detailed and includes useful cross-agent overlap, data-flow, and testability analysis. It also repeats itself and performs substantially more search work.

As actually saved:

- routing output: about 14.1K characters / 1,665 words / 110 lines;
- cloud output: about 3.1K characters / 459 words / 39 lines.

For the user's requested deliverable, the local saved output is clearly more useful. However, “more text” is not automatically “better quality,” and its missing first half prevents an outright pass.

### 5.7 Source-grounded factual audit

The reports were checked against the current `edgeproxy/` source and repository state at `953d9ce`.

| Claim | Assessment | Evidence / nuance |
| --- | --- | --- |
| No local→cloud retry after an upstream transport error | Correct in both | `server.py:375-391` records and returns 502. A router exception before send does fall back cloud at `312-314`, which is different. |
| Static policy becomes cloud-only if live local tracking is off | Correct | With no exact `local_prompt_tokens`, `router.py:269-270` returns `local-token-count-unavailable`. Config defaults tracking to off. |
| Full request bodies are stored | Correct | `server.py:333` stores decoded `request_json`, including prompts/tool inputs. This is a privacy consideration. |
| Storing bodies contradicts the docstring's “never stored” statement | Overstated by cloud | The docstring specifically says **credentials** are never stored; headers are redacted. It does not claim prompts are never recorded. |
| Cache trackers are “stateless” | Incorrect cloud executive-summary wording | `CloudCacheTracker` contains mutable `entries`, `seen_lineages`, and scale samples. Local correctly described shared state. |
| No tests / entire codebase untested | Incorrect at repository level; correct only inside allowed subtree | There are no tests under `edgeproxy/`, which was the agents' strict scope. The repository has **13 test files, 1,509 lines**, and `python -m unittest discover -s tests -q` passes **73/73 tests**. Both reports should have said “none found within the inspected directory.” |
| `CloudCacheTracker.prune()` has no caller | Correct within `edgeproxy/` | Expired candidates are removed lazily when probed, but never-reprobed entries may remain. |
| `TraceWriter.write()` performs synchronous file I/O inside async request handling | Correct | `record.py:238-256`, called from async paths in `server.py`. Severity depends on traffic and storage latency. |
| Multi-worker state would be “corrupted” | Directionally valid, wording too strong | Separate uvicorn worker processes would have divergent per-worker cache/session state, not shared-memory corruption. Threaded access would raise different concerns. |
| Telemetry sampler dies permanently on any sampling exception | Overstated by cloud | Host RAM, NVML, and metrics calls are individually caught in `_sample()`. An unexpected exception outside those blocks could end `_run`, but ordinary component failures do not. |
| No request-size limit and full body/stream accumulation | Correct | Body is buffered at `server.py:197`; stream bytes accumulate at `500-520`. |
| Cloud cache state is observe-only | Correct | It is recorded but does not affect `StaticPolicy.decide()`. |
| `minimum_cacheable_tokens` has no caller | Incorrect if asserted broadly | It is called by `cost.py:220`; it is not used by placement or tracker prediction. Local's final open question phrases this more accurately. |
| Duplicate cache TTL constants create drift risk | Correct | TTL definitions exist in both routing and cloud-cache logic. |

On factual precision, the reconstructed local report is slightly better. Cloud is more decisive and concise, but it overstates several risks and makes the clearest false architectural statement (“stateless” trackers). Both overgeneralize the scoped test search.

## 6. Which side produced the better outcome?

There is no single winner across all dimensions.

### Cloud wins

- Agent tool-call validity: 5/5 versus local 5/7 attempts.
- Typical tool-turn latency and decode speed.
- Concision: fewer repeated searches and less output sprawl.
- Worker completion: no cloud worker report stopped `max_tokens`.

### Local wins

- GPU/cache observability and exact cache accounting.
- Delivered saved report usefulness: 14.2 KB versus 3.1 KB.
- Reconstructed depth: more detailed architecture, overlap, and open-question analysis.
- Tail latency in this particular run: local P90 total was lower because cloud had approximately five-minute stalls.
- Cost avoidance potential, though the exact dollar figure is not validated.

### Both fail the literal end-to-end requirement

- neither launched five agents in one simultaneous wave;
- neither saved a self-contained complete final report;
- both made at least one scope-sensitive overclaim;
- the runner's completion marker accepted incomplete continuation tails.

If forced to rank the **actual files delivered to the user**, local is better because it contains substantially more useful material. If ranking **control-plane correctness**, cloud is better because it produced valid Agent calls without retries. If ranking **worker suitability**, local is good enough for this read-only exploration workload once its output cap is raised.

## 7. Cost metrics: why no dollar conclusion is justified

The local trace reports `$5.9757295` in cumulative “saved” cloud cost. This is a counterfactual estimate, not a measured bill:

- 49/50 local calls have confidence `conservative-unconfirmed-cache-state`;
- cloud cache prediction was unknown, so the estimate prices large portions as fresh 5-minute cache creation;
- local GPU operating cost is excluded;
- the actual cloud trace lacks input and cache-detail usage, recording only output cost (`$0.29748`).

Comparing `$5.98 saved` against `$0.30 cloud cost` would therefore be invalid. The cloud number omits input; the local number estimates input under an unconfirmed cache state. The pair establishes token volume and potential avoided work, not realized dollar savings.

Cloud cache reuse is similarly unmeasurable here. The cloud provider/gateway returned no `cache_read_input_tokens` or `cache_creation_input_tokens`, and the tracker remained unknown on 35/36 calls. Do not use cloud-cache state in routing from this run.

## 8. Routing policy derived from this study

Routing should have three layers:

1. **Hard feasibility:** can local safely execute the request at all?
2. **Quality/risk gate:** is this a call class where local failures are likely or expensive?
3. **Preference:** among feasible and acceptable choices, which is faster/cheaper given cache, queue, and link state?

The current static policy implements most of layer 1 but almost none of layers 2–3.

### 8.1 Hard gates to retain

Keep the existing cloud decisions for:

- server-side tools that local cannot execute;
- security-monitor calls;
- unavailable local probe/exact token count;
- prompt plus output reserve above the effective local budget;
- local backend known unhealthy.

Add a failure gate for local transport errors. A local connection/OOM failure currently becomes HTTP 502 with no cloud retry. Safe automatic failover is possible only before response bytes have been emitted. For streaming requests, once local bytes are sent, silently restarting cloud risks duplicated or contradictory output.

### 8.2 Experimentation-first policy

Goal: maximize local exposure and collect quality data, while preventing repeated known failures.

Recommended rules:

```text
if hard feasibility fails:
    cloud
elif security monitor:
    cloud
elif previous turn in this lineage returned malformed tool input:
    cloud for one recovery turn
elif previous local turn ended max_tokens:
    cloud for the continuation OR allow one larger local continuation
else:
    local
```

Local generation settings:

- Raise the cap from 4,096 to **8,192**, always bounded by `effective_budget - exact_prompt_tokens`.
- Keep `temperature=0`, strict tool schemas, and `high -> xhigh` mapping.
- Record tool-schema validity, `_unparsed`, stop reason, recovery count, and final task checker result as explicit quality fields.
- Keep cache probe as ground truth; do not route from `kv_cache_usage_pct` alone.
- If `requests_waiting > 0` persists or vLLM is unhealthy, prefer cloud. This run has no data to calibrate a queue-delay threshold, so any numeric cutoff beyond “waiting exists” must be tested.

Expected benefit: nearly all feasible traffic remains local, preserving the experiment's purpose and maximizing cache reuse. The local model already completed the workload despite two retries.

Trade-offs and concerns:

- An 8K cap can double worst-case decode time and may encourage even more verbosity.
- Malformed tool calls may reach Claude Code before the proxy can react, especially with streaming.
- A recovery-on-next-turn policy pays the failed local latency first.
- This is not appropriate where one wrong tool call can mutate production state.
- Local all-call operation took 33% longer than cloud in this run.

### 8.3 Conservative production-quality policy

Goal: keep most low-risk work local but put orchestration, synthesis, and recovery on the stronger control plane.

Recommended rules:

```text
if hard feasibility fails or security monitor:
    cloud
elif request tool names include TaskOutput, ScheduleWakeup, or Workflow:
    cloud  # top-level Claude Code parent/control-plane signature in this run
elif last tool_result is an Agent InputValidationError or other schema error:
    cloud
elif request is a high-consequence mutation/approval decision:
    cloud
else:
    local  # read-heavy worker/subagent loop
```

Why this boundary is evidence-backed:

- all malformed calls occurred on parent Agent orchestration;
- all 75 ordinary local Read/Bash calls succeeded;
- worker calls had the 19-tool suite; parent calls had 22 tools including `TaskOutput`, `ScheduleWakeup`, and `Workflow`;
- top-level final synthesis is also where incomplete user-facing output is most costly.

Counterfactual local share on the observed local trace:

- seven top-level 22-tool calls cloud;
- 42 worker 19-tool calls local;
- one title sidecall can remain local;
- **43/50 local = 86% local request share**.

Trade-offs and concerns:

- Tool-suite signatures are Claude Code-version-specific. They must be recorded and regression-tested rather than hard-coded invisibly.
- Routing the parent cloud loses some local prefix warming before the first worker. The first local worker may be colder, though later siblings can reuse the common worker prefix.
- Cloud parent calls were extremely slow in this run; quality reliability improves, but tail latency may worsen.
- Cloud cache usage is unavailable, so the cost of parent placement cannot yet be estimated accurately.
- Mixed parent/worker placement changes the trajectory; 86% is a descriptive counterfactual, not a guaranteed live result.

### 8.4 Cache-aware preference within locally acceptable calls

For worker calls that pass the quality gate:

- Strongly prefer local when the direct probe shows >75% reuse and the queue is empty. Warm local TTFT was 2.15 seconds median, faster than the cloud-wide 3.23-second median.
- Treat 25–75% reuse as workload-dependent. Local median TTFT was 14.52 seconds in this band; cloud may be faster for short outputs.
- Cold large-prefill calls should be cloud candidates when latency matters, unless serving them locally creates valuable cache for several known siblings.
- Do not optimize solely on TTFT for long-form reports: decode dominates, and cloud decoded approximately five times faster.

A fan-out-aware router should reason about cohort value:

```text
cost of first cold local worker
minus value of warming the shared 17,248-token prefix
across the remaining sibling workers
```

This trace demonstrates that the shared worker prefix was repeatedly reused, so a cold first local placement can be worthwhile even if that single call would be faster in cloud.

### 8.5 What not to route on yet

- **Raw VRAM percentage:** 92.26% was normal preallocation, not distress.
- **Sampled KV usage alone:** it read zero on calls that provably reused 17K and 64K tokens.
- **Cloud cache prediction:** provider usage was unavailable.
- **HTTP 200:** both malformed Agent calls occurred inside otherwise successful HTTP responses.
- **Valid response/tool envelope alone:** `_unparsed` Agent inputs still produced a tool-shaped response that Claude Code rejected.
- **Cost savings dollar field:** it is an unconfirmed counterfactual here.

## 9. Exact “local is good enough” boundary from this run

### Good enough locally, with high confidence for this task class

- read-only repository exploration;
- file reads, directory listings, grep/find/wc inspection;
- worker/subagent turns with bounded consequences;
- exact prompt below the 90K effective budget;
- live probe available;
- queue empty;
- warm or partially warm prefixes;
- workflows where a retry is acceptable.

Evidence: 75/75 ordinary Read/Bash calls succeeded, 50/50 requests returned HTTP 200, 50/50 cache probes agreed exactly, no waiting/OOM occurred, and all five worker assignments eventually completed.

### Not yet good enough locally without safeguards

- generation of multiple large Agent tool calls in one response;
- final user-facing synthesis constrained to 4,096 tokens;
- irreversible/high-consequence actions where one malformed call is unacceptable;
- workflows requiring guaranteed completion in one turn;
- production failover expectations, because local transport failure currently returns 502;
- decisions based on supposed KV eviction from the sampled gauge.

### Unknown from this pair

- edit/write correctness, because the workload prohibited mutation;
- behavior near KV saturation or with queued sequences;
- actual cloud prompt-cache reuse;
- quality under a larger 8K local output cap;
- whether Agent JSON errors generalize beyond two long delegation prompts;
- RTX 5090 behavior, because this run used an RTX 6000 Ada.

## 10. Recommended next experiments, in order

1. **Fix report capture before another quality claim.** Store every parent assistant synthesis fragment or require a final self-contained report. Validate that the saved file begins with the requested title/executive summary, not just that it ends with a marker.
2. **Rerun with an 8,192 local cap.** Measure whether worker `max_tokens` falls from 10/42 without excessive latency or new memory pressure.
3. **Shorten/predefine the five agent prompts.** Confirm five agents actually launch in one wave. Record peak overlap and worker start times.
4. **Run sequential conditions once.** This removes cross-condition local-probe contamination and shared host/network contention. Keep concurrent mode as a separate realism test.
5. **Add a programmatic task checker.** At minimum: five successful Agent results, zero scope violations, zero writes, required report sections present, and no partial/continued opening.
6. **Record tool-schema validity before execution.** Count `_unparsed` and input-schema errors by tool name and call class.
7. **Test the conservative parent-cloud/worker-local policy live.** Do not infer its outcome only from this trace.
8. **Restore cloud input/cache usage visibility.** Until the gateway returns those fields, cloud caching and cost cannot guide placement.
9. **Stress KV/queue behavior deliberately.** This healthy run never exceeded 18.46% sampled KV usage or zero queued requests, so it cannot calibrate pressure thresholds.
10. **Repeat across multiple tasks and seeds.** Two malformed Agent calls in one trajectory identify a failure mode, not a stable probability.

## Final recommendation

For continued experimentation, keep local-first placement but immediately raise the headroom-aware output cap to 8K and add cloud recovery after malformed tool calls or local truncation. This maximizes exposure to Qwen while turning known failures into measurements rather than terminal outcomes.

For a conservative mixed deployment, route Claude Code parent/control-plane calls and final synthesis cloud, while keeping read-heavy worker branches local. This pair suggests that such a policy can retain roughly 86% local calls while avoiding the exact class where local diverged. Treat that percentage as a hypothesis for the next live run, not a proven production result.

The central result is not “cloud good, local bad.” It is narrower and more actionable: **the local RTX 6000/Qwen stack is healthy and cache-effective, and Qwen is competent at grounded worker exploration; the quality boundary appears at orchestration-tool serialization and long-form synthesis under the 4K cap.** That is the boundary the next router should encode and validate.
