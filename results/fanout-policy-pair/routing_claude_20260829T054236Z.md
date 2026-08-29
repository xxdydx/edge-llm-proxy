

- **shaping.py (126):** `LinkShaper` (shaping.py:39-70) is the experiment's independent variable — applied only to cloud (server.py:373); `active` (shaping.py:46-48) is `delay_ms>0 or bandwidth_mbps>0`, so a jitter-only preset would be treated as inactive (latent gap, no such preset exists); `from_preset` raises `SystemExit` (shaping.py:42) on unknown preset. `LinkMonitor` (shaping.py:81-126) is a passive EWMA RTT estimate the router may observe; no lock (shaping.py:89-94).
- **cost.py (296):** Anthropic list-price accounting. `pricing_for_model` (cost.py:37-71) maps model name→pricing by substring (ordering load-bearing); `_actual_cloud_partitions` (cost.py:150-179) and `_estimated_local_partitions` (cost.py:190-236) implement mixed-TTL billing (read A, 1h write B−A, 5m write C−B, uncached after C); `build_cost_savings` (cost.py:239-296) is the entry point; `running_saved_usd` left `None` and filled atomically by `TraceWriter` (cost.py:127-128, record.py:245-250). Pure functions, no shared state.
- **Cross-module:** config is the hub; timing and shaping are the two halves of the latency story (both land in the record's `link`/`timing` blocks); cost depends on `cloud_cache`, not on telemetry/timing/shaping. **Inconsistencies:** the single-event-loop assumption is shared but unstated (telemetry + LinkMonitor); `net_ms` is None-aware but `shaped_ms` is not (a 0.0 shaped wait becomes `None` in the trace, server.py:408); `CACHE_TTLS` is duplicated in `cloud_cache.py:22` and `router.py:98`; `resource_sample_interval_s` floor (config.py:161) vs. telemetry `timeout=2.0` (telemetry.py:243) can let a slow `/metrics` overrun the next sample.

### 2.5 Tests and Testability

- **Test inventory: zero.** No `test_*.py`, `*_test.py`, `conftest.py`, `pytest.ini`, `tox.ini`, `tests/` dir, or any file importing pytest/unittest. No CI. No dependency manifest (`pyproject.toml`/`requirements.txt`/`setup.cfg`) in the tree.
- **Most testable (pure, no mocking):** `router.extract_features` + `StaticPolicy.decide` (router.py:147-280); `cost.pricing_for_model` + `build_cost_savings` + helpers (cost.py:37-296); `cloud_cache.prefix_chain`/`prompt_elements`/`cache_scope`/`CloudCacheTracker` (time-injected via `now` param, cloud_cache.py:284-574); `trace.record.reassemble`/`SSEDecoder`/`build_token_accounting`/`redact_headers` (record.py:29-195); `telemetry.parse_vllm_metrics` (telemetry.py:38-122); `shaping.LinkShaper`/`LinkMonitor.observe` (shaping.py:39-109); `timing.ConnTiming` (timing.py:33-48); `config.Config.backends` (config.py:40-47); `local_cache.local_cache_trace` (local_cache.py:88-155); `trace.replay`/`trace.inspect` helpers (pure over record lists).
- **Hardest to test:** `server.proxy` handler (server.py:191-605 — real httpx send, `time.monotonic`/`time.time`, `uuid.uuid4`, shared `last_seen`, disk writes); `telemetry.LocalResourceSampler` (NVML + `/proc/meminfo` not injectable); `trace.record.TraceWriter` (file + clock + lock); `config.parse_args` (reads `os.environ` inline).
- **Suggested strategy:** Layer 1 unit tests for the pure core (highest ROI); Layer 2 integration test via `make_app(cfg)` with a mocked upstream (inject a fake `httpx.AsyncClient` into `app.state.clients`, server.py:147-155, or `httpx.MockTransport`/`respx`); Layer 3 use `trace.replay`/`trace.inspect` as a regression harness (but **no `traces/*.jsonl` fixtures are committed**); Layer 4 clock isolation (most modules already inject or can monkeypatch the clock; `CloudCacheTracker` needs none).
- **Highest-risk untested logic:** cost partitioning (silent mis-partition → wrong `request_saved_usd` and cumulative `running_saved_usd`), cache-prediction state transitions, routing short-circuit order, SSE reassembly.

## 3. Architecture and Request Data Flow

**Components:** `server.py` (FastAPI app + catch-all handler + orchestration) · `router.py` (placement policy engine) · `config.py` (env-driven `Config`) · `cloud_cache.py` + `local_cache.py` (cache prediction/observation) · `telemetry.py` (background resource sampler) · `timing.py` (httpx trace extension) · `shaping.py` (link shaper + monitor) · `cost.py` (list-price accounting) · `trace/record.py` (JSONL writer) · `trace/replay.py` + `trace/inspect.py` (offline analysis).

**Request data flow (live path):**
1. Request arrives at the single catch-all handler (server.py:191-195); body buffered (server.py:197).
2. If path is `v1/messages` (server.py:227): `router.extract_features` (server.py:236) builds `CallFeatures`.
3. If `local_cache_tracking=="observe"`: local probe populates `local_prompt_tokens`, `local_cache_state`, confidence (server.py:237-262).
4. If `cloud_cache_tracking=="observe"`: `cache_scope` + `prefix_chain` + `cloud_tracker.probe` populate `cloud_cache_*` fields (server.py:263-281).
5. `policy.decide(features)` → placement (server.py:289).
6. Local-only rewrites applied to the local body (server.py:292-311); cloud body left untouched.
7. Backend selected via `cfg.backends[placement]` (server.py:330, 361); forwarded via shared `httpx.AsyncClient` (server.py:363-376).
8. Response relayed — streaming (SSE tee, server.py:494-605) or non-streaming (server.py:448-492).
9. `observe_cloud_usage` updates the tracker (server.py:415-467 / 513-565); `build_cost_savings` called (server.py:353, 479, 588).
10. `writer.write(record)` → append-only JSONL (server.py:383/486/598).

**Offline path:** `inspect` reads the JSONL to characterize traffic; `replay` re-runs recorded requests through candidate policies (no GPU, no network) and reports a confusion matrix + cloud-cache reconstruction.

**Key architectural facts:** placement is a base-URL choice (both backends speak Anthropic API, so nothing downstream needs to know which was picked); observability is strictly side-effect-free (every cache/trace path is wrapped so it "must never break a session"); the cloud cache is observe-only today (only the local probe gates placement); the resource sampler runs out-of-band so routing never blocks on NVML/vLLM metrics.

## 4. Risks

**Concurrency / shared state (raised by 3 agents — the dominant theme):**
- `cloud_tracker` (server.py:119), `LinkMonitor` (server.py:135), and `last_seen` (server.py:140) are shared mutable state with **no lock**. Safe only under a single asyncio event loop; `main()` uses a single `uvicorn.run` (server.py:618), so it holds today, but the assumption is unenforced and undocumented — a move to `--workers`/`--threads` would silently corrupt them.
- `LocalResourceSampler` (telemetry.py:158-251) and `LinkMonitor` (shaping.py:89-94) share the same unstated single-loop assumption.

**Resource / growth:**
- `prune` is never called (cloud_cache.py:397); un-reprobed prefix entries accumulate forever. `seen_lineages` unbounded (cloud_cache.py:378). `last_seen` unbounded (server.py:140). `token_scale_samples` bounded per key but key count unbounded.

**Blocking / performance:**
- `writer.write` does synchronous open/write/close under a `threading.Lock` and is called from the async handler (record.py:238-256, server.py:383/486/598) — blocks the event loop; a new file handle opened per write.
- Local probe adds up to 5s to every request when `local_cache_tracking=="observe"` (local_cache.py:54, server.py:244).
- `_load_running_total` reads the whole day file on each rollover (record.py:217-236).

**Correctness / logic:**
- Cost partitioning: mixed-TTL billing with several `None`-returning guard branches (cost.py:168-176, 197-203); a silent mis-partition produces wrong `request_saved_usd` and cumulative `running_saved_usd`. `pricing_for_model` substring ordering is load-bearing and fragile (cost.py:44-70).
- Cache prediction: `observe_cloud_usage` is long and branchy (cloud_cache.py:482-552); confirmed-read-without-entry gap (cloud_cache.py:516-526) may fail to seed a warm entry.
- Routing: `StaticPolicy.decide` short-circuit order is load-bearing (router.py:252-280); `has_server_tools` heuristic conflates server-tool with malformed client-tool (router.py:167-169).
- SSE reassembly: `reassemble` parses `input_json_delta` only at the end (record.py:186-191); `SSEDecoder` must handle chunk boundaries splitting a JSON payload.

**Security / robustness:**
- No auth on the proxy; it relays `x-api-key`/`authorization` as headers (server.py:198) and trusts any client.
- No request-size limit; the full body is buffered (server.py:197).
- `from_preset` raises `SystemExit` on an unknown preset (shaping.py:42) — a hard process exit, not a clean error.

**Config / behavior surprises:**
- Default `local_cache_tracking="off"` makes the live static policy effectively cloud-only (router.py:269-270).
- `model` never used in placement (assumes local backend serves all model names via rewrite).
- `clamp_max_tokens`/`local_can_tool_call` not exposed via CLI (only set by `trace/replay.py`).
- No local→cloud failover on transport error (record-and-502, server.py:377-391).
- `CACHE_TTLS` duplicated in `cloud_cache.py:22` and `router.py:98` (drift risk).
- `shaped_ms` None-vs-zero inconsistency (server.py:408).

## 5. Testing Gaps

- **Zero tests, zero CI, no dependency manifest** — the entire codebase is untested.
- **No committed trace fixtures** — `traces/` is absent/gitignored, so the codebase's own regression tooling (`trace.replay`/`trace.inspect`) has no data to run against; a suite must commit a synthetic/redacted fixture set or generate records.
- **`server.py` is testable only at the app level** — the ~400-line `proxy` handler (server.py:191-605) couples routing, rewriting, forwarding, and record-building; extracting pure helpers (as was done for `_apply_local_generation_controls`, server.py:74-108) would raise testability.
- **`telemetry.LocalResourceSampler` is not injectable** — `pynvml` (telemetry.py:200) and `/proc/meminfo` (telemetry.py:142) are read at the import/OS boundary; testing without a GPU requires a seam.
- **`config.parse_args` reads `os.environ` inline** (config.py:51) rather than taking an env mapping, forcing tests to manipulate the real environment.
- **Highest-risk untested logic (priority order):** cost partitioning → cache-prediction state transitions → routing short-circuit order → SSE reassembly → token accounting (`build_token_accounting`, record.py:43-86) → `TraceWriter` running-total (record.py:217-258).

## 6. Disagreements and Overlaps Between Agents

**Overlaps (same concern surfaced by multiple agents — high confidence):**
- **Shared unguarded mutable state / single-event-loop assumption** — raised by request-handling (`cloud_tracker`, `monitor`, `last_seen`), cache/trace (`entries`, `seen_lineages`, `last_seen`, blocking `writer.write`), and telemetry (`LocalResourceSampler`, `LinkMonitor`). The strongest cross-agent signal.
- **Blocking `writer.write` in the async path** — raised by both request-handling and cache/trace.
- **Cloud cache is observe-only** — raised by both cache/trace and telemetry.
- **`model` never used in placement** — raised by routing and echoed by request-handling.
- **Cost math is the highest-risk area** — raised by both the cost agent and the tests agent.
- **`trace.replay`/`trace.inspect` as a natural regression harness** — raised by both cache/trace and tests.
- **No auth / no request-size limit** — raised by request-handling (and implied by tests' "hardest to test" list).

**Nuances / framing differences (not true contradictions):**
- **`cloud_tracker` safety:** request-handling framed it as "mostly safe between awaits in a single-threaded loop" (low risk, dict ops atomic in CPython), while cache/trace flagged the shared mutable state as a latent concern. Consistent, but different severity framing.
- **Confirmed-read-without-entry gap (cloud_cache.py:516-526):** cache/trace flagged it as a possible bug; it remains an open question (bug vs. intentional conservatism).
- **"By design" vs. "concern":** several items (no local→cloud failover, conservative-by-default seeding, `last_seen` leak) were described as intentional by one agent and flagged as a concern by another — all plausibly by-design for an edge client, but unconfirmed.

**No direct contradictions** were found between agents on any factual claim; the differences are in severity framing and whether a behavior is "by design."

## 7. Open Questions

- **Is the default `local_cache_tracking="off"` intentional?** It makes the live static policy behave like `CloudOnly` (router.py:269-270). Confirm whether production sets `--local-cache-tracking observe`.
- **Why is `model` never used in placement?** Confirm the local backend is expected to serve all requested model names via the `local_model_name` rewrite (server.py:298-300), or whether per-model routing is a planned gap.
- **Is the deployment single-worker/single-loop?** The shared unguarded state (`cloud_tracker`, `monitor`, `last_seen`, sampler) is safe only under one asyncio loop; `main()` uses a single `uvicorn.run` (server.py:618) but the assumption is unenforced.
- **Is `prune` intended to run on a schedule**, or is lazy expiry in `probe` the only mechanism? (It is never called.)
- **Is the blocking `writer.write` acceptable** for the async event loop, or should it be offloaded?
- **Will the cloud cache state ever feed placement**, or is it permanently observe-only? (router.py:52-53 implies "later".)
- **Is the confirmed-read-without-entry gap (cloud_cache.py:516-526) a bug or intentional conservatism?**
- **Is `minimum_cacheable_tokens` deliberately excluded from the prediction path** (used only in cost.py:220), so the tracker may predict "warm" below the provider's minimum cacheable threshold?
- **Is the duplicated `CACHE_TTLS` (cloud_cache.py:22 vs. router.py:98) intentional or a refactor leftover?**
- **Is `probe_local_cache`'s 5s timeout (local_cache.py:54) intended to be hardcoded** rather than configurable via `Config`?
- **Is the local-path cost partition understatement material** when scaled breakpoint depths overlap the read region (cost.py:227-235)?
- **Is the project managed outside this directory** (no dependency manifest, no committed trace fixtures)?

<!-- FANOUT_REPORT_COMPLETE -->
