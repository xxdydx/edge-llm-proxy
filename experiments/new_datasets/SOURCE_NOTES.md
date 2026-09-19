# Source notes for the implementing agents

Prepared 2026-09-17. Recheck installed versions, official source revisions and live endpoint mappings. Numbers and collection limits in the master prompt are proposed design targets, not published guarantees of model quality.

## User-provided project evidence

The user's handoff specifies Qwen 27B served locally, DeepSeek Flash via the configured cloud provider, and per-request routing within continuing Claude Code runs. The supplied legacy JSONL example contains original and separately replayed responses/timing, partial usage, and a tool observation reporting missing pytest despite `is_error: false`. Historical request-only replay does not establish verified execution branches. Inspect original project artifacts; do not treat this packet as a new raw trace.

## Primary references

1. Claude Code programmatic execution and stream output:
   https://code.claude.com/docs/en/headless
   Stream-JSON lines are events. Verify the pinned CLI/SDK settings, isolation, session and streaming semantics instead of counting output lines as calls.

2. Claude Code checkpoint limitations:
   https://code.claude.com/docs/en/checkpointing
   Ordinary rewind/checkpointing is not a complete environment snapshot; Bash-created changes and other state have limitations. Prove a pre-call restore contract independently.

3. Docker commit:
   https://docs.docker.com/reference/cli/docker/container/commit/
   A committed image excludes mounted-volume contents. Do not mistake image creation for a clone of all filesystem mounts or live harness memory.

4. SWE-Gym task environments and evaluator references:
   https://github.com/SWE-Gym/SWE-Gym
   https://huggingface.co/datasets/SWE-Gym/SWE-Gym
   https://huggingface.co/datasets/SWE-Gym/SWE-Gym-Lite
   Reuse task/environment/evaluator definitions, while keeping Claude Code as the experimental agent harness. Pin revisions; do not infer exact data size from an older README.

5. SWE-smith task and evaluation semantics:
   https://huggingface.co/datasets/SWE-bench/SWE-smith
   https://swesmith.com/guides/harnesses/
   The task patch introduces a bug, while a generated prediction patch repairs it. Source-specific adapters must preserve this distinction.

6. Reserved final evaluation source:
   https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
   Reserve untouched tasks and check overlap with development sources before any collection.

7. vLLM metrics:
   https://docs.vllm.ai/en/latest/design/metrics/
   Discover metrics supported by the actual serving version. Establish units, aggregation scope and request attribution instead of inventing unavailable per-request telemetry.

8. Optional public trajectories, not default target-pair labels:
   https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories
   Source model/harness outcomes remain source-specific and must not be renamed as outcomes from the user's Qwen/DeepSeek pair.

## Citation boundary

The branch sampling design, proposed campaign sizes, schemas, judge rubric, continuation policy and work allocation are recommendations in this packet. They are not claimed as tested features of the user's current code or as prescribed by the cited projects.

## Locked task-source update (17 September 2026)

The following source selection is a new design recommendation, not part of the user's historical experiment results. Web source inspection supports the dataset availability/metadata and environment semantics; task quotas and runtime caps are proposed experimental constraints.

- SWE-Gym release: https://huggingface.co/datasets/SWE-Gym/SWE-Gym
- Pinned Gym commit: https://huggingface.co/datasets/SWE-Gym/SWE-Gym/commit/bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb
- Lite release and visible startup rows: https://huggingface.co/datasets/SWE-Gym/SWE-Gym-Lite
- Pinned Lite commit: https://huggingface.co/datasets/SWE-Gym/SWE-Gym-Lite/commit/f70b1a29ab120eb0a0ee7a1deb029825e735b2b0
- SWE-Gym task setup/available images: https://github.com/SWE-Gym/SWE-Gym
- SWE-Gym environment/evaluation fork: https://github.com/SWE-Gym/SWE-Bench-Fork
- Official dataset transition notice: https://huggingface.co/datasets/SWE-bench/SWE-smith/commit/ea6d7173829c7ec8fa16c22055699ff2e9188091
- Python-specific tasks: https://huggingface.co/datasets/SWE-bench/SWE-smith-py
- Pinned Python release: https://huggingface.co/datasets/SWE-bench/SWE-smith-py/commit/77cab9055d42ab4a5c25c89a8f937096db13558e
- Official Python environment profiles: https://github.com/SWE-bench/SWE-smith/blob/main/swesmith/profiles/python.py
- Bug-patch versus repair-patch semantics: https://swesmith.com/guides/harnesses/
- Final-evaluation reservation source: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
- Pinned Verified commit: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/commit/c104f840cc67f8b6eec6f759ebc8b2693d585d4a

Public web inspection did not include full source parquet download. The included script is the deterministic mechanism for enumerating the real task IDs on the user's machine. Do not present synthetic test fixtures or fixed target quotas as verified real-data row counts. Source-profile existence does not prove every task is runnable or fast. The collector must still pin its adapter/harness commits and certify each task environment.
