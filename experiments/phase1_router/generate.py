"""Paired generation: send one MBPP+ problem's prompt to one backend with
deterministic decoding, reusing the hardened
``experiments.capability_router.executor`` (identity allowlist, bounded
fail-fast: 2 attempts / 90 s read-inactivity / 10 s connect / 2 s backoff,
capacity precheck, per-attempt latency accounting).
"""

from __future__ import annotations

from experiments.capability_router import executor
from experiments.capability_router.schema import TeacherCall

from .config import CLOUD_DEEPSEEK, LOCAL_27B, Phase1Config
from .prompting import build_prompt, extract_code
from .schema import GenerationOutcome, Problem

BACKENDS = {"local_27b": LOCAL_27B, "cloud_deepseek": CLOUD_DEEPSEEK}


def _request_for(p: Problem, cfg: Phase1Config) -> dict:
    """Identical bytes for both backends. `_sanitize_for_backend` later swaps
    the `model` field per backend.

    `chat_template_kwargs.enable_thinking=false` disables Qwen3.8's `<think>`
    block: with thinking ON, non-trivial MBPP problems make the local model
    spend the ENTIRE output budget reasoning and never emit code (2/3 of the
    first smoke was NO_CODE for this reason; harder ones take >4 min). Direct
    code generation is also the realistic edge-routing scenario. Verified a
    no-op for DeepSeek-V4-Flash through the Lumid gateway (identical output),
    so the arms stay comparable.
    """
    return {
        "model": "claude-sonnet-5",  # replaced per-backend by the executor
        "max_tokens": cfg.max_output_tokens,
        "temperature": cfg.temperature,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": build_prompt(p)}],
        "stream": True,
    }


def _as_teacher_call(p: Problem, cfg: Phase1Config) -> TeacherCall:
    return TeacherCall(
        call_id=p.task_id.replace("/", "_"),
        task_group=str(p.task_group),
        source_campaign="phase1-mbppplus",
        source_trace_path=str(cfg.dataset_gz),
        record_index=p.order,
        request=_request_for(p, cfg),
        teacher_response={"content": []},  # unused here; grading is objective
        teacher_placement="n/a",
    )


def reference_input_tokens(p: Problem, cfg: Phase1Config) -> tuple[int, str]:
    """One reference input-token count for this task's (identical) prompt,
    used to make local and cloud comparable (cloud never reports input
    tokens). Prefer the local vLLM tokenizer via the patched count endpoint;
    fall back to the deterministic character estimate, explicitly labelled."""
    req = _request_for(p, cfg)
    # the count endpoint validates `model` against the served name -> use the
    # local served model id, not the placeholder that _sanitize_for_backend
    # swaps in at generation time.
    probe_req = {**req, "model": LOCAL_27B.request_model}
    exact = executor._probe_input_tokens(LOCAL_27B, probe_req)
    if exact is not None:
        return int(exact), "vllm_tokenizer"
    return int(len(req["messages"][0]["content"]) / 4), "prompt_est_tokens"


def generate_one(p: Problem, backend_name: str, cfg: Phase1Config) -> GenerationOutcome:
    backend = BACKENDS[backend_name]
    outcome, transform = executor.replay_call(_as_teacher_call(p, cfg), backend)
    resp = outcome.response or {}
    raw_text = _text_of(resp)
    completion = extract_code(raw_text, p.entry_point) if outcome.status == "OK" else None
    return GenerationOutcome(
        backend=backend_name,
        status=outcome.status,
        completion=completion,
        raw_text=raw_text if outcome.status == "OK" else None,
        usage=resp.get("_usage") if resp else None,
        detail=outcome.detail,
        latency_s=outcome.latency_s,
        retry_attempts=outcome.retry_attempts,
        end_to_end_wall_s=outcome.end_to_end_wall_s,
        final_attempt_latency_s=outcome.final_attempt_latency_s,
        attempts_meta=outcome.attempts_meta,
        stop_reason=resp.get("stop_reason"),
        model_identity=resp.get("model"),
    )


def _text_of(response: dict) -> str | None:
    blocks = response.get("content") if isinstance(response, dict) else None
    if not isinstance(blocks, list):
        return None
    parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    return "".join(parts) if parts else None
