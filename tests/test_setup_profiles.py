import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolved_config(script: str, setup: str | None = None) -> dict[str, str]:
    command = [str(ROOT / script)]
    if setup is not None:
        command.extend(["--setup", setup])
    command.append("--print-config")
    env = os.environ.copy()
    for name in (
        "FLOWMESH_SETUP",
        "FLOWMESH_WORKFLOW",
        "FLOWMESH_SSH_ALIAS",
        "FLOWMESH_TUNNEL_NAME",
        "EXPERIMENT_NAMESPACE",
        "MODEL",
        "TOOL_CALL_PARSER",
        "REASONING_PARSER",
        "QUANTIZATION",
        "LANGUAGE_MODEL_ONLY",
        "MAX_MODEL_LEN",
        "NATIVE_MAX_MODEL_LEN",
        "ATTENTION_BACKEND",
        "KV_CACHE_DTYPE",
        "GPU_MEM_UTIL",
        "VLLM_EXTRA_ARGS",
        "EDGEPROXY_MAX_LOCAL_TOKENS",
        "EDGEPROXY_LOCAL_TOKEN_MARGIN",
        "WORKFLOW",
    ):
        env.pop(name, None)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


class SetupProfileTests(unittest.TestCase):
    def test_default_driver_profile_is_qwen25_7b(self):
        self.assertEqual(
            resolved_config("flowmesh-up.sh"),
            resolved_config("flowmesh-up.sh", "qwen25-7b"),
        )

    def test_qwen25_7b_profile(self):
        config = resolved_config("flowmesh-up.sh", "qwen25-7b")
        self.assertEqual(config["workflow"], "ssh-workflow.yaml")
        self.assertEqual(config["ssh_alias"], "fmbox-qwen25-7b")
        self.assertEqual(config["model"], "Qwen/Qwen2.5-7B-Instruct-AWQ")
        self.assertEqual(config["tool_call_parser"], "hermes")
        self.assertEqual(config["native_max_model_len"], "32768")
        self.assertEqual(config["attention_backend"], "FLASH_ATTN")

    def test_qwen38_27b_profile(self):
        config = resolved_config("flowmesh-up.sh", "qwen38-27b")
        self.assertEqual(config["workflow"], "ssh-workflow-5090.yaml")
        self.assertEqual(config["ssh_alias"], "fmbox-qwen38-27b")
        self.assertEqual(config["model"], "Inferact/Qwen3.8-27B-NVFP4")
        self.assertEqual(config["tool_call_parser"], "qwen3_coder")
        self.assertEqual(config["reasoning_parser"], "qwen3")
        self.assertEqual(config["quantization"], "none")
        self.assertEqual(config["language_model_only"], "1")
        self.assertEqual(config["max_model_len"], "100000")
        self.assertEqual(config["native_max_model_len"], "262144")
        self.assertEqual(config["attention_backend"], "auto")
        self.assertEqual(config["kv_cache_dtype"], "fp8")
        self.assertEqual(config["vllm_extra_args"], "--enforce-eager --max-num-seqs 8")
        self.assertEqual(config["edgeproxy_max_local_tokens"], "100000")
        self.assertEqual(config["edgeproxy_local_token_margin"], "0.90")

    def test_bootstrap_and_driver_resolve_same_model_settings(self):
        shared_keys = {
            "setup",
            "experiment_namespace",
            "model",
            "tool_call_parser",
            "reasoning_parser",
            "quantization",
            "language_model_only",
            "max_model_len",
            "native_max_model_len",
            "attention_backend",
            "kv_cache_dtype",
            "gpu_mem_util",
            "vllm_extra_args",
            "edgeproxy_max_local_tokens",
            "edgeproxy_local_token_margin",
        }
        for setup in ("qwen25-7b", "qwen38-27b"):
            driver = resolved_config("flowmesh-up.sh", setup)
            bootstrap = resolved_config("bootstrap.sh", setup)
            self.assertEqual(
                {key: driver[key] for key in shared_keys},
                {key: bootstrap[key] for key in shared_keys},
            )

    def test_unknown_setup_fails_before_submission(self):
        result = subprocess.run(
            [str(ROOT / "flowmesh-up.sh"), "--setup", "not-a-setup", "--print-config"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown setup", result.stderr)

    def test_5090_workflow_reuses_image_and_selects_rtx_5090(self):
        baseline = (ROOT / "ssh-workflow.yaml").read_text()
        comparison = (ROOT / "ssh-workflow-5090.yaml").read_text()
        image_line = next(
            line.strip() for line in baseline.splitlines() if line.strip().startswith("image:")
        )
        self.assertIn(image_line, comparison)
        self.assertIn("type: RTX 5090", comparison)


if __name__ == "__main__":
    unittest.main()
