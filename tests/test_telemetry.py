import unittest
from unittest.mock import mock_open, patch

from edgeproxy.telemetry import parse_vllm_metrics, read_host_ram


METRICS = '''
# HELP vllm:kv_cache_usage_perc KV usage
vllm:kv_cache_usage_perc{model_name="local"} 0.25
vllm:num_requests_running{model_name="local"} 2.0
vllm:num_requests_waiting{model_name="local"} 1.0
vllm:cache_config_info{block_size="16",cache_dtype="auto",kv_cache_memory_bytes="None",kv_cache_size_tokens="131072",num_gpu_blocks="8192"} 1.0
'''


class TelemetryTests(unittest.TestCase):
    def test_vllm_metrics_include_kv_gib_estimate(self):
        parsed = parse_vllm_metrics(METRICS, kv_bytes_per_token=57_344)

        self.assertEqual(parsed["kv_cache_usage_pct"], 25.0)
        self.assertEqual(parsed["kv_cache_used_tokens_est"], 32_768)
        self.assertEqual(parsed["kv_cache_pool_gib_est"], 7.0)
        self.assertEqual(parsed["kv_cache_used_gib_est"], 1.75)
        self.assertEqual(parsed["requests_running"], 2)
        self.assertEqual(parsed["requests_waiting"], 1)

    @patch(
        "builtins.open",
        mock_open(read_data="MemTotal:       33554432 kB\nMemAvailable:   25165824 kB\n"),
    )
    def test_host_ram_uses_mem_available(self):
        parsed = read_host_ram()

        self.assertEqual(parsed["total_gib"], 32.0)
        self.assertEqual(parsed["used_gib"], 8.0)
        self.assertEqual(parsed["used_pct"], 25.0)


if __name__ == "__main__":
    unittest.main()
