import unittest
from types import SimpleNamespace
from unittest.mock import mock_open, patch

from edgeproxy.telemetry import _gpu_snapshot, parse_vllm_metrics, read_host_ram


METRICS = '''
# HELP vllm:kv_cache_usage_perc KV usage
vllm:kv_cache_usage_perc{model_name="local"} 0.25
vllm:num_requests_running{model_name="local"} 2.0
vllm:num_requests_waiting{model_name="local"} 1.0
vllm:cache_config_info{block_size="16",cache_dtype="auto",kv_cache_memory_bytes="None",kv_cache_size_tokens="131072",num_gpu_blocks="8192"} 1.0
vllm:prefix_cache_queries_total{model_name="local",engine="0"} 100
vllm:prefix_cache_queries_total{model_name="local",engine="1"} 50
vllm:prefix_cache_hits_total{model_name="local",engine="0"} 75
vllm:prefix_cache_hits_total{model_name="local",engine="1"} 25
vllm:prompt_tokens_cached_total{model_name="local"} 4096
'''


class TelemetryTests(unittest.TestCase):
    def test_gpu_snapshot_includes_load_clocks_power_temperature_and_events(self):
        class FakeNvml:
            NVML_CLOCK_GRAPHICS = 0
            NVML_CLOCK_SM = 1
            NVML_CLOCK_MEM = 2
            NVML_TEMPERATURE_GPU = 0
            nvmlClocksEventReasonGpuIdle = 0x1
            nvmlClocksEventReasonSwPowerCap = 0x4
            nvmlClocksThrottleReasonHwThermalSlowdown = 0x40

            @staticmethod
            def nvmlDeviceGetMemoryInfo(_gpu):
                return SimpleNamespace(total=48 * 1024**3, used=36 * 1024**3, free=12 * 1024**3)

            @staticmethod
            def nvmlDeviceGetName(_gpu):
                return b"Test GPU"

            @staticmethod
            def nvmlDeviceGetUtilizationRates(_gpu):
                return SimpleNamespace(gpu=97, memory=61)

            @staticmethod
            def nvmlDeviceGetClockInfo(_gpu, clock_type):
                return {0: 2415, 1: 2385, 2: 9001}[clock_type]

            @staticmethod
            def nvmlDeviceGetPowerUsage(_gpu):
                return 287_654

            @staticmethod
            def nvmlDeviceGetEnforcedPowerLimit(_gpu):
                return 300_000

            @staticmethod
            def nvmlDeviceGetTemperature(_gpu, _sensor):
                return 76

            @staticmethod
            def nvmlDeviceGetCurrentClocksEventReasons(_gpu):
                return 0x4 | 0x40

        parsed = _gpu_snapshot(FakeNvml, object(), 2)

        self.assertEqual(parsed["index"], 2)
        self.assertEqual(parsed["name"], "Test GPU")
        self.assertEqual(parsed["vram_used_pct"], 75.0)
        self.assertEqual(parsed["sm_utilization_pct"], 97)
        self.assertEqual(parsed["memory_controller_utilization_pct"], 61)
        self.assertEqual(parsed["graphics_clock_mhz"], 2415)
        self.assertEqual(parsed["sm_clock_mhz"], 2385)
        self.assertEqual(parsed["memory_clock_mhz"], 9001)
        self.assertEqual(parsed["power_draw_w"], 287.654)
        self.assertEqual(parsed["power_limit_w"], 300.0)
        self.assertEqual(parsed["temperature_c"], 76)
        self.assertEqual(parsed["clock_event_reasons_mask"], 0x44)
        self.assertEqual(
            parsed["clock_event_reasons"],
            ["software_power_cap", "hardware_thermal_slowdown"],
        )

    def test_gpu_snapshot_keeps_stable_nulls_for_unsupported_nvml_fields(self):
        class MinimalNvml:
            @staticmethod
            def nvmlDeviceGetMemoryInfo(_gpu):
                return SimpleNamespace(total=16 * 1024**3, used=8 * 1024**3, free=8 * 1024**3)

            @staticmethod
            def nvmlDeviceGetName(_gpu):
                return "Minimal GPU"

        parsed = _gpu_snapshot(MinimalNvml, object(), 0)

        self.assertEqual(parsed["vram_used_pct"], 50.0)
        self.assertIsNone(parsed["sm_utilization_pct"])
        self.assertIsNone(parsed["memory_controller_utilization_pct"])
        self.assertIsNone(parsed["graphics_clock_mhz"])
        self.assertIsNone(parsed["power_draw_w"])
        self.assertIsNone(parsed["temperature_c"])
        self.assertIsNone(parsed["clock_event_reasons"])

    def test_vllm_metrics_include_kv_gib_estimate(self):
        parsed = parse_vllm_metrics(METRICS, kv_bytes_per_token=57_344)

        self.assertEqual(parsed["kv_cache_usage_pct"], 25.0)
        self.assertEqual(parsed["kv_cache_used_tokens_est"], 32_768)
        self.assertEqual(parsed["kv_cache_pool_gib_est"], 7.0)
        self.assertEqual(parsed["kv_cache_used_gib_est"], 1.75)
        self.assertEqual(parsed["requests_running"], 2)
        self.assertEqual(parsed["requests_waiting"], 1)
        self.assertEqual(parsed["prefix_cache_queries_total"], 150)
        self.assertEqual(parsed["prefix_cache_hits_total"], 100)
        self.assertEqual(parsed["prefix_cache_hit_fraction_lifetime"], 0.666667)
        self.assertEqual(parsed["prompt_tokens_cached_total"], 4096)

    def test_missing_prefix_counters_are_unknown_not_zero(self):
        parsed = parse_vllm_metrics("", kv_bytes_per_token=None)

        self.assertIsNone(parsed["prefix_cache_queries_total"])
        self.assertIsNone(parsed["prefix_cache_hits_total"])
        self.assertIsNone(parsed["prefix_cache_hit_fraction_lifetime"])

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
