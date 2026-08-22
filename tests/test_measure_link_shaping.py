import unittest

from scripts.measure_link_shaping import corrected_timing, summarize


class CorrectedTimingTests(unittest.TestCase):
    def test_adds_shaped_delay_to_effective_network(self):
        record = {
            "timing": {"ttft_ms": 210.0, "network_ms": 30.0},
            "link": {"shaped_ms": 80.0},
        }
        self.assertEqual(
            corrected_timing(record),
            {"effective_network_ms": 110.0, "corrected_server_ttft_ms": 100.0},
        )

    def test_treats_unshaped_record_as_zero_delay(self):
        record = {
            "timing": {"ttft_ms": 130.0, "network_ms": 30.0},
            "link": {"shaped_ms": None},
        }
        self.assertEqual(
            corrected_timing(record),
            {"effective_network_ms": 30.0, "corrected_server_ttft_ms": 100.0},
        )


class SummaryTests(unittest.TestCase):
    def test_reports_raw_leak_and_corrected_stability(self):
        rows = []
        for delay, ttft, raw_server, effective, corrected in (
            (0.0, 130.0, 100.0, 30.0, 100.0),
            (80.0, 210.0, 180.0, 110.0, 100.0),
        ):
            rows.append(
                {
                    "configured_delay_ms": delay,
                    "valid": True,
                    "shaped_ms": delay,
                    "ttft_ms": ttft,
                    "total_ms": ttft + 10,
                    "transport_network_ms": 30.0,
                    "effective_network_ms": effective,
                    "raw_server_ttft_ms": raw_server,
                    "corrected_server_ttft_ms": corrected,
                    "response_headers_ms": 10.0,
                }
            )
        shaped = summarize(rows, tolerance_ms=5.0)[1]
        self.assertEqual(shaped["delta_ttft_vs_zero_ms"], 80.0)
        self.assertEqual(shaped["delta_raw_server_vs_zero_ms"], 80.0)
        self.assertEqual(shaped["delta_corrected_server_vs_zero_ms"], 0.0)
        self.assertTrue(shaped["ttft_tracks_shaping"])
        self.assertTrue(shaped["corrected_server_stable"])


if __name__ == "__main__":
    unittest.main()
