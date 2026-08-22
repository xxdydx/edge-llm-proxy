import unittest

from scripts.measure_prefix_ttft import detect_match_unit


class MatchUnitDetectionTests(unittest.TestCase):
    def test_prefers_numeric_prefix_match_unit(self):
        self.assertEqual(
            detect_match_unit({"prefix_match_unit": "32", "block_size": "16"}),
            32,
        )

    def test_falls_back_when_prefix_match_unit_is_none_sentinel(self):
        self.assertEqual(
            detect_match_unit({"prefix_match_unit": "None", "block_size": "16"}),
            16,
        )

    def test_falls_back_when_prefix_match_unit_is_invalid(self):
        self.assertEqual(
            detect_match_unit({"prefix_match_unit": "unknown", "block_size": "16"}),
            16,
        )

    def test_returns_none_without_positive_numeric_value(self):
        self.assertIsNone(
            detect_match_unit({"prefix_match_unit": "None", "block_size": "0"})
        )


if __name__ == "__main__":
    unittest.main()
