import tempfile
import unittest
from pathlib import Path

from scripts.measure_local_throughput import exact_prompt_tokens, load_prompts, percentile


class FakeTokenizer:
    all_special_ids = [0]

    def __len__(self):
        return 1000

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [10 + (ord(character) % 100) for character in text]


class LocalThroughputHelpersTests(unittest.TestCase):
    def test_load_prompts_ignores_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("# note\n\nfirst\tWrite a guide.\nsecond\tAnalyze this.\n")
            self.assertEqual(
                load_prompts(path),
                [("first", "Write a guide."), ("second", "Analyze this.")],
            )

    def test_load_prompts_requires_tab_separator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("missing separator\n")
            with self.assertRaises(ValueError):
                load_prompts(path)

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([10.0, 20.0], 0.9), 19.0)

    def test_exact_prompt_length_and_unique_first_cache_block(self):
        tokenizer = FakeTokenizer()
        first = exact_prompt_tokens(tokenizer, "repeat me", 128, 123, 0)
        second = exact_prompt_tokens(tokenizer, "repeat me", 128, 123, 1)
        self.assertEqual(len(first), 128)
        self.assertEqual(len(second), 128)
        self.assertNotEqual(first[0], second[0])


if __name__ == "__main__":
    unittest.main()
