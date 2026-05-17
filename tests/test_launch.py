import unittest

from sglang_group.cli.launch import (
    _consume_group_args,
    _ensure_legacy_ngram_flags,
    _rewrite_algorithm,
    _uses_sglang_group,
)


class LaunchTests(unittest.TestCase):
    def test_detects_sglang_group_algorithm(self):
        argv = ["--model-path", "target", "--speculative-algorithm", "SGLANG_GROUP"]
        self.assertTrue(_uses_sglang_group(argv))

    def test_rewrites_for_sglang_059(self):
        argv = ["--speculative-algorithm=SGLANG_GROUP"]
        self.assertEqual(_rewrite_algorithm(argv), ["--speculative-algorithm=NGRAM"])

    def test_consumes_group_flags_into_environment(self):
        environ = {}
        argv = [
            "--model-path",
            "target",
            "--sglang-group-method",
            "itl",
            "--sglang-group-auto-high-temp-threshold=0.95",
            "--sglang-group-draft-backend",
            "sglang",
            "--sglang-group-native-draft-quantization=awq",
            "--sglang-group-native-draft-cache-tokens",
            "8192",
            "--sglang-group-native-draft-max-requests=2",
            "--no-sglang-group-draft-cache",
        ]

        remaining = _consume_group_args(argv, environ=environ)

        self.assertEqual(remaining, ["--model-path", "target"])
        self.assertEqual(environ["SGLANG_GROUP_METHOD"], "itl")
        self.assertEqual(environ["SGLANG_GROUP_AUTO_HIGH_TEMP_THRESHOLD"], "0.95")
        self.assertEqual(environ["SGLANG_GROUP_DRAFT_BACKEND"], "sglang")
        self.assertEqual(environ["SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION"], "awq")
        self.assertEqual(environ["SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS"], "8192")
        self.assertEqual(environ["SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS"], "2")
        self.assertEqual(environ["SGLANG_GROUP_ENABLE_DRAFT_CACHE"], "false")

    def test_adds_legacy_ngram_flags(self):
        argv = ["--speculative-algorithm=NGRAM"]
        rewritten = _ensure_legacy_ngram_flags(argv)

        self.assertIn("--speculative-ngram-max-bfs-breadth", rewritten)
        self.assertIn("--disable-cuda-graph", rewritten)
        self.assertIn("--disable-overlap-schedule", rewritten)


if __name__ == "__main__":
    unittest.main()
