import unittest
from unittest.mock import patch

from sglang_group.sglang.config import (
    GroupSGLangConfig,
    normalize_draft_backend,
    normalize_group_method,
)


class ConfigTests(unittest.TestCase):
    def test_auto_method_selection(self):
        config = GroupSGLangConfig(method="auto")
        self.assertEqual(
            config.method_for_batch(is_all_greedy=True, max_temperature=0.0),
            "itl-base-slem",
        )
        self.assertEqual(
            config.method_for_batch(is_all_greedy=False, max_temperature=0.6),
            "itl-base-tli",
        )
        self.assertEqual(
            config.method_for_batch(is_all_greedy=False, max_temperature=1.0),
            "itl",
        )

    def test_method_aliases(self):
        self.assertEqual(normalize_group_method("slem"), "itl-base-slem")
        self.assertEqual(normalize_group_method("tli"), "itl-base-tli")
        self.assertEqual(normalize_group_method("token_itl"), "itl")

    def test_draft_backend_aliases(self):
        self.assertEqual(normalize_draft_backend("hf"), "transformers")
        self.assertEqual(normalize_draft_backend("sglang-native"), "sglang")

    def test_default_draft_backend_is_sglang(self):
        config = GroupSGLangConfig.from_env()
        self.assertEqual(config.draft_backend, "sglang")
        self.assertFalse(config.native_draft_kv_cache)

    def test_env_validation(self):
        with patch.dict("os.environ", {"SGLANG_GROUP_METHOD": "bad"}):
            with self.assertRaises(ValueError):
                GroupSGLangConfig.from_env()

    def test_draft_backend_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "SGLANG_GROUP_DRAFT_BACKEND": "sglang",
                "SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS": "4096",
                "SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS": "2",
                "SGLANG_GROUP_ENABLE_NATIVE_DRAFT_KV_CACHE": "1",
            },
        ):
            config = GroupSGLangConfig.from_env()
            self.assertEqual(config.draft_backend, "sglang")
            self.assertEqual(config.native_draft_cache_tokens, 4096)
            self.assertEqual(config.native_draft_max_requests, 2)
            self.assertTrue(config.native_draft_kv_cache)


if __name__ == "__main__":
    unittest.main()
