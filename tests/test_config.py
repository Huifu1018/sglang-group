import unittest
from unittest.mock import patch

from sglang_group.sglang.config import GroupSGLangConfig, normalize_group_method


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

    def test_env_validation(self):
        with patch.dict("os.environ", {"SGLANG_GROUP_METHOD": "bad"}):
            with self.assertRaises(ValueError):
                GroupSGLangConfig.from_env()


if __name__ == "__main__":
    unittest.main()
