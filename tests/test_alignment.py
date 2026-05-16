import unittest

from sglang_group.alignment import dynamic_token_warping, levenshtein_distance


class AlignmentTests(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("abc", "abc"), 0)

    def test_dynamic_token_warping_exact_match(self):
        alignment = dynamic_token_warping(
            ["hello", " world"],
            ["hello", " world"],
            window=1,
        )

        self.assertEqual(alignment.total_cost, 0)
        self.assertEqual(alignment.target_to_draft, ((0,), (1,)))
        self.assertEqual(alignment.draft_to_target, ((0,), (1,)))

    def test_dynamic_token_warping_expands_small_window(self):
        alignment = dynamic_token_warping(["a"], ["", "a"], window=0)

        self.assertEqual(alignment.target_to_draft, ((0,), (0,)))


if __name__ == "__main__":
    unittest.main()
