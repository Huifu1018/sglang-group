import unittest

from sglang_group.sglang.candidates import build_linear_candidate_rows


class CandidateRowTests(unittest.TestCase):
    def test_builds_equal_width_rows(self):
        rows = build_linear_candidate_rows(
            [10, 20],
            [[11, 12, 13], [21]],
            max_draft_token_num=4,
        )
        self.assertEqual(rows.draft_token_num, 2)
        self.assertEqual(rows.rows, ((10, 11), (20, 21)))
        self.assertEqual(rows.proposed_target_tokens, 4)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            build_linear_candidate_rows([1], [[2]], max_draft_token_num=2, draft_prob_rows=[])

    def test_preserves_cache_metadata(self):
        rows = build_linear_candidate_rows(
            [10],
            [[11]],
            max_draft_token_num=2,
            proposal_cache_events=["hit"],
            draft_cache_events=["rebuild"],
            proposal_methods=["itl"],
        )

        self.assertEqual(rows.proposal_cache_events, ("hit",))
        self.assertEqual(rows.draft_cache_events, ("rebuild",))
        self.assertEqual(rows.proposal_methods, ("itl",))


if __name__ == "__main__":
    unittest.main()
