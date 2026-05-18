import unittest
from types import SimpleNamespace

from sglang_group.sglang.native_draft import (
    SGLangNativeDraftBackend,
    SGLangNativeDraftSession,
    _ceil_to_page,
)
from sglang_group.sglang.proposer import HeterogeneousDraftProposer


class NativeDraftSessionTests(unittest.TestCase):
    def test_ceil_to_page(self):
        self.assertEqual(_ceil_to_page(7, 1), 7)
        self.assertEqual(_ceil_to_page(7, 4), 8)
        self.assertEqual(_ceil_to_page(8, 4), 8)

    def test_speculative_rollback_restores_batch_req_and_allocator(self):
        class FakeAllocator:
            def __init__(self):
                self.values = ["accepted"]

            def backup_state(self):
                return list(self.values)

            def restore_state(self, state):
                self.values = list(state)

        class FakeModelRunner:
            def __init__(self):
                self.token_to_kv_pool_allocator = FakeAllocator()

        class FakeBackend:
            def __init__(self):
                self.model_runner = FakeModelRunner()

            def decode(self, session, token_id):
                session.batch.seq_lens[0] += 1
                session.batch.seq_lens_sum += 1
                session.batch.output_ids = [token_id]
                session.batch.reqs[0].kv_committed_len += 1
                session.batch.reqs[0].output_ids.append(token_id)
                self.model_runner.token_to_kv_pool_allocator.values.append(token_id)
                return [f"logits-{token_id}"]

        class FakeReq:
            def __init__(self):
                self.kv_committed_len = 3
                self.kv_allocated_len = 3
                self.decode_batch_idx = 0
                self.extend_batch_idx = None
                self.output_ids = []
                self.fill_ids = [1, 2, 3]

        class FakeBatch:
            def __init__(self):
                self.input_ids = [1, 2, 3]
                self.output_ids = []
                self.out_cache_loc = [11]
                self.seq_lens = [3]
                self.seq_lens_cpu = [3]
                self.seq_lens_sum = 3
                self.orig_seq_lens = [3]
                self.reqs = [FakeReq()]

        backend = FakeBackend()
        session = SGLangNativeDraftSession(
            backend=backend,
            batch=FakeBatch(),
            next_token_logits=["base-logits"],
            rid="r0",
            accepted_input_ids=(1, 2, 3),
        )

        session.begin_speculative()
        self.assertEqual(session.decode(4), ["logits-4"])
        self.assertEqual(session.batch.seq_lens, [4])
        self.assertEqual(backend.model_runner.token_to_kv_pool_allocator.values, ["accepted", 4])

        session.rollback_speculative()

        self.assertEqual(session.next_token_logits, ["base-logits"])
        self.assertEqual(session.batch.seq_lens, [3])
        self.assertEqual(session.batch.seq_lens_sum, 3)
        self.assertEqual(session.batch.output_ids, [])
        self.assertEqual(session.batch.reqs[0].kv_committed_len, 3)
        self.assertEqual(session.batch.reqs[0].output_ids, [])
        self.assertEqual(backend.model_runner.token_to_kv_pool_allocator.values, ["accepted"])

    def test_speculative_rollback_restores_req_to_token_row(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class FakeAllocator:
            def backup_state(self):
                return ("allocator",)

            def restore_state(self, state):
                self.restored = state

        class FakeReqToTokenPool:
            def __init__(self):
                self.req_to_token = torch.tensor([[10, 11, 12, 0]], dtype=torch.int32)

        class FakeModelRunner:
            def __init__(self):
                self.token_to_kv_pool_allocator = FakeAllocator()
                self.req_to_token_pool = FakeReqToTokenPool()

        class FakeBackend:
            def __init__(self):
                self.model_runner = FakeModelRunner()

        class FakeReq:
            req_pool_idx = 0

        class FakeBatch:
            def __init__(self):
                self.seq_lens = torch.tensor([3], dtype=torch.int32)
                self.seq_lens_cpu = torch.tensor([3], dtype=torch.int32)
                self.seq_lens_sum = 3
                self.reqs = [FakeReq()]

        backend = FakeBackend()
        session = SGLangNativeDraftSession(
            backend=backend,
            batch=FakeBatch(),
            next_token_logits=["base-logits"],
            rid="r0",
            accepted_input_ids=(1, 2, 3),
        )

        session.begin_speculative()
        backend.model_runner.req_to_token_pool.req_to_token[0, 1] = 99
        session.rollback_speculative()

        self.assertEqual(
            backend.model_runner.req_to_token_pool.req_to_token[0].tolist(),
            [10, 11, 12, 0],
        )


class NativeProposerHelpersTests(unittest.TestCase):
    def test_native_fork_begins_speculation_and_rollback_restores(self):
        class FakeSession:
            def __init__(self):
                self.begin_count = 0
                self.rollback_count = 0

            def begin_speculative(self):
                self.begin_count += 1
                return self

            def rollback_speculative(self):
                self.rollback_count += 1

        proposer = object.__new__(HeterogeneousDraftProposer)
        proposer.native_backend = object()
        session = FakeSession()

        self.assertIs(proposer._fork_past_key_values(session), session)
        proposer._rollback_past_key_values(session)

        self.assertEqual(session.begin_count, 1)
        self.assertEqual(session.rollback_count, 1)


class NativeDraftCachePolicyTests(unittest.TestCase):
    def _backend(self, *, native_kv_cache: bool):
        backend = object.__new__(SGLangNativeDraftBackend)
        backend.config = SimpleNamespace(
            enable_draft_cache=True,
            native_draft_kv_cache=native_kv_cache,
        )
        backend._cached_session = SimpleNamespace(
            accepted_input_ids=(1, 2),
            committed=[],
            commit_tokens=lambda suffix: None,
        )
        backend._cached_rid = "r0"
        backend._cached_input_ids = (1, 2)
        backend.prefill_calls = []

        def prefill(ids, *, rid):
            backend.prefill_calls.append((tuple(ids), rid))
            return SimpleNamespace(accepted_input_ids=tuple(ids))

        backend.prefill = prefill
        return backend

    def test_native_kv_cache_is_off_by_default_even_when_general_cache_is_enabled(self):
        backend = self._backend(native_kv_cache=False)
        session, event = backend.ensure_session((1, 2), rid="r0")

        self.assertEqual(event, "sglang-rebuild")
        self.assertEqual(session.accepted_input_ids, (1, 2))
        self.assertEqual(backend.prefill_calls, [((1, 2), "r0")])
        self.assertIsNone(backend._cached_session)

    def test_native_kv_cache_reuses_only_when_explicitly_enabled(self):
        backend = self._backend(native_kv_cache=True)
        session, event = backend.ensure_session((1, 2), rid="r0")

        self.assertEqual(event, "sglang-hit")
        self.assertEqual(session.accepted_input_ids, (1, 2))
        self.assertEqual(backend.prefill_calls, [])


if __name__ == "__main__":
    unittest.main()
