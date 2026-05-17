"""SGLang-native draft backend for heterogeneous-vocabulary proposals.

This backend is intentionally independent from SGLang's built-in EAGLE worker:
EAGLE assumes target/draft scheduling state can be advanced together, while the
SGLANG_GROUP methods re-tokenize text between heterogeneous vocabularies. The
backend below uses SGLang's low-level ModelRunner for draft forward passes and
keeps proposal generation scratch-local so rejected draft tokens cannot corrupt
future draft state.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

from .config import GroupSGLangConfig

logger = logging.getLogger(__name__)


class _ScratchTreeCache(SimpleNamespace):
    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True

    def evict(self, *args, **kwargs) -> None:
        return None

    def pretty_print(self) -> None:
        return None

    def available_and_evictable_str(self) -> str:
        allocator = self.token_to_kv_pool_allocator
        available = getattr(allocator, "available_size", lambda: "unknown")()
        return f"available={available}, evictable=0"


@dataclass
class SGLangNativeDraftSession:
    backend: "SGLangNativeDraftBackend"
    batch: object
    next_token_logits: object

    def decode(self, token_id: int) -> object:
        self.next_token_logits = self.backend.decode(self, token_id)
        return self.next_token_logits


class SGLangNativeDraftBackend:
    """Run the draft model through SGLang 0.5.9's ModelRunner.

    The backend uses a single-request scratch batch per proposal. It clears the
    draft runner memory pools before each prefill, then decodes proposed draft
    tokens incrementally inside that scratch batch.
    """

    def __init__(
        self,
        *,
        server_args: object,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int | None,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        config: GroupSGLangConfig,
        trust_remote_code: bool,
    ) -> None:
        if config.draft_device_map:
            raise ValueError(
                "SGLang-native draft backend does not support "
                "SGLANG_GROUP_DRAFT_DEVICE_MAP. Use backend=transformers for "
                "HF device_map placement, or place the SGLang worker with CUDA_VISIBLE_DEVICES."
            )

        self.config = config
        self.server_args = copy.copy(server_args)
        self.server_args.model_path = server_args.speculative_draft_model_path
        self.server_args.revision = getattr(
            server_args, "speculative_draft_model_revision", None
        ) or getattr(server_args, "revision", None)
        self.server_args.skip_tokenizer_init = True
        self.server_args.disable_cuda_graph = True
        self.server_args.disable_overlap_schedule = True
        # Do not inherit the target model quantization for a generic draft model.
        # If the draft checkpoint declares quantization in its config, SGLang can
        # still detect it. Use SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION to force one.
        self.server_args.quantization = config.native_draft_quantization
        self._configure_scratch_cache_size(server_args)
        if config.draft_dtype != "auto":
            self.server_args.dtype = _sglang_dtype_name(config.draft_dtype)

        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port

        self.tokenizer = self._load_tokenizer(
            server_args.speculative_draft_model_path,
            trust_remote_code=trust_remote_code,
        )
        self.model_runner = self._load_model_runner()
        self.device = self.model_runner.device

        logger.info(
            "Initialized SGLang-native draft backend: draft=%s, device=%s, "
            "tp_rank=%s, cache_tokens=%s, max_requests=%s",
            server_args.speculative_draft_model_path,
            self.device,
            tp_rank,
            getattr(self.server_args, "draft_runner_cache_size", None),
            getattr(self.server_args, "max_num_reqs", None),
        )

    def _configure_scratch_cache_size(self, source_server_args: object) -> None:
        page_size = int(getattr(self.server_args, "page_size", 1) or 1)
        requested_tokens = self.config.native_draft_cache_tokens
        if requested_tokens is None and self.config.max_context_tokens is not None:
            draft_tokens = self.config.max_draft_tokens
            if draft_tokens is None:
                draft_tokens = getattr(
                    source_server_args,
                    "speculative_num_draft_tokens",
                    None,
                )
            requested_tokens = self.config.max_context_tokens + int(draft_tokens or 8) + 16

        if requested_tokens is not None:
            requested_tokens = _ceil_to_page(int(requested_tokens), page_size)
            source_tokens = getattr(
                source_server_args,
                "draft_runner_cache_size",
                requested_tokens,
            )
            if source_tokens:
                requested_tokens = min(requested_tokens, int(source_tokens))
            self.server_args.draft_runner_cache_size = requested_tokens

        max_requests = max(1, int(self.config.native_draft_max_requests))
        self.server_args.max_num_reqs = max_requests

    def _load_tokenizer(self, model_path: str, *, trust_remote_code: bool):
        try:
            from sglang.srt.utils.hf_transformers_utils import get_tokenizer

            return get_tokenizer(
                model_path,
                tokenizer_mode=getattr(self.server_args, "tokenizer_mode", "auto"),
                trust_remote_code=trust_remote_code,
                revision=getattr(self.server_args, "revision", None),
            )
        except Exception:
            logger.debug("Falling back to AutoTokenizer for draft tokenizer.", exc_info=True)
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                revision=getattr(self.server_args, "revision", None),
            )

    def _load_model_runner(self):
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_executor.model_runner import ModelRunner

        model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=self.server_args.model_path,
            model_revision=self.server_args.revision,
            is_draft_model=False,
        )
        return ModelRunner(
            model_config=model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.server_args.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
            server_args=self.server_args,
            is_draft_worker=True,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
        )

    def clear(self) -> None:
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    def prefill(self, input_ids: Sequence[int], *, rid: str) -> SGLangNativeDraftSession:
        import torch

        ids = [int(token_id) for token_id in input_ids]
        if not ids:
            raise ValueError("SGLang-native draft context must contain at least one token.")

        with torch.no_grad():
            self.clear()
            batch = self._make_extend_batch(ids, rid=rid)
            logits = self._forward_batch(batch)
        return SGLangNativeDraftSession(
            backend=self,
            batch=batch,
            next_token_logits=logits,
        )

    def decode(self, session: SGLangNativeDraftSession, token_id: int) -> object:
        import torch

        batch = session.batch
        with torch.no_grad():
            batch.output_ids = torch.tensor(
                [int(token_id)],
                dtype=torch.int64,
                device=self.device,
            )
            batch.prepare_for_decode()
            return self._forward_batch(batch)

    def _make_extend_batch(self, input_ids: list[int], *, rid: str):
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        sampling_params = SamplingParams(temperature=0.0, max_new_tokens=1)
        req = Req(
            rid=f"sglang-group-draft-{rid}",
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

        tree_cache = _ScratchTreeCache(
            page_size=self.model_runner.server_args.page_size,
            device=self.device,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            req_to_token_pool=self.model_runner.req_to_token_pool,
        )
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        return batch

    def _forward_batch(self, batch: object) -> object:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        self._maybe_prepare_mlp_sync_batch(batch)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch).logits_output
        return logits_output.next_token_logits

    def _maybe_prepare_mlp_sync_batch(self, batch: object) -> None:
        try:
            from sglang.srt.managers.scheduler_dp_attn_mixin import (
                prepare_mlp_sync_batch_raw,
            )
            from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
        except Exception:
            return

        if not require_mlp_sync(self.model_runner.server_args):
            return

        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=self.model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_group=self.model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(self.model_runner.server_args),
            disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
        )


def _sglang_dtype_name(dtype_name: str) -> str:
    normalized = dtype_name.lower()
    mapping = {
        "float16": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float32": "float32",
        "fp32": "float32",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported SGLANG_GROUP_DRAFT_DTYPE: {dtype_name}") from exc


def _ceil_to_page(value: int, page_size: int) -> int:
    if page_size <= 1:
        return value
    return ((value + page_size - 1) // page_size) * page_size
