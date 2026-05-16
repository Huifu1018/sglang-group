"""Server-argument normalization for SGLANG_GROUP."""

from __future__ import annotations


DEFAULT_SPECULATIVE_STEPS = 4
DEFAULT_MAX_RUNNING_REQUESTS = 48


def validate_server_args(server_args: object) -> None:
    """Normalize SGLang 0.5.9 args for the SGLANG_GROUP worker."""

    if not getattr(server_args, "speculative_draft_model_path", None):
        raise ValueError(
            "SGLANG_GROUP requires --speculative-draft-model-path to point to the "
            "off-the-shelf draft model."
        )

    if getattr(server_args, "enable_dp_attention", False):
        raise ValueError("SGLANG_GROUP does not support --enable-dp-attention yet.")

    if getattr(server_args, "pp_size", 1) != 1:
        raise ValueError("SGLANG_GROUP does not support pipeline parallelism yet.")

    device = getattr(server_args, "device", None)
    if device is not None and not str(device).startswith("cuda"):
        raise ValueError("SGLANG_GROUP currently requires CUDA for SGLang verification.")

    if getattr(server_args, "max_running_requests", None) is None:
        setattr(server_args, "max_running_requests", DEFAULT_MAX_RUNNING_REQUESTS)

    setattr(server_args, "disable_overlap_schedule", True)
    setattr(server_args, "enable_mixed_chunk", False)

    if hasattr(server_args, "disable_cuda_graph"):
        setattr(server_args, "disable_cuda_graph", True)

    if getattr(server_args, "speculative_num_steps", None) is None:
        setattr(server_args, "speculative_num_steps", DEFAULT_SPECULATIVE_STEPS)

    setattr(server_args, "speculative_eagle_topk", 1)
    expected_draft_tokens = int(getattr(server_args, "speculative_num_steps")) + 1
    if getattr(server_args, "speculative_num_draft_tokens", None) is None:
        setattr(server_args, "speculative_num_draft_tokens", expected_draft_tokens)

    draft_tokens = int(getattr(server_args, "speculative_num_draft_tokens"))
    if draft_tokens < 2:
        raise ValueError("SGLANG_GROUP requires --speculative-num-draft-tokens >= 2.")
