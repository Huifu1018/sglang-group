"""Runtime configuration for the SGLang SGLANG_GROUP worker."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value
    return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when set.")
    return parsed


def _env_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"0", "false", "off", "none"}:
        return None
    parsed = float(value)
    if parsed == 0:
        return None
    if parsed < 0:
        raise ValueError(f"{name} must be positive when set.")
    return parsed


GROUP_METHODS = {"itl", "itl-base-slem", "itl-base-tli"}
GROUP_METHOD_ALIASES = {
    "token-itl": "itl",
    "token_itl": "itl",
    "tokentiming": "itl",
    "slem": "itl-base-slem",
    "base-slem": "itl-base-slem",
    "itl_base_slem": "itl-base-slem",
    "itl-base-slem": "itl-base-slem",
    "tli": "itl-base-tli",
    "base-tli": "itl-base-tli",
    "itl_base_tli": "itl-base-tli",
    "itl-base-tli": "itl-base-tli",
}
DRAFT_BACKENDS = {"transformers", "sglang"}
DRAFT_BACKEND_ALIASES = {
    "hf": "transformers",
    "huggingface": "transformers",
    "transformers": "transformers",
    "native": "sglang",
    "sglang": "sglang",
    "sglang-native": "sglang",
    "srt": "sglang",
}


def normalize_group_method(value: str, *, allow_auto: bool = False) -> str:
    method = value.strip().lower().replace("_", "-")
    if allow_auto and method == "auto":
        return method
    method = GROUP_METHOD_ALIASES.get(method, method)
    if method not in GROUP_METHODS:
        allowed = ["auto"] if allow_auto else []
        allowed.extend(sorted(GROUP_METHODS))
        raise ValueError(f"method must be one of: {', '.join(allowed)}.")
    return method


def normalize_draft_backend(value: str) -> str:
    backend = value.strip().lower().replace("_", "-")
    backend = DRAFT_BACKEND_ALIASES.get(backend, backend)
    if backend not in DRAFT_BACKENDS:
        raise ValueError(
            "draft backend must be one of: " + ", ".join(sorted(DRAFT_BACKENDS)) + "."
        )
    return backend


@dataclass(frozen=True)
class GroupSGLangConfig:
    """Configuration read from environment variables.

    SGLang 0.5.9 does not expose plugin-owned CLI flags. The launch wrapper
    keeps the standard SGLang arguments and uses these environment variables for
    integration-specific behavior.
    """

    method: str = "auto"
    auto_greedy_method: str = "itl-base-slem"
    auto_mid_sampling_method: str = "itl-base-tli"
    auto_high_sampling_method: str = "itl"
    auto_high_temp_threshold: float = 0.9
    draft_backend: str = "sglang"
    draft_device: str | None = None
    draft_device_map: str | None = None
    draft_dtype: str = "auto"
    native_draft_quantization: str | None = None
    native_draft_cache_tokens: int | None = None
    native_draft_max_requests: int = 1
    dtw_window: int | None = 8
    max_draft_tokens: int | None = None
    max_context_tokens: int | None = None
    assistant_lookbehind: int = 10
    target_lookbehind: int = 10
    max_cached_requests: int = 256
    add_special_tokens: bool = False
    disable_cuda_graph: bool = True
    enable_draft_cache: bool = True
    clone_draft_cache: bool = True
    tli_min_intersection: int = 1
    metrics_log_interval: float | None = 60.0

    @classmethod
    def from_env(cls, *, default_draft_device: str | None = None) -> "GroupSGLangConfig":
        method = normalize_group_method(
            _env_value("SGLANG_GROUP_METHOD", default="auto") or "auto",
            allow_auto=True,
        )
        auto_greedy_method = normalize_group_method(
            _env_value("SGLANG_GROUP_AUTO_GREEDY_METHOD", default="itl-base-slem")
            or "itl-base-slem"
        )
        auto_mid_sampling_method = normalize_group_method(
            _env_value(
                "SGLANG_GROUP_AUTO_MID_SAMPLING_METHOD",
                default="itl-base-tli",
            )
            or "itl-base-tli"
        )
        auto_high_sampling_method = normalize_group_method(
            _env_value(
                "SGLANG_GROUP_AUTO_HIGH_SAMPLING_METHOD",
                default="itl",
            )
            or "itl"
        )
        return cls(
            method=method,
            auto_greedy_method=auto_greedy_method,
            auto_mid_sampling_method=auto_mid_sampling_method,
            auto_high_sampling_method=auto_high_sampling_method,
            auto_high_temp_threshold=(
                _env_float("SGLANG_GROUP_AUTO_HIGH_TEMP_THRESHOLD", 0.9) or 0.9
            ),
            draft_backend=normalize_draft_backend(
                _env_value("SGLANG_GROUP_DRAFT_BACKEND", default="sglang")
                or "sglang"
            ),
            draft_device=os.getenv("SGLANG_GROUP_DRAFT_DEVICE", default_draft_device),
            draft_device_map=os.getenv("SGLANG_GROUP_DRAFT_DEVICE_MAP") or None,
            draft_dtype=os.getenv("SGLANG_GROUP_DRAFT_DTYPE", "auto"),
            native_draft_quantization=os.getenv(
                "SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION"
            )
            or None,
            native_draft_cache_tokens=_env_int(
                "SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS", None
            ),
            native_draft_max_requests=(
                _env_int("SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS", 1) or 1
            ),
            dtw_window=_env_int("SGLANG_GROUP_DTW_WINDOW", 8),
            max_draft_tokens=_env_int("SGLANG_GROUP_MAX_DRAFT_TOKENS", None),
            max_context_tokens=_env_int("SGLANG_GROUP_MAX_CONTEXT_TOKENS", None),
            assistant_lookbehind=_env_int("SGLANG_GROUP_ASSISTANT_LOOKBEHIND", 10) or 10,
            target_lookbehind=_env_int("SGLANG_GROUP_TARGET_LOOKBEHIND", 10) or 10,
            max_cached_requests=_env_int("SGLANG_GROUP_MAX_CACHED_REQUESTS", 256) or 256,
            add_special_tokens=_env_bool("SGLANG_GROUP_ADD_SPECIAL_TOKENS", False),
            disable_cuda_graph=_env_bool("SGLANG_GROUP_DISABLE_CUDA_GRAPH", True),
            enable_draft_cache=_env_bool("SGLANG_GROUP_ENABLE_DRAFT_CACHE", True),
            clone_draft_cache=_env_bool("SGLANG_GROUP_CLONE_DRAFT_CACHE", True),
            tli_min_intersection=_env_int("SGLANG_GROUP_TLI_MIN_INTERSECTION", 1) or 1,
            metrics_log_interval=_env_float("SGLANG_GROUP_METRICS_LOG_INTERVAL", 60.0),
        )

    def method_for_batch(
        self,
        *,
        is_all_greedy: bool,
        max_temperature: float | None = None,
    ) -> str:
        if self.method != "auto":
            return self.method
        if is_all_greedy or (max_temperature is not None and max_temperature <= 0):
            return self.auto_greedy_method
        if (
            max_temperature is not None
            and max_temperature >= self.auto_high_temp_threshold
        ):
            return self.auto_high_sampling_method
        return self.auto_mid_sampling_method
