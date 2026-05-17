"""Launch SGLang 0.5.9 with SGLANG_GROUP compatibility."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import MutableMapping

from sglang_group import SGLANG_GROUP_ALGORITHM
from sglang_group.sglang.compat import (
    LEGACY_PATCH_ENV,
    has_native_custom_spec_registry,
    patch_legacy_ngram_worker,
)
from sglang_group.sglang.plugin import activate
from sglang_group.sglang.validation import validate_server_args


GROUP_VALUE_FLAGS = {
    "--sglang-group-method": "SGLANG_GROUP_METHOD",
    "--sglang-group-auto-greedy-method": "SGLANG_GROUP_AUTO_GREEDY_METHOD",
    "--sglang-group-auto-mid-sampling-method": "SGLANG_GROUP_AUTO_MID_SAMPLING_METHOD",
    "--sglang-group-auto-high-sampling-method": "SGLANG_GROUP_AUTO_HIGH_SAMPLING_METHOD",
    "--sglang-group-auto-high-temp-threshold": "SGLANG_GROUP_AUTO_HIGH_TEMP_THRESHOLD",
    "--sglang-group-draft-backend": "SGLANG_GROUP_DRAFT_BACKEND",
    "--sglang-group-draft-device": "SGLANG_GROUP_DRAFT_DEVICE",
    "--sglang-group-draft-device-map": "SGLANG_GROUP_DRAFT_DEVICE_MAP",
    "--sglang-group-draft-dtype": "SGLANG_GROUP_DRAFT_DTYPE",
    "--sglang-group-native-draft-quantization": "SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION",
    "--sglang-group-native-draft-cache-tokens": "SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS",
    "--sglang-group-native-draft-max-requests": "SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS",
    "--sglang-group-max-draft-tokens": "SGLANG_GROUP_MAX_DRAFT_TOKENS",
    "--sglang-group-max-context-tokens": "SGLANG_GROUP_MAX_CONTEXT_TOKENS",
    "--sglang-group-assistant-lookbehind": "SGLANG_GROUP_ASSISTANT_LOOKBEHIND",
    "--sglang-group-target-lookbehind": "SGLANG_GROUP_TARGET_LOOKBEHIND",
    "--sglang-group-dtw-window": "SGLANG_GROUP_DTW_WINDOW",
    "--sglang-group-max-cached-requests": "SGLANG_GROUP_MAX_CACHED_REQUESTS",
    "--sglang-group-tli-min-intersection": "SGLANG_GROUP_TLI_MIN_INTERSECTION",
    "--sglang-group-metrics-log-interval": "SGLANG_GROUP_METRICS_LOG_INTERVAL",
}

GROUP_BOOL_FLAGS = {
    "--no-sglang-group-draft-cache": ("SGLANG_GROUP_ENABLE_DRAFT_CACHE", "false"),
    "--no-sglang-group-cache-clone": ("SGLANG_GROUP_CLONE_DRAFT_CACHE", "false"),
}


def _rewrite_algorithm(argv: list[str]) -> list[str]:
    rewritten = list(argv)
    for index, item in enumerate(rewritten):
        if item == "--speculative-algorithm" and index + 1 < len(rewritten):
            if rewritten[index + 1].upper() == SGLANG_GROUP_ALGORITHM:
                rewritten[index + 1] = "NGRAM"
        elif item.startswith("--speculative-algorithm="):
            name = item.split("=", 1)[1]
            if name.upper() == SGLANG_GROUP_ALGORITHM:
                rewritten[index] = "--speculative-algorithm=NGRAM"
    return rewritten


def _uses_sglang_group(argv: list[str]) -> bool:
    for index, item in enumerate(argv):
        if item == "--speculative-algorithm" and index + 1 < len(argv):
            return argv[index + 1].upper() == SGLANG_GROUP_ALGORITHM
        if item.startswith("--speculative-algorithm="):
            return item.split("=", 1)[1].upper() == SGLANG_GROUP_ALGORITHM
    return False


def _consume_group_args(
    argv: list[str],
    *,
    environ: MutableMapping[str, str] | None = None,
) -> list[str]:
    environ = os.environ if environ is None else environ
    remaining: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in GROUP_BOOL_FLAGS:
            key, value = GROUP_BOOL_FLAGS[item]
            environ[key] = value
            index += 1
            continue

        if item in GROUP_VALUE_FLAGS:
            if index + 1 >= len(argv):
                raise SystemExit(f"{item} requires a value.")
            environ[GROUP_VALUE_FLAGS[item]] = argv[index + 1]
            index += 2
            continue

        if item.startswith("--") and "=" in item:
            flag, value = item.split("=", 1)
            if flag in GROUP_VALUE_FLAGS:
                environ[GROUP_VALUE_FLAGS[flag]] = value
                index += 1
                continue

        remaining.append(item)
        index += 1
    return remaining


def _ensure_legacy_ngram_flags(argv: list[str]) -> list[str]:
    rewritten = list(argv)
    if not _has_option(rewritten, "--speculative-ngram-max-bfs-breadth"):
        rewritten += ["--speculative-ngram-max-bfs-breadth", "1"]
    if not _has_option(rewritten, "--disable-cuda-graph"):
        rewritten.append("--disable-cuda-graph")
    if not _has_option(rewritten, "--disable-overlap-schedule"):
        rewritten.append("--disable-overlap-schedule")
    return rewritten


def _has_option(argv: list[str], option: str) -> bool:
    prefix = option + "="
    return any(item == option or item.startswith(prefix) for item in argv)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in argv):
        parser = argparse.ArgumentParser(
            description=(
                "Launch SGLang with SGLANG_GROUP. Pass normal sglang.launch_server "
                "arguments; use --speculative-algorithm SGLANG_GROUP."
            ),
            add_help=True,
        )
        parser.add_argument(
            "--sglang-group-method",
            choices=["auto", "itl", "itl-base-slem", "itl-base-tli"],
            help="Unified speculative method. Default: auto.",
        )
        parser.add_argument(
            "--sglang-group-auto-high-temp-threshold",
            help="Temperature at or above which auto selects the high-temp method.",
        )
        parser.add_argument(
            "--sglang-group-draft-backend",
            choices=["transformers", "sglang"],
            help="Draft execution backend. Default: transformers.",
        )
        parser.add_argument(
            "--sglang-group-native-draft-quantization",
            help="Optional SGLang quantization override for backend=sglang.",
        )
        parser.add_argument(
            "--sglang-group-native-draft-cache-tokens",
            help="Optional draft KV pool token cap for backend=sglang.",
        )
        parser.add_argument(
            "--sglang-group-native-draft-max-requests",
            help="Draft request pool size for backend=sglang. Default: 1.",
        )
        parser.add_argument("sglang_args", nargs=argparse.REMAINDER)
        parser.parse_args(argv)
        return

    argv = _consume_group_args(argv)
    if _uses_sglang_group(argv) and not has_native_custom_spec_registry():
        os.environ[LEGACY_PATCH_ENV] = "1"
        patch_legacy_ngram_worker()
        argv = _rewrite_algorithm(argv)
        argv = _ensure_legacy_ngram_flags(argv)
    else:
        activate()

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(argv)
    if (
        str(getattr(server_args, "speculative_algorithm", "")).upper() == "NGRAM"
        and os.getenv(LEGACY_PATCH_ENV) == "1"
    ):
        validate_server_args(server_args)

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
