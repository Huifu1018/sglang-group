# sglang-group

[中文](README.md) | English

`sglang-group` is a **SGLang 0.5.9** integration for heterogeneous-vocabulary speculative decoding. It supports SLEM, TLI, and TokenTiming-style ITL proposal methods, so **both the target model and the draft model can run through SGLang** while using different tokenizers.

In normal deployments, use the **SGLang-native draft backend**:

```bash
--sglang-group-draft-backend sglang
```

This means:

- The target model is loaded and verified by SGLang.
- The draft model is also loaded and decoded by SGLang 0.5.9's low-level `ModelRunner`.
- The draft model can be a regular causal LM. It does not need to be a target-specific MTP/EAGLE/P-EAGLE model.
- The target tokenizer and draft tokenizer can be different.

The Transformers draft backend is only for compatibility, debugging, or controlled comparisons. See "Compatibility Mode" below.

## Install

Recommended with `uv`:

```bash
uv pip install "sglang-group[sglang] @ git+https://github.com/Huifu1018/sglang-group.git"
```

With `pip`:

```bash
pip install "sglang-group[sglang] @ git+https://github.com/Huifu1018/sglang-group.git"
```

The `sglang` extra pins:

```text
sglang==0.5.9
```

For local development:

```bash
git clone https://github.com/Huifu1018/sglang-group.git
cd sglang-group
uv pip install -e ".[dev]"
python -m unittest discover -s tests -p 'test_*.py'
```

## Recommended Production Path: Target + Draft Both Run Through SGLang

Use `sglang-group-launch`, not `python -m sglang.launch_server` directly. SGLang 0.5.9 does not accept custom speculative algorithm names during argument parsing, so `sglang-group-launch` rewrites `SGLANG_GROUP` to a built-in parser path and registers the `sglang-group` worker in-process.

Recommended standard command:

```bash
CUDA_VISIBLE_DEVICES=0 sglang-group-launch \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --speculative-algorithm SGLANG_GROUP \
  --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 5 \
  --sglang-group-method auto \
  --sglang-group-draft-backend sglang \
  --sglang-group-max-context-tokens 8192
```

In this command:

- `--model-path` is the target model, loaded by SGLang.
- `--speculative-draft-model-path` is the draft model, also loaded by the SGLang-native draft backend.
- `--sglang-group-draft-backend sglang` is the key flag: the draft does not run through Transformers.
- `--sglang-group-method auto` selects `itl-base-slem`, `itl-base-tli`, or `itl` from request temperature.
- `--sglang-group-max-context-tokens 8192` caps draft-side context to reduce long-context rebuild cost.

Check the environment before launch:

```bash
sglang-group-preflight
sglang-group-preflight --json
```

## SGLang-native Draft Backend

The SGLang-native backend loads the draft model through SGLang 0.5.9's low-level `ModelRunner`. Draft prefill/decode uses SGLang KV pools and model kernels. It supports:

- `itl`
- `itl-base-slem`
- `itl-base-tli`
- `auto`

GPU placement follows normal SGLang deployment practice:

```bash
CUDA_VISIBLE_DEVICES=0 sglang-group-launch ...
```

You can also use existing SGLang parallelism flags such as `--tp`. The native backend does not support HF `device_map`; do not combine it with `--sglang-group-draft-device-map`.

The native backend does not inherit the target model quantization. For example, if the target is NVFP4/AWQ and the draft is a normal BF16/FP16 model such as `Qwen/Qwen2.5-1.5B-Instruct`, do not make the draft reuse the target quantization. If the draft checkpoint declares quantization in its config, SGLang can detect it. Otherwise force quantization only when needed:

```bash
--sglang-group-native-draft-quantization awq
```

The current version implements accepted-context draft KV caching:

- The active request keeps accepted draft context.
- Proposal generation snapshots the draft SGLang batch.
- After decoding speculative draft tokens, only speculative allocator and batch state are rolled back.
- On the next proposal, the accepted target text is re-tokenized and the draft suffix is committed into the draft cache.

So single-request or low-concurrency streaming does not fully prefill the draft model every proposal.

For concurrent requests, the implementation is conservative: it keeps one active draft session, and a different request id triggers rebuild. For high-concurrency tests, keep `--sglang-group-max-context-tokens`, for example `4096` or `8192`. Multi-request LRU native draft caching can be added later.

## Method Selection

Force a method:

```bash
--sglang-group-method itl
--sglang-group-method itl-base-slem
--sglang-group-method itl-base-tli
```

Aliases:

```text
slem -> itl-base-slem
tli -> itl-base-tli
token_itl / token-itl / tokentiming -> itl
```

Default `auto` policy:

```text
temperature == 0:
  itl-base-slem

0 < temperature < 0.9:
  itl-base-tli

temperature >= 0.9:
  itl
```

Change the high-temperature threshold:

```bash
--sglang-group-auto-high-temp-threshold 0.95
```

Override the methods used by `auto`:

```bash
--sglang-group-auto-greedy-method itl-base-slem \
--sglang-group-auto-mid-sampling-method itl-base-tli \
--sglang-group-auto-high-sampling-method itl
```

On SGLang 0.5.9, method selection is batch-level because the verify input class is chosen once per decode batch. If a batch mixes temperatures, the highest temperature in the batch drives the method choice.

## Practical Defaults

For the current MiniMax-M2.7-AWQ/NVFP4 tests:

- `temperature=0`: start with `itl-base-slem`.
- `temperature=0.6, top_p=0.95`: start with `itl-base-tli`.
- `temperature=1`: start with `itl`.
- Start with `--speculative-num-draft-tokens 5`, then compare `3`, `5`, and `7`.
- With SGLang-native backend, start with `--sglang-group-max-context-tokens 8192`.

## Compatibility Mode: Draft Runs Through Transformers

This is not the recommended default path. Use it only when:

- You need controlled comparisons with the earlier HF draft implementation.
- The SGLang-native draft backend does not support a draft checkpoint yet.
- You need HF `device_map` for draft placement.
- You are debugging SGLang-native draft cache or memory behavior.

Command:

```bash
sglang-group-launch \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --speculative-algorithm SGLANG_GROUP \
  --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 5 \
  --sglang-group-method auto \
  --sglang-group-draft-backend transformers \
  --sglang-group-draft-device cuda:0
```

In this mode:

- The target model still runs through SGLang.
- The draft model runs through Hugging Face Transformers.
- Draft cache uses HF `past_key_values`.

## Flags

`sglang-group-launch` consumes `--sglang-group-*` flags and forwards the rest to SGLang. For backward compatibility, the code-level backend default is still `transformers`; production deployments should explicitly pass `--sglang-group-draft-backend sglang`.

| Flag | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `--sglang-group-method` | `SGLANG_GROUP_METHOD` | `auto` | `auto`, `itl`, `itl-base-slem`, or `itl-base-tli`. |
| `--sglang-group-auto-high-temp-threshold` | `SGLANG_GROUP_AUTO_HIGH_TEMP_THRESHOLD` | `0.9` | High-temperature routing threshold for `auto`. |
| `--sglang-group-draft-backend` | `SGLANG_GROUP_DRAFT_BACKEND` | `transformers` | Set this explicitly to `sglang` in production so target/draft both run through SGLang; `transformers` means HF draft. |
| `--sglang-group-draft-device` | `SGLANG_GROUP_DRAFT_DEVICE` | target CUDA device | Transformers backend only. |
| `--sglang-group-draft-device-map` | `SGLANG_GROUP_DRAFT_DEVICE_MAP` | unset | Transformers backend only; passed to HF `from_pretrained(..., device_map=...)`. |
| `--sglang-group-draft-dtype` | `SGLANG_GROUP_DRAFT_DTYPE` | `auto` | `auto`, `fp16`, `bf16`, or `fp32`. |
| `--sglang-group-native-draft-quantization` | `SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION` | unset | Draft quantization override for SGLang-native backend. |
| `--sglang-group-native-draft-cache-tokens` | `SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS` | derived | Draft KV pool tokens for SGLang-native backend. |
| `--sglang-group-native-draft-max-requests` | `SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS` | `1` | Draft request pool size for SGLang-native backend. |
| `--sglang-group-max-draft-tokens` | `SGLANG_GROUP_MAX_DRAFT_TOKENS` | derived | Max draft autoregressive steps per proposal. |
| `--sglang-group-max-context-tokens` | `SGLANG_GROUP_MAX_CONTEXT_TOKENS` | unset | Truncate draft-side context before proposal. |
| `--sglang-group-dtw-window` | `SGLANG_GROUP_DTW_WINDOW` | `8` | DTW window for `itl` diagnostics. |
| `--sglang-group-assistant-lookbehind` | `SGLANG_GROUP_ASSISTANT_LOOKBEHIND` | `10` | SLEM assistant-side lookbehind. |
| `--sglang-group-target-lookbehind` | `SGLANG_GROUP_TARGET_LOOKBEHIND` | `10` | SLEM target-side lookbehind. |
| `--sglang-group-max-cached-requests` | `SGLANG_GROUP_MAX_CACHED_REQUESTS` | `256` | Per-request draft KV cache entries for Transformers backend. |
| `--no-sglang-group-draft-cache` | `SGLANG_GROUP_ENABLE_DRAFT_CACHE=false` | enabled | Disable draft KV cache for diagnosis. |
| `--no-sglang-group-cache-clone` | `SGLANG_GROUP_CLONE_DRAFT_CACHE=false` | enabled | Disable conservative cache clone for Transformers backend. |
| `--sglang-group-tli-min-intersection` | `SGLANG_GROUP_TLI_MIN_INTERSECTION` | `1` | Minimum shared-token count for TLI. |
| `--sglang-group-metrics-log-interval` | `SGLANG_GROUP_METRICS_LOG_INTERVAL` | `60` | Worker metrics log interval; `0` disables it. |

## Constraints

- SGLang 0.5.9 only.
- Requires `--disable-overlap-schedule`; the wrapper adds it in legacy mode.
- Pipeline parallelism is not supported yet.
- DP attention is not supported yet.
- One linear candidate chain per request.
- Multimodal requests fall back to target-only verification for that request.
- `itl-base-slem` is greedy only.
- SGLang-native draft backend does not support HF `device_map`.
- Current SGLang-native draft cache is one active request cache, not multi-request LRU.

## Development Checks

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall sglang_group tests
python -m build
```
