# sglang-group

[中文](README.md) | English

`sglang-group` is a unified SGLang 0.5.9 integration for heterogeneous-vocabulary
speculative decoding. It combines:

- `itl`: TokenTiming-style draft text generation, target re-tokenization, and
  DTW alignment diagnostics.
- `itl-base-slem`: first-paper SLEM/UAG-style string re-tokenization path.
- `itl-base-tli`: first-paper TLI path with draft probability rows over the
  target/draft vocabulary intersection.
- `auto`: choose one of the above methods from the request sampling temperature.

The target model and draft model can use different tokenizers. The draft model
can be a normal causal LM; no target-specific MTP/EAGLE/P-EAGLE model is
required.

## Install

For SGLang engine integration:

```bash
uv pip install "sglang-group[sglang] @ git+https://github.com/Huifu1018/sglang-group.git"
```

With pip:

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

## Quick Start

Use the wrapper, not `python -m sglang.launch_server` directly. SGLang 0.5.9
does not accept custom speculative algorithm names during argument parsing, so
the wrapper rewrites `SGLANG_GROUP` to the built-in `NGRAM` parser path and
patches the worker factory in-process.

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

Check the environment:

```bash
sglang-group-preflight
sglang-group-preflight --json
```

## Method Selection

You can force a method:

```bash
--sglang-group-method itl
--sglang-group-method itl-base-slem
--sglang-group-method itl-base-tli
```

Aliases are accepted:

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

Change the threshold:

```bash
--sglang-group-auto-high-temp-threshold 0.95
```

Or override the methods used by `auto`:

```bash
--sglang-group-auto-greedy-method itl-base-slem \
--sglang-group-auto-mid-sampling-method itl-base-tli \
--sglang-group-auto-high-sampling-method itl
```

On SGLang 0.5.9, auto selection is batch-level because the verify input class is
chosen once per decode batch. If a batch mixes temperatures, the highest
temperature in that batch drives the auto choice.

## Draft Backend

Default backend:

```bash
--sglang-group-draft-backend transformers
```

This is the original path: the target model is verified by SGLang, while the
draft model is loaded with Hugging Face Transformers and uses HF `past_key_values`.

SGLang-native draft backend:

```bash
--sglang-group-draft-backend sglang
```

This loads the draft model through SGLang 0.5.9's low-level `ModelRunner` and
runs draft prefill/decode with SGLang KV pools and model kernels. It supports
`itl`, `itl-base-slem`, `itl-base-tli`, and `auto`.

Use this when you want both target and draft execution to stay inside SGLang.
The native backend does not use `SGLANG_GROUP_DRAFT_DEVICE_MAP`; place it with
normal SGLang process placement such as `CUDA_VISIBLE_DEVICES`, `--tp`, and
container GPU assignment.

It also does not inherit the target model quantization. This matters when the
target is NVFP4/AWQ but the draft is a normal BF16/FP16 model such as
`Qwen/Qwen2.5-1.5B-Instruct`. If the draft checkpoint declares quantization in
its config, SGLang can detect it; otherwise force one only when needed:

```bash
--sglang-group-native-draft-quantization awq
```

The native backend keeps an accepted-context draft KV cache for the active
request. During proposal generation it snapshots the draft SGLang batch, decodes
speculative draft tokens, then rolls back only the speculative allocator and
batch state. Accepted target text is committed on the next proposal by extending
that draft cache, so repeated full draft prefill is avoided for single active
request streams.

If `--sglang-group-max-context-tokens` is set, the backend derives a draft KV
pool from that context cap. You can override it directly:

```bash
--sglang-group-native-draft-cache-tokens 8192 \
--sglang-group-native-draft-max-requests 1
```

For concurrent requests, the current native cache is conservative: one active
draft session is kept and different request ids trigger a rebuild. This preserves
correctness while keeping rejected heterogeneous-tokenizer draft tokens isolated.
If you run high concurrency, use `--sglang-group-max-context-tokens` to cap the
rebuild cost, or use backend `transformers` until multi-session native draft
pooling is needed.

## Force-Method Examples

Greedy, first-paper SLEM:

```bash
sglang-group-launch \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --trust-remote-code \
  --speculative-algorithm SGLANG_GROUP \
  --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 5 \
  --sglang-group-method itl-base-slem \
  --sglang-group-draft-device cuda:0
```

High-temperature sampling, TokenTiming ITL:

```bash
sglang-group-launch \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --trust-remote-code \
  --speculative-algorithm SGLANG_GROUP \
  --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 5 \
  --sglang-group-method itl \
  --sglang-group-draft-device cuda:0
```

Mid-temperature sampling, first-paper TLI:

```bash
sglang-group-launch \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --trust-remote-code \
  --speculative-algorithm SGLANG_GROUP \
  --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 5 \
  --sglang-group-method itl-base-tli \
  --sglang-group-draft-device cuda:0
```

## Runtime Flags

The wrapper consumes `--sglang-group-*` flags and forwards all other arguments
to SGLang.

| Flag | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `--sglang-group-method` | `SGLANG_GROUP_METHOD` | `auto` | `auto`, `itl`, `itl-base-slem`, or `itl-base-tli`. |
| `--sglang-group-auto-high-temp-threshold` | `SGLANG_GROUP_AUTO_HIGH_TEMP_THRESHOLD` | `0.9` | Temperature threshold for high-temp auto routing. |
| `--sglang-group-draft-backend` | `SGLANG_GROUP_DRAFT_BACKEND` | `transformers` | `transformers` or `sglang`. |
| `--sglang-group-draft-device` | `SGLANG_GROUP_DRAFT_DEVICE` | target CUDA device | Device for the Transformers draft model. |
| `--sglang-group-draft-device-map` | `SGLANG_GROUP_DRAFT_DEVICE_MAP` | unset | Passed to HF `from_pretrained(..., device_map=...)`. |
| `--sglang-group-draft-dtype` | `SGLANG_GROUP_DRAFT_DTYPE` | `auto` | `auto`, `fp16`, `bf16`, or `fp32`. |
| `--sglang-group-native-draft-quantization` | `SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION` | unset | Optional quantization override for backend `sglang`. |
| `--sglang-group-native-draft-cache-tokens` | `SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS` | derived | Draft KV pool tokens for backend `sglang`. |
| `--sglang-group-native-draft-max-requests` | `SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS` | `1` | Draft request pool size for backend `sglang`. |
| `--sglang-group-max-draft-tokens` | `SGLANG_GROUP_MAX_DRAFT_TOKENS` | derived | Max draft autoregressive steps per proposal. |
| `--sglang-group-max-context-tokens` | `SGLANG_GROUP_MAX_CONTEXT_TOKENS` | unset | Truncate draft-side context before proposal. |
| `--sglang-group-dtw-window` | `SGLANG_GROUP_DTW_WINDOW` | `8` | DTW window for `itl` alignment diagnostics. |
| `--sglang-group-assistant-lookbehind` | `SGLANG_GROUP_ASSISTANT_LOOKBEHIND` | `10` | Assistant-side SLEM lookbehind. |
| `--sglang-group-target-lookbehind` | `SGLANG_GROUP_TARGET_LOOKBEHIND` | `10` | Target-side SLEM lookbehind. |
| `--sglang-group-max-cached-requests` | `SGLANG_GROUP_MAX_CACHED_REQUESTS` | `256` | Per-request draft KV cache entries. |
| `--no-sglang-group-draft-cache` | `SGLANG_GROUP_ENABLE_DRAFT_CACHE=false` | enabled | Disable draft KV cache for diagnosis. |
| `--no-sglang-group-cache-clone` | `SGLANG_GROUP_CLONE_DRAFT_CACHE=false` | enabled | Disable conservative cache cloning. |
| `--sglang-group-tli-min-intersection` | `SGLANG_GROUP_TLI_MIN_INTERSECTION` | `1` | Minimum shared-token count for TLI. |
| `--sglang-group-metrics-log-interval` | `SGLANG_GROUP_METRICS_LOG_INTERVAL` | `60` | Worker metrics log interval; `0` disables. |

## Practical Defaults

For your current MiniMax-M2.7-AWQ/NVFP4 measurements:

- `temperature=0`: start with `itl-base-slem`.
- `temperature=0.6, top_p=0.95`: start with `itl-base-tli`.
- `temperature=1`: start with `itl`.
- Keep `--speculative-num-draft-tokens` at `5` first, then test `3`, `5`, `7`.
- Keep draft cache enabled after correctness is verified.
- For `--sglang-group-draft-backend sglang`, keep
  `--sglang-group-max-context-tokens` during early high-concurrency tests, for
  example `4096` or `8192`, so cross-request rebuilds do not dominate long
  generations.

## Constraints

- SGLang 0.5.9 only.
- Requires `--disable-overlap-schedule`; the wrapper adds it in legacy mode.
- Does not support pipeline parallelism yet.
- Does not support DP attention yet.
- Uses one linear candidate chain per request.
- Multimodal requests fall back to target-only verification for that request.
- `itl-base-slem` is greedy only.
- `--sglang-group-draft-backend sglang` does not support HF `device_map`.

## Development Checks

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall sglang_group tests
python -m build
```
