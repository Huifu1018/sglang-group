# sglang-group

中文 | [English](README_EN.md)

`sglang-group` 是一个面向 **SGLang 0.5.9** 的异构词表 speculative decoding 集成项目。它把两篇论文中的路线合并到同一个 SGLang worker 里，让 target 模型和 draft 模型可以使用不同 tokenizer，并且 draft 模型可以是普通 causal LM，不要求目标模型自带 MTP/EAGLE/P-EAGLE。

当前支持：

- `itl`：TokenTiming 风格的 draft 文本生成、target 重新分词，以及 DTW 对齐诊断。
- `itl-base-slem`：第一篇论文中的 SLEM/UAG 字符串重新分词路线。
- `itl-base-tli`：第一篇论文中的 TLI 词表交集概率路线。
- `auto`：根据请求的采样温度自动选择 `itl`、`itl-base-slem` 或 `itl-base-tli`。

## 安装

用于 SGLang engine 集成：

```bash
uv pip install "sglang-group[sglang] @ git+https://github.com/Huifu1018/sglang-group.git"
```

使用 pip：

```bash
pip install "sglang-group[sglang] @ git+https://github.com/Huifu1018/sglang-group.git"
```

`sglang` extra 会固定：

```text
sglang==0.5.9
```

本地开发：

```bash
git clone https://github.com/Huifu1018/sglang-group.git
cd sglang-group
uv pip install -e ".[dev]"
python -m unittest discover -s tests -p 'test_*.py'
```

## 快速启动

请使用 `sglang-group-launch`，不要直接使用 `python -m sglang.launch_server`。原因是 SGLang 0.5.9 在参数解析阶段不接受自定义 speculative algorithm 名称，所以 wrapper 会把 `SGLANG_GROUP` 改写到内置 `NGRAM` 解析路径，并在进程内 patch worker factory。

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

启动前可以做环境检查：

```bash
sglang-group-preflight
sglang-group-preflight --json
```

## 方法选择

可以强制指定某一种方法：

```bash
--sglang-group-method itl
--sglang-group-method itl-base-slem
--sglang-group-method itl-base-tli
```

支持别名：

```text
slem -> itl-base-slem
tli -> itl-base-tli
token_itl / token-itl / tokentiming -> itl
```

默认 `auto` 策略：

```text
temperature == 0:
  itl-base-slem

0 < temperature < 0.9:
  itl-base-tli

temperature >= 0.9:
  itl
```

修改高温阈值：

```bash
--sglang-group-auto-high-temp-threshold 0.95
```

也可以改写 `auto` 使用的方法：

```bash
--sglang-group-auto-greedy-method itl-base-slem \
--sglang-group-auto-mid-sampling-method itl-base-tli \
--sglang-group-auto-high-sampling-method itl
```

注意：在 SGLang 0.5.9 中，`auto` 是 batch-level 选择，因为 verify input class 在一个 decode batch 内只会选一次。如果同一个 batch 混合不同 temperature，会用 batch 内最高 temperature 决定方法。

## Draft Backend

默认 backend：

```bash
--sglang-group-draft-backend transformers
```

这个路径下，target 模型由 SGLang 验证，draft 模型由 Hugging Face Transformers 加载，并使用 HF `past_key_values`。

SGLang-native draft backend：

```bash
--sglang-group-draft-backend sglang
```

这个路径会通过 SGLang 0.5.9 的低层 `ModelRunner` 加载 draft 模型，draft prefill/decode 使用 SGLang KV pool 和模型 kernel。它支持 `itl`、`itl-base-slem`、`itl-base-tli` 和 `auto`。

如果你希望 target 和 draft 都留在 SGLang 执行路径里，优先使用这个 backend。native backend 不支持 `SGLANG_GROUP_DRAFT_DEVICE_MAP`；请通过常规 SGLang 部署方式放置 GPU，例如 `CUDA_VISIBLE_DEVICES`、`--tp` 或容器 GPU 分配。

native backend 不会继承 target 模型的量化配置。比如 target 是 NVFP4/AWQ，而 draft 是普通 BF16/FP16 的 `Qwen/Qwen2.5-1.5B-Instruct`，不要让 draft 误用 target 的量化方式。如果 draft checkpoint 自己声明了量化配置，SGLang 可以识别；否则只在确实需要时显式指定：

```bash
--sglang-group-native-draft-quantization awq
```

当前版本的 native backend 已经实现 accepted-context draft KV cache：同一个 active request 会保留已经接受的 draft 上下文；proposal 期间会 snapshot draft SGLang batch，decode speculative draft tokens，然后只回滚 speculative allocator 和 batch 状态。下一轮 proposal 会把已接受 target 文本重新映射后的 draft suffix commit 进 draft cache，因此单请求流式生成不会每轮都完整 draft prefill。

如果设置了 `--sglang-group-max-context-tokens`，backend 会根据该上下文上限推导 draft KV pool 大小。也可以直接指定：

```bash
--sglang-group-native-draft-cache-tokens 8192 \
--sglang-group-native-draft-max-requests 1
```

并发请求下当前 native cache 是保守实现：只保留一个 active draft session，不同 request id 会触发 rebuild。这样可以保证异构 tokenizer 下 rejected speculative tokens 不会污染其他请求。高并发测试时建议设置 `--sglang-group-max-context-tokens` 控制 rebuild 成本；如果需要多请求 native draft cache，可以在后续版本继续实现 LRU multi-session pooling。

## 常用启动示例

贪心生成，第一篇论文 SLEM 路线：

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

高温采样，TokenTiming ITL 路线：

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

中温采样，第一篇论文 TLI 路线：

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

使用 SGLang-native draft backend：

```bash
sglang-group-launch \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --trust-remote-code \
  --speculative-algorithm SGLANG_GROUP \
  --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 5 \
  --sglang-group-method auto \
  --sglang-group-draft-backend sglang \
  --sglang-group-max-context-tokens 8192
```

## 运行参数

wrapper 会消费 `--sglang-group-*` 参数，其余参数会继续转发给 SGLang。

| 参数 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--sglang-group-method` | `SGLANG_GROUP_METHOD` | `auto` | `auto`、`itl`、`itl-base-slem` 或 `itl-base-tli`。 |
| `--sglang-group-auto-high-temp-threshold` | `SGLANG_GROUP_AUTO_HIGH_TEMP_THRESHOLD` | `0.9` | `auto` 高温路由阈值。 |
| `--sglang-group-draft-backend` | `SGLANG_GROUP_DRAFT_BACKEND` | `transformers` | `transformers` 或 `sglang`。 |
| `--sglang-group-draft-device` | `SGLANG_GROUP_DRAFT_DEVICE` | target CUDA device | Transformers draft 模型所在设备。 |
| `--sglang-group-draft-device-map` | `SGLANG_GROUP_DRAFT_DEVICE_MAP` | 未设置 | 传给 HF `from_pretrained(..., device_map=...)`。 |
| `--sglang-group-draft-dtype` | `SGLANG_GROUP_DRAFT_DTYPE` | `auto` | `auto`、`fp16`、`bf16` 或 `fp32`。 |
| `--sglang-group-native-draft-quantization` | `SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION` | 未设置 | backend 为 `sglang` 时可选的 draft 量化覆盖。 |
| `--sglang-group-native-draft-cache-tokens` | `SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS` | 自动推导 | backend 为 `sglang` 时的 draft KV pool token 数。 |
| `--sglang-group-native-draft-max-requests` | `SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS` | `1` | backend 为 `sglang` 时的 draft request pool 大小。 |
| `--sglang-group-max-draft-tokens` | `SGLANG_GROUP_MAX_DRAFT_TOKENS` | 自动推导 | 每次 proposal 的最大 draft 自回归步数。 |
| `--sglang-group-max-context-tokens` | `SGLANG_GROUP_MAX_CONTEXT_TOKENS` | 未设置 | proposal 前截断 draft-side context。 |
| `--sglang-group-dtw-window` | `SGLANG_GROUP_DTW_WINDOW` | `8` | `itl` 对齐诊断使用的 DTW window。 |
| `--sglang-group-assistant-lookbehind` | `SGLANG_GROUP_ASSISTANT_LOOKBEHIND` | `10` | SLEM assistant-side lookbehind。 |
| `--sglang-group-target-lookbehind` | `SGLANG_GROUP_TARGET_LOOKBEHIND` | `10` | SLEM target-side lookbehind。 |
| `--sglang-group-max-cached-requests` | `SGLANG_GROUP_MAX_CACHED_REQUESTS` | `256` | Transformers backend 的 per-request draft KV cache 数。 |
| `--no-sglang-group-draft-cache` | `SGLANG_GROUP_ENABLE_DRAFT_CACHE=false` | enabled | 关闭 draft KV cache，用于诊断。 |
| `--no-sglang-group-cache-clone` | `SGLANG_GROUP_CLONE_DRAFT_CACHE=false` | enabled | 关闭保守 cache clone。 |
| `--sglang-group-tli-min-intersection` | `SGLANG_GROUP_TLI_MIN_INTERSECTION` | `1` | TLI 最小共享 token 数。 |
| `--sglang-group-metrics-log-interval` | `SGLANG_GROUP_METRICS_LOG_INTERVAL` | `60` | worker metrics 日志间隔；`0` 表示关闭。 |

## 推荐默认值

基于当前 MiniMax-M2.7-AWQ/NVFP4 的测试结果，可以先按下面方式试：

- `temperature=0`：优先 `itl-base-slem`。
- `temperature=0.6, top_p=0.95`：优先 `itl-base-tli`。
- `temperature=1`：优先 `itl`。
- `--speculative-num-draft-tokens` 先用 `5`，再对比 `3`、`5`、`7`。
- 正确性确认后保持 draft cache 开启。
- 如果使用 `--sglang-group-draft-backend sglang`，高并发早期测试建议设置 `--sglang-group-max-context-tokens`，例如 `4096` 或 `8192`，避免跨请求 rebuild 成本过高。

## 限制

- 只适配 SGLang 0.5.9。
- 需要 `--disable-overlap-schedule`；legacy mode 下 wrapper 会自动添加。
- 暂不支持 pipeline parallelism。
- 暂不支持 DP attention。
- 每个请求使用一条线性 candidate chain。
- 多模态请求会对该请求 fallback 到 target-only verification。
- `itl-base-slem` 只支持 greedy。
- `--sglang-group-draft-backend sglang` 不支持 HF `device_map`。
- 当前 SGLang-native draft cache 是单 active request cache，尚不是多请求 LRU cache。

## 开发检查

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall sglang_group tests
python -m build
```
