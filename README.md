# sglang-group

中文 | [English](README_EN.md)

`sglang-group` 是面向 **SGLang 0.5.9** 的异构词表 speculative decoding 集成。它支持 SLEM、TLI 和 TokenTiming-style ITL 三类异构词表 proposal 方法，让 **target 模型和 draft 模型都可以由 SGLang 执行**，同时允许两者使用不同 tokenizer。

普通情况下请使用 **SGLang-native draft backend**：

```bash
--sglang-group-draft-backend sglang
```

这表示：

- target 模型由 SGLang 原生 serving 路径加载和验证。
- draft 模型也由 SGLang 0.5.9 的 low-level `ModelRunner` 加载和 decode。
- draft 模型不需要是专门为 target 训练的 MTP/EAGLE/P-EAGLE；普通 causal LM 也可以作为 draft。
- target tokenizer 和 draft tokenizer 可以不同。

Transformers draft backend 只建议作为兼容、调试或对照实验使用，见后面的“兼容模式”章节。

## 安装

推荐使用 `uv`：

```bash
uv pip install "sglang-group[sglang] @ git+https://github.com/Huifu1018/sglang-group.git"
```

也可以使用 `pip`：

```bash
pip install "sglang-group[sglang] @ git+https://github.com/Huifu1018/sglang-group.git"
```

`sglang` extra 会安装并固定：

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

## 推荐：安装 SGLang 源码级集成

为了避免 `NGRAM` rewrite 和 monkey patch，生产环境建议先把 `SGLANG_GROUP`
安装进当前 Python 环境里的 SGLang 0.5.9 源码：

```bash
sglang-group-install-sglang-patch
sglang-group-install-sglang-patch --check
```

这个命令会修改当前环境中的两个 SGLang 文件：

- `sglang/srt/speculative/spec_info.py`
- `sglang/srt/server_args.py`

原文件会保留 `.sglang-group.bak` 备份。打完补丁后，SGLang 会原生接受
`--speculative-algorithm SGLANG_GROUP`，`sglang-group-launch` 不再把它改写成
`NGRAM`，也不再需要 scheduler 子进程 monkey patch。

如果你使用的是 SGLang 源码目录或镜像构建阶段，可以显式指定路径：

```bash
sglang-group-install-sglang-patch --sglang-root /path/to/sglang
```

## 生产推荐用法：Target + Draft 都走 SGLang

推荐先执行上一节的源码级集成，然后使用 `sglang-group-launch` 启动。这个
launcher 仍然负责消费 `--sglang-group-*` 参数并转成环境变量，但在源码级集成
已经安装时，会直接保留 `--speculative-algorithm SGLANG_GROUP`，不再走
`NGRAM` 兼容路径。

下面是推荐的标准启动方式：

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

这条命令中：

- `--model-path` 是 target 模型，由 SGLang 正常加载。
- `--speculative-draft-model-path` 是 draft 模型，也由 SGLang-native draft backend 加载。
- `--sglang-group-draft-backend sglang` 是关键参数，表示 draft 不走 Transformers，而是走 SGLang。
- `--sglang-group-method auto` 会根据请求温度自动选择 `itl-base-slem`、`itl-base-tli` 或 `itl`。
- `--sglang-group-max-context-tokens 8192` 用于限制 draft-side context，避免长上下文下 draft cache rebuild 成本过高。

源码级集成生效后，日志中的 speculative algorithm 应该是 `SGLANG_GROUP`。
真正生效时还应该看到 `Initialized SGLANG_GROUP worker` 和后续
`SGLANG_GROUP metrics` 日志。

如果没有安装源码级集成，`sglang-group-launch` 会自动回退到旧的兼容模式：
把 `SGLANG_GROUP` 改写到 SGLang 内置 `NGRAM` 解析路径，并在父进程和
scheduler 子进程里安装 patch。此时日志里看到 `NGRAM` 是兼容表现，但仍然
必须看到 `Initialized SGLANG_GROUP worker` 才能确认没有跑成原生 NGRAM。

启动前可以检查环境：

```bash
sglang-group-preflight
sglang-group-preflight --json
```

## SGLang-native Draft Backend 说明

SGLang-native backend 会通过 SGLang 0.5.9 的低层 `ModelRunner` 加载 draft 模型，draft prefill/decode 使用 SGLang KV pool 和模型 kernel。它支持：

- `itl`
- `itl-base-slem`
- `itl-base-tli`
- `auto`

GPU 放置方式遵循普通 SGLang 部署习惯，例如：

```bash
CUDA_VISIBLE_DEVICES=0 sglang-group-launch ...
```

或使用 SGLang 原有并行参数，例如 `--tp`。native backend 不支持 HF `device_map`，因此不要和 `--sglang-group-draft-device-map` 混用。

native backend 不会继承 target 模型的量化配置。例如 target 是 NVFP4/AWQ，而 draft 是普通 BF16/FP16 的 `Qwen/Qwen2.5-1.5B-Instruct`，不要让 draft 误用 target 的量化方式。如果 draft checkpoint 自己声明了量化配置，SGLang 可以识别；否则只在确实需要时显式指定：

```bash
--sglang-group-native-draft-quantization awq
```

默认情况下，SGLang-native draft backend 每轮 proposal 都会重新 prefill 当前
draft-side context。这样会多一些 draft 侧开销，但可以避免 SGLang 内部
`ScheduleBatch` / KV allocator rollback 不完整时出现的重复输出和异常
accept rate。

`SGLANG_GROUP_ENABLE_DRAFT_CACHE` 仍然控制 Transformers backend 的 HF
`past_key_values` cache。对于 SGLang-native backend，accepted-context draft
KV cache 现在是实验功能，需要显式打开：

```bash
--sglang-group-enable-native-draft-kv-cache
```

打开后，行为是：

- 同一个 active request 会保留已接受的 draft 上下文。
- proposal 期间会 snapshot draft SGLang batch。
- speculative draft tokens decode 完后，只回滚 speculative allocator 和 batch 状态。
- 下一轮 proposal 会把已接受 target 文本重新映射后的 draft suffix commit 进 draft cache。

如果打开该实验开关后看到重复输出、`acceptance rate=1.0` 或输出退化，请先关闭
该开关，保留默认安全 rebuild 路径。

并发请求下当前实现是保守的：只保留一个 active draft session，不同 request id 会触发 rebuild。高并发测试时建议保留 `--sglang-group-max-context-tokens`，例如 `4096` 或 `8192`。多请求 LRU native draft cache 可以作为后续优化继续做。

## 方法选择

可以强制指定方法：

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

改写 `auto` 使用的方法：

```bash
--sglang-group-auto-greedy-method itl-base-slem \
--sglang-group-auto-mid-sampling-method itl-base-tli \
--sglang-group-auto-high-sampling-method itl
```

注意：SGLang 0.5.9 中方法选择是 batch-level 的，因为 verify input class 在一个 decode batch 内只会选一次。如果同一个 batch 混合不同 temperature，会用 batch 内最高 temperature 决定方法。

## 常用配置建议

基于 MiniMax-M2.7-AWQ/NVFP4 当前测试结果，可以先按下面方式试：

- `temperature=0`：优先 `itl-base-slem`。
- `temperature=0.6, top_p=0.95`：优先 `itl-base-tli`。
- `temperature=1`：优先 `itl`。
- `--speculative-num-draft-tokens` 先用 `5`，再对比 `3`、`5`、`7`。
- 使用 SGLang-native backend 时，建议先设置 `--sglang-group-max-context-tokens 8192`。

## 兼容模式：Draft 使用 Transformers

这不是默认推荐路径。只有在下面情况才建议使用：

- 需要和早期 HF draft 实现做对照实验。
- SGLang-native draft backend 暂时不支持某个 draft checkpoint。
- 希望使用 HF `device_map` 对 draft 单独放置。
- 排查 SGLang-native draft cache 或显存问题。

启动方式：

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

这个模式下：

- target 模型仍然由 SGLang 跑。
- draft 模型由 Hugging Face Transformers 跑。
- draft cache 使用 HF `past_key_values`。

## 参数列表

`sglang-group-launch` 会消费 `--sglang-group-*` 参数，其余参数继续转发给
SGLang。默认 draft backend 是 `sglang`，也就是 target/draft 都走 SGLang；
如果要做对照实验，可以显式传入 `--sglang-group-draft-backend transformers`。

| 参数 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--sglang-group-method` | `SGLANG_GROUP_METHOD` | `auto` | `auto`、`itl`、`itl-base-slem` 或 `itl-base-tli`。 |
| `--sglang-group-auto-high-temp-threshold` | `SGLANG_GROUP_AUTO_HIGH_TEMP_THRESHOLD` | `0.9` | `auto` 高温路由阈值。 |
| `--sglang-group-draft-backend` | `SGLANG_GROUP_DRAFT_BACKEND` | `sglang` | target/draft 都走 SGLang；`transformers` 表示 draft 走 HF，仅建议用于兼容或对照实验。 |
| `--sglang-group-draft-device` | `SGLANG_GROUP_DRAFT_DEVICE` | target CUDA device | 仅 Transformers backend 使用。 |
| `--sglang-group-draft-device-map` | `SGLANG_GROUP_DRAFT_DEVICE_MAP` | 未设置 | 仅 Transformers backend 使用，传给 HF `from_pretrained(..., device_map=...)`。 |
| `--sglang-group-draft-dtype` | `SGLANG_GROUP_DRAFT_DTYPE` | `auto` | `auto`、`fp16`、`bf16` 或 `fp32`。 |
| `--sglang-group-native-draft-quantization` | `SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION` | 未设置 | SGLang-native draft backend 的 draft 量化覆盖。 |
| `--sglang-group-native-draft-cache-tokens` | `SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS` | 自动推导 | SGLang-native draft KV pool token 数。 |
| `--sglang-group-native-draft-max-requests` | `SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS` | `1` | SGLang-native draft request pool 大小。 |
| `--sglang-group-enable-native-draft-kv-cache` | `SGLANG_GROUP_ENABLE_NATIVE_DRAFT_KV_CACHE=true` | disabled | 实验性开启 SGLang-native accepted-context KV cache；默认关闭以保证正确性。 |
| `--sglang-group-max-draft-tokens` | `SGLANG_GROUP_MAX_DRAFT_TOKENS` | 自动推导 | 每次 proposal 的最大 draft 自回归步数。 |
| `--sglang-group-max-context-tokens` | `SGLANG_GROUP_MAX_CONTEXT_TOKENS` | 未设置 | proposal 前截断 draft-side context。 |
| `--sglang-group-dtw-window` | `SGLANG_GROUP_DTW_WINDOW` | `8` | `itl` 对齐诊断使用的 DTW window。 |
| `--sglang-group-assistant-lookbehind` | `SGLANG_GROUP_ASSISTANT_LOOKBEHIND` | `10` | SLEM assistant-side lookbehind。 |
| `--sglang-group-target-lookbehind` | `SGLANG_GROUP_TARGET_LOOKBEHIND` | `10` | SLEM target-side lookbehind。 |
| `--sglang-group-max-cached-requests` | `SGLANG_GROUP_MAX_CACHED_REQUESTS` | `256` | Transformers backend 的 per-request draft KV cache 数。 |
| `--no-sglang-group-draft-cache` | `SGLANG_GROUP_ENABLE_DRAFT_CACHE=false` | enabled | 关闭 Transformers backend 的 HF draft cache；SGLang-native KV cache 需另行显式开启。 |
| `--no-sglang-group-cache-clone` | `SGLANG_GROUP_CLONE_DRAFT_CACHE=false` | enabled | 关闭 Transformers backend 的保守 cache clone。 |
| `--sglang-group-tli-min-intersection` | `SGLANG_GROUP_TLI_MIN_INTERSECTION` | `1` | TLI 最小共享 token 数。 |
| `--sglang-group-metrics-log-interval` | `SGLANG_GROUP_METRICS_LOG_INTERVAL` | `60` | worker metrics 日志间隔；`0` 表示关闭。 |

## 限制

- 只适配 SGLang 0.5.9。
- 需要 `--disable-overlap-schedule`；legacy mode 下 wrapper 会自动添加。
- 暂不支持 pipeline parallelism。
- 暂不支持 DP attention。
- 每个请求使用一条线性 candidate chain。
- 多模态请求会对该请求 fallback 到 target-only verification。
- `itl-base-slem` 只支持 greedy。
- SGLang-native draft backend 不支持 HF `device_map`。
- 当前 SGLang-native draft KV cache 是实验功能，默认关闭；打开后也是单 active request cache，尚不是多请求 LRU cache。

## 开发检查

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall sglang_group tests
python -m build
```
