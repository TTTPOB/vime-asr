# vime-asr

面向单用户的 Qwen3-ASR 0.6B CPU 真流式后端，适配 [vime](https://github.com/TTTPOB/vime) 的 WebSocket 转写协议。

- 常驻模型，边接收音频边计算并发送增量文字。
- Decoder 逐 token 运算使用 Q4_K body / Q8_0 head；编码器、embedding lookup 和多 token prefill 保持原精度。
- 量化由离线脚本生成普通权重文件，服务只负责读取。
- 一次只处理一个录音任务，不保存音频或转写文字，不包含 LLM 清洗。

## 安装发布包

[Releases](https://github.com/TTTPOB/vime-asr/releases) 提供 `vime-asr-linux-x86_64.tar.gz`。预编译环境要求：Linux x86_64、AVX2/FMA、glibc ≥ 2.39（例如 Ubuntu 24.04）、系统 OpenBLAS 和 [uv](https://docs.astral.sh/uv/)。

```bash
sudo apt-get install libopenblas0
# 解压下载的发布包后，在 vime-asr/ 中执行：
uv sync --locked
```

发布包包含 Python 服务、native 库及所需 GGML CPU 库。模型权重和 API token 由使用者单独准备。

## 准备模型和量化权重

1. 下载官方 [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) 至 `models/qwen3-asr-0.6b/`：

   ```bash
   uvx --from huggingface-hub hf download Qwen/Qwen3-ASR-0.6B \
     --local-dir models/qwen3-asr-0.6b
   ```

2. 生成 Decoder AR 权重一次：

   ```bash
   uv run python -m vime_asr.quantize \
     --model-dir models/qwen3-asr-0.6b \
     --output models/qwen3-asr-0.6b/decoder-ar-q4k.weights
   ```

   脚本直接写出普通文件，退出成功后再部署该文件。默认是 Q4_K body / Q8_0 head；`--body q8_0` 可生成较保守的全 Q8 AR 版本。

3. 保留原始模型与生成的 AR 权重。原始 BF16 文件仍用于编码器、embedding lookup 和 prefill，AR 文件不是完整模型的替代品。

权重格式为固定 x86_64 / GGML 布局。更换原模型或量化实现时重新执行转换脚本；服务不会自动重建权重，不检查原权重时间戳或哈希。加载时只解析文件格式和矩阵形状，文件无法读取会直接报错。

## 启动服务

准备一个只包含一行 API token 的文件，然后运行：

```bash
uv run python -m vime_asr.server \
  --host 127.0.0.1 --port 18766 \
  --model-dir models/qwen3-asr-0.6b \
  --ar-weights models/qwen3-asr-0.6b/decoder-ar-q4k.weights \
  --token-file /path/to/asr-token
```

省略 `--ar-weights` 时使用原 BF16 运算，不在启动时量化。默认 native 和 BLAS 各 4 线程。

客户端连接 `ws://127.0.0.1:18766/v1/realtime`，发送 `Authorization: Bearer <token>`。远程使用可在服务前配置支持 WebSocket 的 TLS 反向代理，客户端连接对应 `wss://` 地址。

`GET /health` 匿名返回 `ready` / `busy`。健康端点在模型加载完成后可用。API token、音频和文字不会写入服务日志。

## 流式行为

- 输入为 24 kHz、mono、PCM16LE，通过 soxr 连续重采样为 16 kHz float32。
- 2 秒计算步长、8 秒编码窗口、4 个已完成窗口缓存、5 token rollback。
- 默认 `unfixed_chunks=0`，尽早交付已稳定的前缀；`--unfixed-chunks 1` 或 `2` 可延后初始输出。
- 收到 commit 后，有稳定文本时一次最多合并 16 秒积压音频，并同步增加生成预算；较长积压继续分步消费。
- UTF-8 增量解码处理跨 token 中文字节；全部 delta 发送后再交付完整 final。

这条路径调用 `qwen_transcribe_stream_live`，不是收到完整录音后才启动离线转写。编码器会重算当前未完成窗口，Decoder 使用前缀回滚与 KV 复用，因此不是完全因果模型。

clear 或断线会丢弃迟到结果。运行时没有即时 native abort，识别名额在已有计算退出后释放；忙时返回错误，不排队。

## 协议

实现 [vime 协议](https://github.com/TTTPOB/vime/blob/master/docs/PROTOCOL.md) 的转写子集：

1. `session.update`：嵌套 `session.type=transcription`、`audio.input.format={type: audio/pcm, rate: 24000}`、`turn_detection=null`。
2. `session.updated` 回显会话。
3. `input_audio_buffer.append`：base64 PCM16LE。
4. 录音中发送 `conversation.item.input_audio_transcription.delta`。
5. `input_audio_buffer.commit` → `input_audio_buffer.committed` → `conversation.item.input_audio_transcription.completed`。
6. delta、commit ACK 与 completed 使用同一 `item_id`，最终 `transcript` 为准。

这不是完整的 OpenAI Realtime API。

## 从源码构建

```bash
sudo apt-get install gcc g++ git curl patch libopenblas-dev
uv sync --locked
bash scripts/build-native.sh
uv run python -m unittest discover -s tests -v
```

构建固定使用：

- [antirez/qwen-asr](https://github.com/antirez/qwen-asr)，commit `924694251d9e0f18e5d86bbd06aa3ab5f870002d`。
- [transcribe.cpp v0.4.0](https://github.com/handy-computer/transcribe.cpp/releases/tag/v0.4.0) 的 GGML base / haswell CPU 库和匹配头文件。

输出为 `build/native/libvime_qwen.so` 与 `build/lib/`。GGML CPU traits 属于内部接口，更新时应同时更换匹配头文件和库，验证量化权重格式及转写。编译脚本修补上游源码副本，不改上游 checkout；新共享库完成后替换，运行中的进程继续使用旧映射，重启后加载新版本。

## CI 与发布

main push、PR 和手动触发会运行协议测试、构建 native 库并打包。推送 `v*` tag 会自动创建 Release 并上传 Linux 运行包：

```bash
git tag v0.1.0
git push origin v0.1.0
```

发布包不包含模型、生成的 AR 权重、凭据、录音或构建工作树。CI 的 fake-inference 测试验证协议，不衡量模型准确率；真实音频仍需在目标机器验收。

## 性能参考

Xeon E3-1220 v6、4 核、AVX2/FMA、4 个 native/BLAS 线程，同组音频每段两轮。下表为模型已加载后的离线推理均值，单位秒：

| 音频 | 原 BF16 | Q4_K body / Q8 head |
|---|---:|---:|
| 中文 5s | 2.465 | 1.733 |
| 中文 15s | 5.269 | 3.842 |
| 中文 30s | 11.839 | 7.853 |
| 中文 45s | 18.550 | 13.153 |
| 英文约 11s | 3.385 | 2.431 |

离线生成 AR 权重约 46.3 秒，产物约 394 MiB；随后两个新进程加载原模型与 AR 文件约 0.724 / 0.716 秒（OS 页缓存热，未测冷盘）。中文有少量用字与标点差异，英文样本保持原文；样本有限，没有报告整体准确率或 P95。

流式策略单独测试中，原 BF16 的首 delta 从约 8 秒降至约 3.4 秒，30 秒录音停止后的等待从约 10–11 秒降至约 4.7 秒。它与量化方案组合后的时延需要端到端测量，不能把两组收益直接相加。

## 许可

本项目使用 MIT，见 [LICENSE](LICENSE)。依赖声明见 [GGML vendor 目录](native/vendor/ggml/README.md)，包括 GGML、transcribe.cpp 和实际 CPU 库包含的 Mozilla llamafile MIT 声明。Qwen 模型遵循其上游模型许可，不随此仓库分发。
