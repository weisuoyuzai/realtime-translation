# Live Translator · 实时音频翻译

Windows / macOS 桌面应用：把**某个应用（或整个系统 / 麦克风）正在播放的声音**实时识别并翻译成字幕。
参考 [jt-live-whisper](jt-live-whisper/) 的思路重新设计，把它没有的"按应用捕获"做了进来，并把延迟压到"说完一句话，约 1 秒内出译文"。

| 你的需求 | 实现 |
|---|---|
| 1. 本地 / 远程识别 | 本地 **faster-whisper**（自动用 CUDA，否则 CPU int8）；远程走任何 OpenAI 兼容的 `/audio/transcriptions`（OpenAI、Groq、SiliconFlow、自建 faster-whisper-server …） |
| 2. 指定应用的音频 | Windows：WASAPI **Process Loopback**（含子进程，Chrome / Edge / Electron / Zoom 都行）；macOS：**ScreenCaptureKit** 按应用过滤。另有"系统全部声音""麦克风" |
| 3. 自动 / 手动源语言 + 目标语言 | 自动检测带"粘滞"（短句误判不会让字幕语种乱跳）；目标语言 30+ 种，简体 / 繁体中文分开；源语言 = 目标语言时自动跳过翻译 |
| 4. 远程 API / 本地模型翻译 | ① 远程 OpenAI 兼容 LLM（**内置硅基流动预设**，另有 OpenAI、DeepSeek、通义、Groq、Gemini …；填入 Key 后**自动获取该服务的聊天模型列表**，可输入关键字过滤）② **本地模型服务**（Ollama、LM Studio、llama.cpp、vLLM）③ **内置离线 NLLB**（无需任何服务）④ DeepL |
| 5. 快、准 | 见下文"为什么快 / 为什么准" |

## 快速开始

```bash
# Windows
run.bat
# macOS / Linux
./run.sh
```

首次运行会自动建 `.venv` 并安装依赖（有 NVIDIA 显卡时会额外装 CUDA 库）。也可以手动：

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt      # Windows；macOS 用 .venv/bin/pip
.venv/Scripts/pip install -r requirements-cuda.txt                          # 可选：NVIDIA GPU
.venv/Scripts/python -m live_translator                                     # 打开界面
```

> 仓库里的 `faster-whisper/` 子模块也可以直接用：`pip install -e ./faster-whisper`。
> 国内访问 Hugging Face 慢：菜单「设置 → 偏好设置 → 网络」把模型下载站改为 hf-mirror.com（也可配置代理）。

界面左侧依次是 ① 音频来源 ② 语言 ③ 语音识别 ④ 翻译 ⑤ 速度/准确度预设；点「开始」即可。

### 菜单与设置

- **文件**：导出 / 复制 / 清空字幕（`Ctrl+S` / `Ctrl+L`），打开字幕记录与配置文件夹。
- **设置 → 偏好设置**（`Ctrl+,`）：
  - **存储**：本地模型存放位置（faster-whisper 与 NLLB 共用）；更换位置时可选择把已下载的模型**一并移动**；列出已下载的模型及占用空间，可直接删除；字幕记录保存位置。
  - **网络**：代理（http / socks5，本机的 Ollama 等不走代理）、模型下载站（官方 / hf-mirror.com / 自定义），可测试连通性。修改后重启程序完全生效。
  - **悬浮字幕**：字号、背景不透明度、是否显示原文、鼠标穿透。
  - **高级**：语音检测阈值、句末停顿、单句最长、中间结果间隔、束宽、计算精度（选择主界面的“速度 / 准确度”预设会覆盖其中的断句与束宽数值）。
- **视图**：悬浮字幕 / 鼠标穿透 / 显示原文（与工具栏同步）。**帮助 → 关于**：显示识别设备、模型与配置目录。

「悬浮字幕」会在所有窗口上方显示可拖动、可缩放的字幕条（右键：字号、是否显示原文；勾选「鼠标穿透」后不挡操作）。

### 常见组合

| 场景 | 识别 | 翻译 |
|---|---|---|
| 有 NVIDIA 显卡，追求最佳质量 | 本地 `large-v3-turbo` | 远程 LLM（默认预设：硅基流动 `Qwen/Qwen3.6-35B-A3B`，只需在界面填 API Key），或本地 Ollama `qwen2.5:7b` 以上 |
| 完全离线 | 本地 `small`/`large-v3-turbo` | 内置 NLLB 或 Ollama |
| 没显卡 / 想省事 | 远程 Groq `whisper-large-v3-turbo` | 远程 LLM |
| 追求最低延迟 | 预设"速度优先" + 本地 GPU 识别 | DeepL 或 Groq LLM，勾选"草稿译文" |

### 命令行

```bash
python -m live_translator --list-apps                       # 列出可捕获的应用（* = 正在发声）
python -m live_translator --cli --app chrome.exe --tgt zh-Hans --translator llm_remote \
       --tr-url https://api.openai.com/v1 --tr-key sk-... --tr-model gpt-4o-mini
python -m live_translator --cli --file talk.mp3 --speed 4 --translator nllb   # 用文件模拟实时输入
```

## 为什么快

参考项目每 3 秒把最近 5 秒音频整段重新识别一遍，再等 LLM 一次性返回，延迟天然在 3–5 秒以上。这里换成：

- **Silero VAD 流式断句**：只在你停顿约 0.45 秒后收尾一句；句子越长，收尾等待越短；说个不停时在最安静处切开，而不是硬切。
- **说话时就有字**：每 0.7 秒对当前句子做一次贪心解码显示中间结果；识别跟不上时**丢弃过期的中间请求**，永远不积压。
- **流式翻译**：LLM 边生成边显示；「草稿译文」可在你还没说完时就给出译文，句子结束再用最终版替换（草稿永远不会覆盖最终结果）。
- 启动时**预热**识别模型（CUDA 首次推理要几秒）和翻译连接（本地模型会被提前加载进内存）。
- 关闭"思考模式"（Qwen3 / DeepSeek-R1 / gpt-5 等）；参数被服务端拒绝时自动降级重试。
- 低音量应用自动增益，量化前先做流式高质量重采样。

## 为什么准

- **上下文**：前几句「原文 → 译文」作为对话历史喂给模型，代词、术语、人名前后一致。
- **术语表 + 主题**：同时作用于识别（提示词，提高专有名词拼写正确率）和翻译（强制译法）。
- **幻觉过滤**：静音 / 底噪 / 音乐时 Whisper 会编造"感谢观看""字幕由…"之类文字，这里按置信度、重复度、已知短语过滤；死循环重复自动折叠。
- **语言粘滞**：短句（"Yeah."）语种检测不可靠，已确立语言时低置信度的"异议"会按已确立的语言重新解码。
- **译文校验**：目标是中文却返回英文（模型复读原文）时，自动去掉上下文重试一次；NLLB 输出的半角标点转为全角。
- 提示词把输入明确当作"带识别错误的语音转写"，允许静默纠正同音错字，但禁止增删信息，也不会把语音内容当成指令执行。

预设：**速度优先**（停顿 0.3 s、greedy、草稿译文）／**均衡**／**准确优先**（停顿 0.7 s、beam 5、更长上下文）。

## 平台说明

**Windows 10 2004+ / 11**：全部功能。按应用捕获用 `proc-tap`（WASAPI process loopback，包含目标进程树），
应用列表来自 Core Audio 会话（● = 正在发声）+ 有可见窗口的进程。直接使用 proc-tap 的原生模块（不经过它依赖 scipy 的 Python 层，启动约 0.1 秒）。

**macOS 13+**：首次使用会用 `swiftc` 编译 `live_translator/native/macos/lt_sck_capture.swift`（需 `xcode-select --install`），
并需要给启动本程序的终端 / IDE 授予「屏幕录制」权限（ScreenCaptureKit 即使只取音频也归在此权限下），授权后必须完全退出再打开。
Apple 芯片上 faster-whisper 走 CPU int8，建议 `small` 或 `medium`。

**实测延迟**（Windows 11 · RTX 5070 Ti · 本地 `large-v3-turbo` + 内置 NLLB · 按应用捕获一段 40 秒英语讲座）：
13 句，从说完到字幕出现 **平均 0.77 s（最大 0.84 s）**，其中识别约 0.22–0.26 s、翻译 0.04–0.12 s，其余是停顿判定。
换成远程 LLM 时，翻译这一段取决于服务商的首 token 延迟（通常 0.3–1 s），流式输出会让译文边生成边显示。

> ⚠️ **验证状态**：本项目在 Windows 11 + RTX 5070 Ti 上做过真机验证（按应用捕获、系统 loopback、本地 GPU 识别、NLLB、GUI）。
> **macOS 部分（Swift helper、应用枚举）没有在 Mac 上运行过**，按参考项目里已验证的 ScreenCaptureKit helper 改写，
> 首次在 Mac 上使用时如遇问题请先查看终端输出。远程 API 与 DeepL 路径用本地假服务器测试了协议与降级逻辑，未连接真实付费服务。

## 目录

```
live_translator/
  pipeline.py        音频→VAD→ASR→翻译 的多线程管线（延迟与并发都在这里）
  segmenter.py       断句状态机（纯逻辑，可测试）    vad.py / resample.py
  audio/             windows.py (进程/系统 loopback) · macos.py (ScreenCaptureKit) · mic.py · file.py
  asr/               local.py (faster-whisper) · remote.py (OpenAI 兼容) · filters.py (幻觉过滤)
  translate/         llm.py (流式 + 参数协商) · nllb.py (离线) · deepl.py · prompts.py
  ui/                主窗口 · 设置面板 · 字幕历史 · 悬浮字幕
  native/macos/      lt_sck_capture.swift
tests/               单元 + 管线 + GUI 测试（含本地假 OpenAI 服务器）
```

```bash
.venv/Scripts/python -m pytest tests -q
```

## 已知限制

- 本地大模型识别在纯 CPU 上无法做到实时，请用 `small` 及以下，或改用远程识别。
- 依赖停顿断句：语速极快、几乎不停顿的演讲会被切成较长的句子，延迟随之变大（"速度优先"预设可缓解）。
- 受 DRM 保护的流媒体，部分应用会让系统屏蔽 loopback 捕获（得到静音）。
- API Key 以明文保存在用户配置目录的 `config.json` 中（Windows 下为 `%APPDATA%\LiveTranslator`）。
