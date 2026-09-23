# Live Translator · 实时音频翻译

Windows / macOS 桌面应用：把**某个应用（或整个系统 / 麦克风）正在播放的声音**实时识别并翻译成字幕。
参考 [jt-live-whisper](jt-live-whisper/) 的思路重新设计，把它没有的"按应用捕获"做了进来，并把延迟压到"说完一句话，约 1 秒内出译文"。

| 你的需求 | 实现 |
|---|---|
| 1. 本地 / 远程识别 | 本地 **faster-whisper**（自动用 CUDA，否则 CPU int8）；远程走任何 OpenAI 兼容的 `/audio/transcriptions`（OpenAI、Groq、SiliconFlow、自建 faster-whisper-server …） |
| 2. 指定应用的音频 | Windows：WASAPI **Process Loopback**（含子进程，Chrome / Edge / Electron / Zoom 都行；同一应用**独立开了多个实例**时会分开列出，可只捕获其中一个）；macOS：**ScreenCaptureKit** 按应用过滤。另有"系统全部声音""麦克风" |
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
  - **悬浮字幕**：字号、同时显示句数（1 = 只显示当前一句，默认 2，最多 8）、自动调整高度、背景不透明度、是否显示原文、鼠标穿透；**学习模式**开关、注音标在译文 / 原文 / 两者、朗读内容（译文 / 原文）。
  - **高级**：说话人区分的相似度阈值与人数上限；语音检测阈值、句末停顿、单句最长、中间结果间隔、束宽、计算精度（选择主界面的“速度 / 准确度”预设会覆盖其中的断句与束宽数值）。
- **视图**：悬浮字幕 / 鼠标穿透 / 显示原文 / 学习模式（与工具栏同步）。**帮助 → 关于**：显示识别设备、模型与配置目录。

「悬浮字幕」会在所有窗口上方显示可拖动的字幕条（右键：字号、是否显示原文、同时显示句数、自动调整高度、学习模式；勾选「鼠标穿透」后不挡操作）。
「同时显示句数」是屏幕上同时留几**句**：设为 2 以上时，前面的句子（颜色稍暗）留在当前句上方，当前句还在流式识别时也不会把它顶掉，超过句数时最旧的先消失；1 就只显示当前这一句。每句都完整显示，可以折成多行。
高度默认自动（刚好装下内容，向上长，底边不动）；**上下拖右下角**就会固定成你拖出来的高度（此时最新一句贴着底边，放不下的旧字幕从顶部裁掉），想恢复自动就在右键菜单 / 偏好设置里勾上「自动调整高度」。

### 区分说话人

主界面 ⑤ 勾选「区分说话人」：每句话取一个声纹（3D-Speaker **CAM++** 中英文模型，ONNX，约 28 MB，首次使用自动下载到模型目录，本地运行、每句约 40 ms），
与之前听到的人比较，像就沿用编号，不像就新增：字幕历史里每条前有彩色的「说话人 N」，悬浮字幕在出现第二个人之后给每句加上同色的 `[N]`，字幕记录文件里也会写上。
每次点「开始」编号重新从 1 计；换了一批人可以用 文件 → 重置说话人编号。模型固定到指定版本并校验 SHA-256，文件不符会拒绝加载。

**为什么用 CAM++、背景音乐下表现如何**：实测过 5 个候选模型。用真实人声（AISHELL-1 中文 8 位说话人，LibriSpeech 英文 9 位）和真实音乐（GTZAN）按不同信噪比混合，
上一版用的 WeSpeaker ResNet34 在背景音乐和人声一样响（0 dB）时把 8 个人拆成了约 39 个，5 dB 时约 19 个；CAM++ 在同样条件下人数仍是 8–9 个，
说话人是否属于同一人的判别错误率（EER）0 dB 时约 1%（WeSpeaker 13.5%）。评测脚本用的是朗读语音，真实直播 / 会议的口语、抢话、回声会比这难，绝对数字仅供参考。

它做的是**声纹聚类**，不是身份识别，几个已知的局限：
- 句子越短越不可靠：不到 1 秒沿用上一位；1–2 秒只归入已认识的人（不会新增、也不会改动已有声纹）；2 秒以上才可能新增说话人。
- 多人抢话、比人声还响的背景音乐（−5 dB 以下）、带歌词的人声 BGM 会让声纹变糊；BGM 里的歌声被识别成文字时也会被当成一个“说话人”。
- 声音很像的两个人可能被并成一个，同一个人情绪差别大时可能被拆成两个——这两种错误方向相反，可在偏好设置 → 高级里调「相似度阈值」（默认 0.55；调高更容易拆开，调低更容易合并）或限定最多人数。
- 标签是按发言当时的判断打的，之后不会回头修改。

### 学习模式

工具栏勾选「学习模式」（或右键字幕条 / 偏好设置里开启）：

- **注音**：中文的每个汉字上方标带声调的拼音（多音字按上下文判断），日文的汉字上方标平假名（送假名不标，如 取り組み → と・く），韩文每个词上方标按实际发音的罗马字（如 감사합니다 → gamsahamnida），英、法、德、西、葡、意、俄等其他语言每个词上方标 IPA 国际音标（泰语、越南语、菲律宾语暂不支持）。「注音标在」可选译文 / 原文 / 两者（各按文本自己的语言标注），悬浮字幕和主窗口的字幕历史都会显示；依赖 `pypinyin`、`pykakasi` 与 `espeakng-loader`（已在 requirements.txt 中，缺哪个就缺哪种语言的注音；韩文罗马字无需额外依赖）。日文读音与 IPA 来自词典 / 规则，个别词可能不准；IPA 按单词逐个标注，不含连读。
- **朗读**：字幕前出现一个小喇叭，点击朗读这句，再点一次停止；主窗口每条字幕前也有，右键字幕还可复制原文 / 译文。
  朗读内容可选译文或原文，使用系统自带的语音（Windows：设置 → 时间和语言 → 语音，需装有对应语言的语音包，缺少时会在状态栏提示）。
  「鼠标穿透」开启时悬浮字幕收不到点击，喇叭会被隐藏，主窗口里的仍可用。
  **没有对应语言的语音时**会有三层提示：打开学习模式 / 点「开始」时弹出一个不挡字幕的窗口（每种语言每次运行只弹一次，说明去哪里装）；状态栏显示同样的说明；悬浮字幕上的小喇叭被划掉，鼠标停上去看原因。装好语音后需重启本程序。

主窗口字幕历史滚动到上面查看旧字幕时，新字幕不会再把画面拉走；滚回最底部才恢复自动跟随。

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

## 安装包与打包

不想装 Python？直接下载打包好的程序（[Releases](../../releases)，由 GitHub Actions 构建）：

| 系统 | 文件 | 运行 |
|---|---|---|
| Windows 10 2004+ / 11（x64） | `LiveTranslator-<版本>-windows-x64.zip` | 解压后双击 `LiveTranslator.exe`；`live-translator-cli.exe` 是同一程序的控制台版（`--cli`、`--list-apps`） |
| Windows + NVIDIA 显卡 | `…-windows-x64-cuda.zip` | 同上，已内置 cuBLAS / cuDNN，GPU 开箱即用（体积约大 1 GB） |
| macOS 13+（Apple 芯片） | `LiveTranslator-<版本>-macos-arm64.dmg` | 拖进「应用程序」。应用未经 Apple 公证，首次打开请右键 →「打开」，或执行 `xattr -dr com.apple.quarantine /Applications/LiveTranslator.app`。屏幕录制权限授予 **Live Translator** 本身；音频捕获组件已预编译，无需 Xcode |
| Linux（x64） | `LiveTranslator-<版本>-linux-x64.tar.gz` | `tar xzf` 后运行 `LiveTranslator/LiveTranslator`。仅支持麦克风 / 文件 / 远程识别（暂无系统与按应用捕获）。需要系统库：`sudo apt install libportaudio2 libxcb-cursor0 libxkbcommon-x11-0 libegl1`（PortAudio 用于麦克风，其余是 Qt 界面所需） |

> 安装包不含 Whisper / NLLB 模型，首次使用时按需下载到模型目录（可在「偏好设置 → 存储」修改）。
> 没有代码签名：Windows 可能出现 SmartScreen 提示，选「更多信息 → 仍要运行」。

### 本地打包

PyInstaller 不能交叉编译，要给哪个系统打包就在哪个系统上运行（需要 Python 3.10+，macOS 还需要 `xcode-select --install`）：

```bash
python scripts/build.py                  # 新建 .venv-build → 装依赖 → 打包 → 冒烟测试 → 生成压缩包，产物在 dist/
python scripts/build.py --current-env    # 直接用当前 Python 环境（依赖已装好时更快）
python scripts/build.py --cuda           # 仅 Windows：连同 NVIDIA cuBLAS / cuDNN 一起打包
python scripts/build.py --help           # 其它选项：--no-test  --no-archive  --version
```

产物：Windows `dist/LiveTranslator/` + `.zip`；macOS `dist/LiveTranslator.app` + `.dmg`；Linux `dist/LiveTranslator/` + `.tar.gz`。
构建结束前会运行 `--self-test`（导入全部原生依赖、加载并运行一次 Silero VAD、创建 Qt 窗口类），打包遗漏文件会在这里直接报错。
打包配置在 [packaging/live_translator.spec](packaging/live_translator.spec)，入口是 [packaging/entry.py](packaging/entry.py)。
无控制台的 GUI 版把日志写到配置目录下的 `live-translator.log`。

### GitHub Actions

[.github/workflows/build.yml](.github/workflows/build.yml)：

- 每次 push / PR：运行测试（Ubuntu），并在 Windows / macOS / Linux 上各打一份包，作为 Actions 构建产物保留 14 天。
- **发布**：推送版本标签即自动构建并创建 Release，附上全部安装包（含 Windows CUDA 版）：

  ```bash
  git tag v0.1.0 && git push origin v0.1.0
  ```

- 手动运行（Actions → Build → Run workflow）可勾选「Also build the Windows CUDA variant」。

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
  ui/                主窗口 · 设置面板 · 字幕历史 · 悬浮字幕 · ruby_text.py（带注音的换行文字）· tts.py（朗读）
  ruby.py            学习模式的分词与注音：拼音 / 假名 / 韩文罗马字 / IPA（纯逻辑）
  native/macos/      lt_sck_capture.swift
tests/               单元 + 管线 + GUI 测试（含本地假 OpenAI 服务器）
packaging/           PyInstaller 入口与 spec
scripts/build.py     本地打包脚本（CI 也用它）
.github/workflows/   测试 + 三平台打包 + 发布
```

```bash
.venv/Scripts/python -m pytest tests -q
```

## 已知限制

- 本地大模型识别在纯 CPU 上无法做到实时，请用 `small` 及以下，或改用远程识别。
- 依赖停顿断句：语速极快、几乎不停顿的演讲会被切成较长的句子，延迟随之变大（"速度优先"预设可缓解）。
- 受 DRM 保护的流媒体，部分应用会让系统屏蔽 loopback 捕获（得到静音）。
- **分不开同一个浏览器里的窗口 / 标签页**：Chrome、Edge 把所有标签页的声音都交给同一个“音频服务”子进程输出，操作系统层面只有一个声源，按进程捕获只能整体要或整体不要。
  只想要其中一个时：在浏览器里把其它标签页「静音网站」（静音后不会进入捕获），或用 `chrome --user-data-dir=<另一个目录>` 另开一个独立实例——它是单独的进程树，会作为单独一项列出。
  （同一个 Chrome 里的多个“用户资料”共用一个浏览器进程，同样分不开。）
- 说话人区分是声纹聚类，短句、抢话、背景音乐会影响准确度，详见上文「区分说话人」。
- API Key 以明文保存在用户配置目录的 `config.json` 中（Windows 下为 `%APPDATA%\LiveTranslator`）。
