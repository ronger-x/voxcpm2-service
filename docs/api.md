# API 与调用示例

[返回项目首页](../README.md) · [部署说明](deployment.md)

本文以 `http://localhost:7860` 为例。请按实际部署地址调整 URL；API key 来自部署目录的 `.env`。

## 快速开始

```bash
curl http://localhost:7860/v1/audio/speech \
  -H "Authorization: Bearer $VOXCPM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "你好，欢迎使用 VoxCPM2。", "response_format": "mp3"}' \
  -o out.mp3
```

API key 写在部署目录的 `.env` 文件中，变量名为 `VOXCPM_API_KEYS`。

## 认证

| 访问方式 | 认证方法 |
|---|---|
| API | 在请求头中带 API key，下面两种写法任选一种 |
| 连续播放页 | 页面可直接打开；生成时填写 API key。key 只保留在当前页面内存中 |
| 网页 | 用账号和密码登录，账号和密码在 `.env` 的 `VOXCPM_USER`、`VOXCPM_PASSWORD` 中 |
| `GET /health` | 不需要认证 |

API key 的两种写法：

```
Authorization: Bearer <API key>
X-API-Key: <API key>
```

key 无效或缺失时，接口返回 `401`。

## 接口说明

### POST /v1/audio/speech

把文本合成为语音。请求体为 JSON。

#### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `input` | string | 必填 | 要合成的文本，不能为空。传了 `segments` 时可以省略 |
| `segments` | array | 无 | 分段文本，每段可带非语言标签、语速 / 音高 / 音量 / 停顿，详见[分段语气](#分段语气)。优先于 `input` |
| `response_format` | string | `wav` | 输出格式：`wav`、`pcm`、`mp3`、`opus`、`aac`、`flac`，详见[输出格式](#输出格式) |
| `sample_rate` | int | `48000` | 输出采样率，可选值见[采样率](#采样率) |
| `speed` | float | `1.0` | 语速，范围 0.25–4.0。只改变速度，不改变音调 |
| `instructions` | string | 无 | 音色或风格描述，例如 `"年轻女性，温柔自然，语速稍慢"` |
| `reference_audio` | string | 无 | 用于克隆音色的参考音频，base64 编码，可以带 `data:audio/wav;base64,` 前缀。优先于 `voice` |
| `reference_text` | string | 无 | `reference_audio` 的逐字文本。提供后改用续写式克隆，音色还原更稳定；文本必须与音频内容一致 |
| `split_sentences` | bool | `true` | 自动按句末标点、换行分句；过长的单句在逗号、空格或 100 字符处拆分。没有参考音频时，固定用首句音频作为后续句子的音色参考 |
| `inference_timesteps` | int | `10` | 推理步数，范围 5–20。步数越高越慢 |
| `stream` | bool | `false` | 设为 `true` 时，接口流式返回原始音频字节 |
| `stream_format` | string | 无 | `audio`：流式返回原始音频字节；`sse`：流式返回 SSE 事件。设置本参数后，即使不写 `stream`，接口也按流式返回 |
| `model` | string | `openbmb/VoxCPM2` | 为兼容 OpenAI 客户端而保留，服务端忽略此参数 |
| `voice` | string | 无 | 预置音色名，对应 `voices/` 目录下的参考音频文件名，详见[预置音色](#预置音色) |

#### 输出格式

| `response_format` | Content-Type | 内容 |
|---|---|---|
| `wav` | `audio/wav` | 16 位 PCM 的 WAV 文件 |
| `pcm` | `audio/pcm` | 裸 PCM，没有文件头：16 位有符号整数、小端序、单声道 |
| `mp3` | `audio/mpeg` | MP3 |
| `opus` | `audio/ogg` | Ogg 容器封装的 Opus |
| `aac` | `audio/aac` | ADTS 封装的 AAC |
| `flac` | `audio/flac` | FLAC |

响应头 `X-Sample-Rate` 返回实际输出的采样率。`pcm` 格式没有文件头，播放时要用这个值。响应头 `X-Voice` 返回实际使用的音色：预置音色名（URL 编码）、`reference_audio` 或 `none`。

#### 采样率

| 格式 | 可选采样率（Hz） |
|---|---|
| `opus` | 8000、12000、16000、24000、48000 |
| 其他格式 | 8000、11025、12000、16000、22050、24000、32000、44100、48000 |

模型原生输出 48000 Hz，其他采样率由服务端重采样得到。

### 预置音色

不传 `reference_audio` 也不传 `instructions` 时，模型每个请求随机生成一个说话人，同一段话多次请求的音色会不同。预置音色用固定的参考音频解决这个问题。

预置音色是 `voices/` 目录（`VOXCPM_VOICES_DIR`）下的 `.wav`、`.flac`、`.mp3`、`.ogg` 文件，文件名（不含扩展名）就是音色名。服务端每次请求都会重新扫描该目录，增删文件不需要重启。

同名的 `.txt`（例如 `default.txt`）是参考音频的逐字文本。存在时服务端把参考音频和文本一起交给模型做续写式克隆（官方的 Ultimate Cloning），音色还原比只用参考音频更稳定；文本必须与音频内容逐字一致，否则会影响发音。没有 `.txt` 时只按参考音频克隆。

参考音频的素材质量直接决定各句音色是否一致。实测用一段 16 kHz 的外部录音作参考时，同一请求内各句的说话人相似度明显波动；改用模型自己按描述生成的参考（`scripts/create_voice.py`）后稳定得多。更换参考后，建议用长文本试听，确认各句音色一致。

服务端按以下顺序决定音色：

| 请求 | 使用的音色 |
|---|---|
| 带 `reference_audio` | 上传的参考音频 |
| `voice` 与预置音色同名 | 该预置音色；同时带 `instructions` 时，描述作为风格控制叠加 |
| `voice` 未匹配，带 `instructions` | 按描述生成，不使用参考音频 |
| `voice` 未匹配或未传，不带 `instructions` | 预置音色 `default` |
| 以上都不满足（没有 `default`） | 随机音色 |

因此只要放一个 `default.wav`，OpenAI 客户端传 `alloy` 等任意音色名时也能得到固定音色。

`scripts/create_voice.py` 按描述生成一段参考音频并保存为预置音色。生成结果每次不同，试听不满意时加 `--force` 重新生成：

```powershell
$env:VOXCPM_API_KEY = '<API key>'
python scripts/create_voice.py default '年轻女性，温柔自然，语速适中' --url http://localhost:7860 --dir voices
```

也可以直接放入录好的人声，5–15 秒、单人、无背景音乐的清晰录音效果最好。

### 分段语气

一段回复里语气有变化时（例如角色先笑着说、再淡淡地说、最后轻声说），用 `segments` 把文本分段，每段可以带三样东西：

```json
{
  "voice": "default",
  "response_format": "pcm",
  "stream_format": "audio",
  "segments": [
    {"text": "[laughing]好、好一则趣话。这断舍离的法子，倒是新颖得紧。", "prosody": {"pause": 0.4}},
    {"text": "我这一向，偏是不做这断舍离的人。", "prosody": {"rate": 0.92}},
    {"text": "你且说说，你眼前有那样该舍的东西么？", "prosody": {"rate": 0.85, "volume": -6, "pause": 0.3}}
  ]
}
```

| 字段 | 说明 |
|---|---|
| `segments[].text` | 本段要朗读的文本，可以带上游的非语言标签（见下文）。空白段落会被跳过；全部为空时返回 `400` |
| `segments[].prosody.rate` | 语速倍数，0.5–2，变速不变调 |
| `segments[].prosody.pitch` | 音高，−12 到 12 个半音，保留共振峰。会改变听感上的音色，见下文 |
| `segments[].prosody.volume` | 音量增益，−30 到 10 dB |
| `segments[].prosody.pause` | 本段开头插入的静音，0–3 秒 |
| `segments[].instructions` | 本段的风格描述，与顶层 `instructions` 用逗号拼接后作为括号控制前缀。效果很弱，见下文 |

**哪些手段听得出来**（用预置音色 `default` 实测，SenseVoice 听写与情绪识别、基频与响度测量）：

| 手段 | 效果 |
|---|---|
| 非语言标签 `[laughing]`、`[sigh]`、`[Dissatisfaction-hnn]` 等 | 稳定发出笑声、叹气、"哼"，SenseVoice 情绪判为 HAPPY / SAD。上游建议少用，一段一个；`[Surprise-wa]` 等会念成"哇"，但情绪变化不明显 |
| `prosody` | 确定性处理，必定生效。实测"轻声慢"（rate 0.82、pitch −2、volume −7）把语速从 5.8 降到 4.8 字/秒、基频从 238 降到 205 Hz、响度降 7.8 dB |
| `instructions` 风格描述 | 克隆音色下几乎无效。"非常小声的耳语，语速非常慢"只让语速变化不到 10%、基频变化不到 10 Hz；调高 cfg 也没有帮助。只有不用参考音色（按描述设计声音）时才明显 |

规则：

- 段内照常按 `split_sentences` 分句。`rate`、`pitch`、`volume` 作用于本段每一句，`pause` 只加在本段第一句前。
- 带 `rate` 或 `pitch` 的句子要整句生成完才能变速变调，流式时这些句子整句发出，首段延迟会变长；只调 `volume` 的句子仍按块流式输出。
- 慎用 `pitch`：保留共振峰也会改变听感上的音色。同一句用 `default` 音色、比较 ECAPA 说话人向量与原样的相似度：重复生成 0.98，`rate` 0.85–0.92 为 0.95–0.96，`volume` −6 为 0.98，`pitch` −1 为 0.96，`pitch` −2 掉到 0.86，试听像换了一个人。相邻段落的音高差不宜超过 1 个半音。
- 音色只由 `voice`、`reference_audio` 和顶层 `instructions` 决定，段落设置不影响音色选择。全部段落在同一个请求里共用一个参考音色，中途不会换声音。
- 只要有一段带 `instructions`，整个请求就不再使用参考文本，改为只按参考音频克隆（上游说明续写式克隆会忽略括号描述）。只用标签和 `prosody` 时仍按续写式克隆，音色最稳定。

### GET /v1/voices

使用与语音接口相同的 API key 认证，返回预置音色列表：

```json
{"object":"list","data":[{"id":"default","object":"voice"}]}
```

### GET /v1/models

使用与语音接口相同的 API key 认证，返回 OpenAI 客户端可读取的模型列表。此接口不加载模型。

```json
{"object":"list","data":[{"id":"openbmb/VoxCPM2","object":"model","created":0,"owned_by":"openbmb"}]}
```

### GET /health

健康检查，不需要认证。服务正常时返回：

```json
{"status": "ok"}
```

## 流式输出

流式模式下，服务端边生成边返回音频。预热后的首段延迟通常低于 1 秒；首次加载模型、编译或请求排队会增加等待时间。所有 6 种格式都支持流式输出。

### 连续播放页

打开 `/player`，填写 API key 和文本，然后点击“生成并播放”。播放器先缓冲约 0.6 秒音频，再从环形缓冲区连续读取 PCM；遇到数据不足时，它会重新缓冲。“停止”会取消网络请求并通知 GPU 工作线程停止生成。完整生成后可下载 WAV。

播放器在 HTTPS 或 localhost 下使用 AudioWorklet，在局域网 HTTP 下使用 ScriptProcessor 兼容路径。兼容路径运行在浏览器主线程，页面繁忙时仍可能影响播放；需要更稳定的播放时，使用 localhost 或 HTTPS。环形缓冲区最多保存 30 秒待播音频，满后暂停读取网络数据；完整音频另保留一份用于下载。

分句默认开启，网页和 API 使用相同的生成流程。上传参考音频时，各句复用该参考；没有上传时，后续各句固定复用首句，避免逐句更换参考导致音色漂移。参考音频在每个请求中只编码一次。将 `split_sentences` 设为 `false` 可关闭分句。

| 写法 | 返回内容 |
|---|---|
| `"stream_format": "audio"` 或 `"stream": true` | 原始音频字节流，Content-Type 与[输出格式](#输出格式)一致 |
| `"stream_format": "sse"` | SSE 事件流，Content-Type 为 `text/event-stream` |

### 原始音频字节流

把收到的字节按顺序拼接，就得到完整的音频文件；也可以边收边交给播放器播放。

流式 `wav` 的文件头里，长度字段固定为 `0xFFFFFFFF`，因为发送文件头时还不知道总长度。大多数播放器能正常处理。

### SSE 事件

服务端按顺序发送以下事件：

```
event: speech.audio.delta
data: {"type":"speech.audio.delta","audio":"<base64 编码的音频片段>"}

event: speech.audio.delta
data: {"type":"speech.audio.delta","audio":"..."}

event: speech.audio.done
data: {"type":"speech.audio.done"}
```

- 每个 `speech.audio.delta` 事件携带编码后音频流中的一个片段。把各片段 base64 解码后按顺序拼接，得到完整的音频文件。单个片段不一定能单独解码。
- 收到 `speech.audio.done` 表示音频已全部发送。
- 生成过程中出错时，服务端发送一个 `error` 事件，然后结束事件流：

```
event: error
data: {"type":"error","error":{"message":"<错误信息>"}}
```

### 中途断开

客户端在流式传输中途断开连接后，服务端会立即停止生成，后面排队的请求不需要等待。

## 代码示例

以下示例从环境变量 `VOXCPM_API_KEY` 读取 key。

### curl

```bash
# 非流式：24 kHz 的 Opus
curl http://localhost:7860/v1/audio/speech \
  -H "Authorization: Bearer $VOXCPM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "你好", "response_format": "opus", "sample_rate": 24000}' \
  -o out.opus

# 流式：边收边播（需要安装 ffmpeg）
curl -sN http://localhost:7860/v1/audio/speech \
  -H "Authorization: Bearer $VOXCPM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "这段语音会边生成边播放。", "stream_format": "audio", "response_format": "wav"}' \
  | ffplay -autoexit -nodisp -

# SSE：查看事件流
curl -sN http://localhost:7860/v1/audio/speech \
  -H "Authorization: Bearer $VOXCPM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "SSE 示例。", "stream_format": "sse", "response_format": "pcm"}'
```

### Python（requests）

```python
import base64
import json
import os

import requests

URL = "http://localhost:7860/v1/audio/speech"
HEADERS = {"Authorization": f"Bearer {os.environ['VOXCPM_API_KEY']}"}

# 非流式
r = requests.post(URL, headers=HEADERS, json={
    "input": "你好，欢迎使用 VoxCPM2。",
    "response_format": "mp3",
    "sample_rate": 24000,
}, timeout=600)
r.raise_for_status()
with open("out.mp3", "wb") as f:
    f.write(r.content)

# 流式：原始音频字节
with requests.post(URL, headers=HEADERS, json={
    "input": "这段语音会边生成边返回。",
    "stream_format": "audio",
    "response_format": "pcm",
    "sample_rate": 24000,
}, stream=True, timeout=600) as r:
    r.raise_for_status()
    with open("out.pcm", "wb") as f:
        for chunk in r.iter_content(chunk_size=None):
            f.write(chunk)  # 实时播放时，在这里把 chunk 交给播放器

# 流式：SSE
with requests.post(URL, headers=HEADERS, json={
    "input": "SSE 模式示例。",
    "stream_format": "sse",
    "response_format": "opus",
}, stream=True, timeout=600) as r:
    r.raise_for_status()
    with open("out.opus", "wb") as f:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])
            if event["type"] == "speech.audio.delta":
                f.write(base64.b64decode(event["audio"]))
            elif event["type"] == "error":
                raise RuntimeError(event["error"]["message"])

# 音色克隆
with open("speaker.wav", "rb") as f:
    reference = base64.b64encode(f.read()).decode()
r = requests.post(URL, headers=HEADERS, json={
    "input": "这句话会用参考音频的音色读出来。",
    "reference_audio": reference,
}, timeout=600)
r.raise_for_status()
with open("clone.wav", "wb") as f:
    f.write(r.content)
```

### Python（OpenAI SDK）

OpenAI SDK 没有的参数（`sample_rate`、`stream_format`、`reference_audio`、`inference_timesteps`）通过 `extra_body` 传入。

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7860/v1",
    api_key=os.environ["VOXCPM_API_KEY"],
)

# 非流式
client.audio.speech.create(
    model="openbmb/VoxCPM2",
    voice="default",  # 预置音色名
    input="你好，欢迎使用 VoxCPM2。",
    instructions="年轻女性，温柔自然",
    response_format="opus",
    speed=1.2,
    extra_body={"sample_rate": 24000},
).write_to_file("sdk.opus")

# 流式
with client.audio.speech.with_streaming_response.create(
    model="openbmb/VoxCPM2",
    voice="default",
    input="这段语音会边生成边返回。",
    response_format="pcm",
    extra_body={"stream_format": "audio", "sample_rate": 24000},
) as response:
    with open("sdk.pcm", "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
```

### PowerShell

请求体必须用 UTF-8 编码发送，否则中文会变成乱码。

```powershell
$headers = @{ Authorization = "Bearer $env:VOXCPM_API_KEY" }
$body = @{ input = '你好，欢迎使用 VoxCPM2。'; response_format = 'mp3' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:7860/v1/audio/speech `
  -Headers $headers -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes($body)) -OutFile out.mp3
```

### 播放裸 PCM

`pcm` 格式没有文件头，播放时要指定采样率：

```bash
ffplay -f s16le -ar 24000 -ch_layout mono out.pcm
```

## 错误码

除 `500` 外，错误响应体都是 JSON，错误信息在 `detail` 字段中。

| 状态码 | 原因 | 示例 |
|---|---|---|
| `400` | `input` 为空 | `{"detail":"input 不能为空"}` |
| `400` | 所选格式不支持该采样率 | `{"detail":"opus 的 sample_rate 只支持 8000, 12000, 16000, 24000, 48000"}` |
| `400` | `reference_audio` 不是有效的 base64 | `{"detail":"reference_audio 不是有效的 base64"}` |
| `401` | API key 缺失或无效 | `{"detail":"API key 无效"}` |
| `422` | 参数类型或取值不合法，例如 `response_format` 不在列表中、`speed` 超出范围 | FastAPI 标准校验错误，`detail` 为数组 |
| `500` | 生成失败，例如参考音频的 base64 合法但内容无法解码为音频 | 纯文本 `Internal Server Error` |

接口在生成出第一段音频后才发送响应头，所以生成开始时就出错的请求，也能返回准确的状态码。流式传输开始后再出错时，原始字节流会直接中断；SSE 会先发送 `error` 事件，再结束事件流。

## 注意事项

- **请求排队**：所有请求在同一张 GPU 上依次执行。多个请求同时到达时，后到的请求要等前面的完成，首段延迟会相应变长。
- **冷启动**：容器启动后，第一个请求要加载模型并编译特征网络，可能需要数分钟。首次使用某个长度档位还会录制 CUDA graph；后续请求复用编译结果和图。
- **流式不自动重试**：模型偶尔会生成长度异常的音频。非流式请求会检测这种情况并自动重试，流式请求不会。对稳定性要求高、对延迟不敏感时，使用非流式请求。
- **Opus 采样率**：Opus 解码器一律输出 48 kHz，所以播放器和 ffprobe 显示的采样率总是 48000。指定的 `sample_rate` 用于编码，并记录在文件头中。
- **流式 FLAC**：流式 FLAC 文件头里没有总时长，部分播放器会显示 0 秒，但能正常播放。需要时长信息时，使用非流式请求。
- **参考音频**：使用清晰的人声片段。已验证 WAV 可用；FLAC、OGG、MP3 等常见格式也能解码。
- **明文传输**：服务使用 HTTP，API key 和网页密码在网络上以明文传输。请只在可信的局域网内使用。
- **长请求超时**：长文本的非流式请求可能需要几十秒。客户端的超时时间应设得足够长，示例中使用 600 秒。
