# VoxCPM2 Service

VoxCPM2 本地语音合成服务，提供 OpenAI 兼容 API、Gradio 音色测试页和连续 PCM 播放器。服务支持中文语音、音色描述、参考音频克隆、自动分句和固定参考音色。

工程使用 Git 管理。模型权重、Hugging Face 缓存、虚拟环境、真实凭据和生成音频不纳入版本控制；Docker 从仓库外挂载模型缓存。

## 快速启动

需要 Docker Compose、NVIDIA GPU 和可用的容器 GPU 支持。Windows 使用 Docker Desktop 的 WSL2 后端。服务镜像使用 Python 3.11、PyTorch 2.9.1 和 CUDA 12.8。

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，设置以下值：

```dotenv
MODEL_CACHE_DIR=D:/ai/VoxCPM2/hf-cache
VOXCPM_USER=admin
VOXCPM_PASSWORD=<设置网页登录密码>
VOXCPM_API_KEYS=<设置API密钥>
VOXCPM_PORT=7860
```

`MODEL_CACHE_DIR` 指向已有 Hugging Face 缓存目录，其下包含 `hub/`。目录必须存在。若已有服务占用 7860，将新项目端口设为 7861。模型缺失时，服务会下载到这个外部目录；只使用现有模型时设置 `HF_HUB_OFFLINE=1`。

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose logs -f
```

默认访问地址如下，修改端口后请同步调整 URL：

| 地址 | 用途 |
| --- | --- |
| `http://localhost:7860/` | 登录后试听、设计或克隆音色 |
| `http://localhost:7860/player` | 填写 API key 后分句生成、连续播放和下载 WAV |
| `POST /v1/audio/speech` | 流式或完整音频 API |
| `GET /v1/models` | 需要 API key 的模型列表 |
| `GET /health` | 无需认证的健康检查 |

首次语音请求包含加载、编译和预热。后续请求复用模型及 CUDA graph。健康检查通过表示 HTTP 服务可用，不代表模型已经预热。

## 工程结构

```text
voxcpm2-service/
├── app.py                     # HTTP API、Gradio 和串行 GPU 队列
├── inference_accel.py         # KV 裁剪与 CUDA graph
├── synthesis.py               # 分句、音色参考和取消生成
├── static/                    # PCM 环形缓冲播放器
├── tests/                     # Python 与 JavaScript 测试
├── scripts/benchmark_stream.py
├── docs/                      # API、部署和迁移记录
├── requirements.txt           # 已验证的直接运行依赖
├── requirements-dev.txt       # 测试依赖
├── Dockerfile
├── docker-compose.yml
└── .env.example               # 可提交的配置模板
```

`outputs/` 在运行时创建，用于网页生成的 WAV；编译缓存保存在 Compose 命名卷中。两者都不属于工程源码。项目不包含另一套 vLLM-Omni 服务或早期性能实验脚本。

## 开发与验证

在已安装依赖和 ffmpeg 的 Python 3.11/3.12 环境中运行现有测试；JavaScript 测试需要 Node.js 20 或更高版本。

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/pcm-ring.test.mjs
```

Python 测试使用小模型和模拟音频，不读取 VoxCPM2 权重。CUDA graph 测试在没有 CUDA 时跳过。容器内测试和依赖安装方法见[开发说明](CONTRIBUTING.md)。

服务启动后，可以测量首段延迟、实时率（RTF）和 0.6 秒缓冲下的模拟断流次数：

```powershell
$env:VOXCPM_API_KEY = '<API key>'
python scripts/benchmark_stream.py --url http://localhost:7860 --runs 3
```

RTF 小于 1 表示生成速度超过播放速度。此前在 RTX 4060 Ti 上的预热实测约为 0.36–0.44；结果随文本、音频长度、设备和系统负载变化，不代表首个请求的等待时间。

## 文档

- [API 与调用示例](docs/api.md)
- [部署、配置和运维](docs/deployment.md)
- [开发与 Git 管理](CONTRIBUTING.md)
- [工程迁移记录](docs/migration.md)

本项目依赖上游 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 和 `openbmb/VoxCPM2` 模型。模型及第三方依赖沿用各自的许可；本仓库不分发模型权重。
