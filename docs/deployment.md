# 部署与运维

本服务通过 Docker Compose 使用一张 NVIDIA GPU。模型文件通过 `MODEL_CACHE_DIR` 指定的外部目录挂载；构建上下文只包含运行源码、前端资源和依赖清单。

## 配置

复制 `.env.example` 为 `.env`，填写真实配置。`.env` 不纳入 Git。密码、API key 和模型缓存路径为空时，Compose 拒绝启动。

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MODEL_CACHE_DIR` | 必填 | 主机上的 Hugging Face 缓存目录，包含 `hub/`，挂载到 `/data/hf-cache` |
| `VOXCPM_USER` | `admin` | Gradio 登录用户名 |
| `VOXCPM_PASSWORD` | 必填 | Gradio 登录密码 |
| `VOXCPM_API_KEYS` | 必填 | API key，多个值用逗号分隔 |
| `VOXCPM_HOST` | `0.0.0.0` | 主机绑定地址；只供本机使用时设为 `127.0.0.1` |
| `VOXCPM_PORT` | `7860` | 主机端口 |
| `VOXCPM_VOICES_DIR` | `./voices` | 主机上的预置音色目录，只读挂载到 `/app/voices`；目录不存在时 Compose 自动创建 |
| `VOXCPM_ENGINE` | `nano` | 推理引擎：`nano` 为 Nano-vLLM，`torch` 为上游 PyTorch 实现，见[推理与播放](#推理与播放) |
| `VOXCPM_GPU_MEMORY_UTILIZATION` | `0.55` | 仅 `nano`：可占用整卡显存的比例，模型与 KV cache 都在其中。与 Ollama 等共用显卡时调小 |
| `VOXCPM_CUDA_GRAPH` | `1` | 仅 `torch`：设为 `0` 关闭 CUDA graph，保留 KV 长度裁剪 |
| `VOXCPM_COMPILE_FEATURES` | `1` | 仅 `torch`：设为 `0` 关闭特征网络编译 |
| `HF_HUB_OFFLINE` | `0` | 设为 `1` 时只读取本地已有模型 |

Windows 路径建议使用正斜杠，例如 `D:/ai/VoxCPM2/hf-cache`。Compose 不会自动创建模型目录；路径错误时直接报错。缓存目录不存在模型且未开启离线模式时，程序将模型下载到该外部目录。

服务使用 HTTP。通过局域网访问时，请使用可信网络或在前面配置 HTTPS 代理。`/player` 页面可以直接打开，生成请求仍需 API key；Gradio 页面使用单独的账号和密码。

## 启动与更新

```bash
docker compose config --quiet
docker compose up -d --build
docker compose ps
docker compose logs -f
docker compose stop
docker compose down
```

Compose 项目名为 `voxcpm2-service`，镜像为 `voxcpm2-service:local`。使用 `docker compose` 管理本项目，可与旧目录的 `voxcpm2` 容器区分。修改 `.env` 后再次执行 `docker compose up -d`；修改源码、前端或依赖后使用 `--build`。

如果需要更换 PyPI 镜像，在构建时传入地址：

```bash
docker compose build --build-arg PIP_INDEX_URL=https://pypi.org/simple
```

镜像构建时从 GitHub 下载 FlashAttention 预编译包（约 250 MB，社区构建，官方没有 torch 2.9 的包），并校验 SHA-256。主机访问 GitHub 慢时，先把 `Dockerfile` 里 `FLASH_ATTN_WHEEL` 对应的文件放进仓库的 `wheels/` 目录，构建会直接使用。

## 数据与备份

| 数据 | 位置 | 处理方式 |
| --- | --- | --- |
| 工程源码 | Git 仓库 | 使用 Git 提交与备份 |
| 模型文件 | `MODEL_CACHE_DIR` | 仓库外保存，可独立备份 |
| 密码和 API key | `.env` | 单独保管，不提交 |
| 网页输出音频 | `outputs/latest.wav` | 运行时创建，不提交 |
| 预置音色 | `VOXCPM_VOICES_DIR` | 参考音频不提交；替换后各客户端的音色随之改变，建议单独备份 |
| 编译缓存 | Compose 的 `compile-cache` 命名卷 | 可以重新生成 |

`docker compose down` 停止服务并移除容器，保留命名卷。首次使用新编译缓存时需要更长预热时间。复用模型目录不会复制模型文件，原目录中的模型仍供旧服务使用。

## 推理与播放

GPU 工作线程串行处理请求，两种引擎都逐句生成，分句、分段语气、非语言标签和合成后韵律处理相同。

默认的 `nano` 引擎是 [Nano-vLLM-VoxCPM](https://github.com/a710128/nanovllm-voxcpm)：模型在单独的进程里运行，使用分页 KV cache、FlashAttention 和 CUDA graph。预置音色的参考音频只在第一次使用时编码，之后复用。在 RTX 4060 Ti（16 GB，与 Ollama 9B 模型共用）上逐句测得：

| | `torch` | `nano` |
|---|---|---|
| 首段音频 | 0.37 s | 0.09 s |
| RTF（20 字以上的句子） | 0.42 | 0.26 |
| 与参考音的声纹相似度 | 0.86 | 0.87 |
| 听写准确率 | 0.85 | 0.87 |

`nano` 只支持 Linux 下的 NVIDIA GPU（容器即可），扩散步数在启动时固定为 `VOXCPM_INFERENCE_TIMESTEPS`（默认 10），请求里的 `inference_timesteps` 不生效。它按 `VOXCPM_GPU_MEMORY_UTILIZATION` 预占显存，与其他 GPU 程序共用时要留出余量。

`torch` 引擎的注意力只计算已用 KV 前缀；CUDA graph 以 256 个位置分档捕获，缓存到 2048 个位置，之后采用精确长度的普通计算。图捕获失败时回退到普通计算。

服务在发出首段音频之前完成短句预热，减少首次播放期间的编译停顿。`/health` 只检查 HTTP 服务，不检查模型是否已经加载。首个请求可能需要数分钟；其他请求排队时首段延迟也会变长。

连续播放页预缓冲约 0.6 秒，最多保留 30 秒待播音频。HTTPS 和 localhost 使用 AudioWorklet，局域网 HTTP 使用 ScriptProcessor 兼容路径。兼容路径受主线程负载影响；浏览器播放不能仅凭服务端 RTF 保证无卡顿。
