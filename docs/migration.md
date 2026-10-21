# 工程迁移记录

2026-10-21 从 `D:\ai\VoxCPM2` 整理出独立项目 `D:\ai\voxcpm2-service`。本次迁移复制工程文件，原目录中的文件、模型和运行服务保持原状。

## 文件映射

| 原位置 | 新位置 | 说明 |
| --- | --- | --- |
| `app.py`、`inference_accel.py`、`synthesis.py` | 仓库根目录 | 保留入口和模块导入方式；本地模型缓存默认位于用户缓存目录 |
| `static/` | `static/` | 保留连续播放页和环形 PCM 缓冲器 |
| `tests/` | `tests/` | 保留 Python 和 JavaScript 测试源码 |
| `benchmark_stream.py` | `scripts/benchmark_stream.py` | 独立放置性能测量工具 |
| `README.md` | `README.md`、`docs/api.md`、`docs/deployment.md` | 分开说明快速启动、API 和运维 |
| `Dockerfile` | `Dockerfile`、`requirements.txt` | 将运行依赖集中到一个清单 |
| `docker-compose.yml` | `docker-compose.yml` | 使用独立项目名、可配置端口和外部模型目录 |

## 未迁入的内容

模型缓存 `hf-cache/`、虚拟环境 `.venv/`、生成音频 `outputs/` 和 `test-zh.wav` 均留在原目录。真实 `.env` 不进入 Git；本机新项目通过被忽略的 `.env` 复用现有认证配置和外部模型路径。

`vllm-omni/` 及其启动、停止和接口试验脚本属于另一套服务。`.tmp-perf/` 是之前的性能探索记录，其有效实现已整理到 `inference_accel.py`、`synthesis.py` 和基准脚本。它们都保留在原目录，没有纳入新项目。

原来的 `start-web.ps1`、`stop-web.ps1` 和 `test.py` 依赖旧目录的虚拟环境或硬编码路径。新项目使用 Docker Compose 作为运行入口，本地开发方法见 `CONTRIBUTING.md`。

## 本机运行安排

迁移时旧服务使用 7860 端口。新项目的本地 `.env` 使用 7861，并指向原来的 `D:/ai/VoxCPM2/hf-cache`；这份本机设置不影响 `.env.example` 的通用默认值。项目创建过程不切换正在运行的旧服务。
