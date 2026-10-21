# 开发与 Git 管理

从仓库根目录运行以下命令。常规功能修改在单独分支上完成，并将源码、相关测试和说明一起提交。

## 修改与提交

```bash
git switch -c feat/your-change
git status --short
git diff
git add <修改的文件>
git diff --cached
git commit -m "Describe the change"
```

提交前确认暂存区只包含工程文件。`.env`、模型、缓存、运行日志和音频已加入 `.gitignore`；配置变更通过 `.env.example` 和部署文档记录。

本仓库的 `main` 分支保存当前可运行版本。初始版本标签为 `v0.1.0`。远程仓库地址由项目所有者配置；没有远程仓库时，提交仍保存在本机。

## Python 环境

生产环境优先使用 Docker。需要本地开发时，使用 Python 3.11 或 3.12，创建独立虚拟环境，并安装系统 ffmpeg：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/pcm-ring.test.mjs
```

`requirements.txt` 锁定已验证的直接依赖，不是完整的传递依赖锁文件。Docker 基础镜像已经提供 CUDA 版 PyTorch；本地安装后使用 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 检查设备支持。原生 Windows 缺少 Triton 时跳过特征网络编译，CUDA graph 仍按设备能力启用。

直接运行 `python app.py` 时，程序读取进程环境变量，不自动加载 `.env`。将 `HF_HOME` 设为仓库外的模型缓存路径，并在进程环境中设置认证变量；`.env` 由 Docker Compose 读取。

## 容器内测试

先构建运行镜像，再挂载当前源码运行测试。测试命令创建一次性容器，不暴露端口、不启动另一份服务，也不读取模型权重：

```powershell
docker compose build
docker run --rm --gpus all --mount "type=bind,source=$($PWD.Path),target=/app" -e PYTHONDONTWRITEBYTECODE=1 voxcpm2-service:local python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/pcm-ring.test.mjs
```

当前 Gradio 依赖包含 FastAPI TestClient 所需的 httpx；本地开发环境通过 `requirements-dev.txt` 显式锁定它。没有 GPU 的测试机可以移除 `--gpus all`，CUDA 专用测试会跳过。

修改部署配置后运行 `docker compose config --quiet`。修改 Dockerfile 或依赖时重新构建镜像。修改模型推理代码时，还应在实际 GPU 上运行 `scripts/benchmark_stream.py`，分别记录冷启动和预热后的结果。
