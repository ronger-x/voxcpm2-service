FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
# FlashAttention 官方 release 没有 torch 2.9 的预编译包，用社区构建（mjun0812/flash-attention-prebuild-wheels）。
# 网络慢时可先下载到构建目录的 wheels/，构建时优先使用本地文件。
ARG FLASH_ATTN_WHEEL=flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl
ARG FLASH_ATTN_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/${FLASH_ATTN_WHEEL}
ARG FLASH_ATTN_SHA256=d4c842693349ef3e8b3b1396056003ecb9d21e94493dd1ea9ad6a2b3c6ce5c73

# gcc/libc6-dev：Triton 在运行时编译内核；ffmpeg：解码 mp3 等格式的参考音频、合成后韵律处理
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
COPY wheels/ /tmp/wheels/
# 锁定基础镜像自带的 CUDA 版 torch，避免被依赖解析升级；torchcodec 0.8 对应 torch 2.9
RUN pip freeze | grep -iE '^(torch|torchaudio|torchvision|triton)==' > /tmp/constraints.txt \
    && wheel=/tmp/wheels/${FLASH_ATTN_WHEEL} \
    && if [ ! -f "$wheel" ]; then python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "${FLASH_ATTN_URL}" "$wheel"; fi \
    && echo "${FLASH_ATTN_SHA256}  $wheel" | sha256sum -c - \
    && pip install --no-cache-dir -i ${PIP_INDEX_URL} -c /tmp/constraints.txt "$wheel" -r requirements.txt \
    && rm -rf /tmp/wheels

COPY app.py inference_accel.py nano_engine.py synthesis.py ./
COPY static ./static

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    VOXCPM_BUILD_ON_GPU=1 \
    TORCHINDUCTOR_COMPILE_THREADS=1 \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TRITON_CACHE_DIR=/cache/triton

EXPOSE 7860
CMD ["python", "app.py"]
