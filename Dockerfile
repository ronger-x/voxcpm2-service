FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple

# gcc/libc6-dev：torch.compile 通过 Triton 在运行时编译内核；ffmpeg：解码 mp3 等格式的参考音频
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
# 锁定基础镜像自带的 CUDA 版 torch，避免被依赖解析升级；torchcodec 0.8 对应 torch 2.9
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} -r requirements.txt

COPY app.py inference_accel.py synthesis.py ./
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
