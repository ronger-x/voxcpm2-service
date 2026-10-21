import asyncio
import base64
import binascii
import io
import json
import os
import queue
import secrets
import subprocess
import tempfile
import threading
from contextlib import aclosing, closing
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from voxcpm import VoxCPM

from inference_accel import optimize_model
from synthesis import generate_segments

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))).expanduser()
OUTPUT_DIR = Path(os.environ.get('VOXCPM_OUTPUT_DIR', str(ROOT / 'outputs'))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# 预置音色：目录下的 <名称>.wav 等参考音频，请求的 voice 按文件名匹配
VOICES_DIR = Path(os.environ.get('VOXCPM_VOICES_DIR', str(ROOT / 'voices'))).expanduser()
VOICE_EXTENSIONS = ('.wav', '.flac', '.mp3', '.ogg')
DEFAULT_VOICE = 'default'
os.environ.setdefault('HF_HOME', str(CACHE_DIR))

MODEL = None
API_KEYS = [k.strip() for k in os.environ.get('VOXCPM_API_KEYS', '').split(',') if k.strip()]
# 格式 -> (ffmpeg 输出参数, Content-Type)
AUDIO_FORMATS = {
    'wav': (['-c:a', 'pcm_s16le', '-fflags', '+bitexact', '-f', 'wav'], 'audio/wav'),
    'pcm': (['-c:a', 'pcm_s16le', '-f', 's16le'], 'audio/pcm'),
    'mp3': (['-c:a', 'libmp3lame', '-f', 'mp3'], 'audio/mpeg'),
    'opus': (['-c:a', 'libopus', '-page_duration', '20000', '-f', 'ogg'], 'audio/ogg'),
    'aac': (['-c:a', 'aac', '-f', 'adts'], 'audio/aac'),
    'flac': (['-c:a', 'flac', '-f', 'flac'], 'audio/flac'),
}
SAMPLE_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
OPUS_SAMPLE_RATES = (8000, 12000, 16000, 24000, 48000)


def build_on_gpu():
    # Docker Desktop 的 VM 内存有限：默认流程先在 CPU 上以 fp32 构建模型（约 9 GiB），会被 OOM 杀掉。
    # 这里改为直接在 GPU 上构建，并把权重直接读到显存。
    from safetensors.torch import load_file
    from voxcpm.model import voxcpm2

    init = voxcpm2.VoxCPM2Model.__init__

    def init_on_gpu(self, *args, **kwargs):
        with torch.device('cuda'):
            init(self, *args, **kwargs)

    voxcpm2.VoxCPM2Model.__init__ = init_on_gpu
    voxcpm2.load_file = lambda path, device='cpu': load_file(path, device='cuda')


def load_model():
    global MODEL
    if MODEL is None:
        if os.environ.get('VOXCPM_BUILD_ON_GPU') == '1' and torch.cuda.is_available():
            build_on_gpu()
        model = VoxCPM.from_pretrained('openbmb/VoxCPM2', load_denoiser=False, optimize=False)
        optimize_model(
            model.tts_model,
            use_graph=os.environ.get('VOXCPM_CUDA_GRAPH', '1') == '1',
            compile_features=os.environ.get('VOXCPM_COMPILE_FEATURES', '1') == '1',
        )
        # Complete feature compilation and the first graph capture before
        # exposing any PCM to a player. Otherwise its first buffer can drain
        # while the second audio patch waits for compilation.
        warmup = build_kwargs('你好，欢迎使用语音服务。模型预热已经完成。')
        warmup['max_len'] = 4
        with torch.inference_mode(), closing(generate_segments(model, warmup, True, threading.Event())) as chunks:
            for _ in chunks:
                pass
        MODEL = model
    return MODEL


# 所有 GPU 计算都在这一个线程里串行执行：模型只有一份 KV cache，
# 而且 torch.compile 的 CUDA graph 按线程管理，换线程调用会重新录制
JOBS = queue.Queue()
_DONE = object()


def gpu_worker():
    while True:
        kwargs, streaming, out, cancelled, cleanup_reference = JOBS.get()

        def deliver(item):
            while not cancelled.is_set():
                try:
                    out.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        try:
            if cancelled.is_set():
                continue
            model = load_model()
            sample_rate = model.tts_model.sample_rate
            complete = []
            with torch.inference_mode(), closing(generate_segments(model, kwargs, streaming, cancelled)) as chunks:
                for wav in chunks:
                    if streaming and not deliver((sample_rate, wav)):
                        break
                    if not streaming:
                        complete.append(wav)
            if not streaming and complete and not cancelled.is_set():
                deliver((sample_rate, np.concatenate(complete)))
        except Exception as e:
            deliver(e)
        finally:
            if cleanup_reference:
                Path(cleanup_reference).unlink(missing_ok=True)
            if not deliver(_DONE):
                # Wake an out.get() already running in asyncio's thread pool.
                while True:
                    try:
                        out.get_nowait()
                    except queue.Empty:
                        break
                out.put_nowait(_DONE)
            JOBS.task_done()


threading.Thread(target=gpu_worker, daemon=True).start()


async def run_on_gpu(kwargs, streaming, cleanup_reference=None):
    """提交生成任务，逐段返回 (采样率, 波形)。非流式只返回一段完整音频。"""
    out = queue.Queue(maxsize=8)
    cancelled = threading.Event()
    JOBS.put((kwargs, streaming, out, cancelled, cleanup_reference))
    try:
        while (item := await asyncio.to_thread(out.get)) is not _DONE:
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        # 客户端断开或提前结束时通知工作线程停止
        cancelled.set()


def build_kwargs(
    text, control=None, reference_audio=None, inference_timesteps=10, split_sentences=True, reference_text=None
):
    kwargs = {
        'text': text,
        'control': control,
        'split_sentences': split_sentences,
        'cfg_value': 2.0,
        'inference_timesteps': int(inference_timesteps),
        'max_len': 4096,
    }
    if reference_audio:
        kwargs['reference_wav_path'] = reference_audio
        if reference_text and reference_text.strip():
            kwargs['reference_text'] = reference_text.strip()
    return kwargs


def wav_bytes(wav, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format='WAV')
    return buffer.getvalue()


def atempo_filter(speed):
    # atempo 单级只支持 0.5 以上，更慢的速度拆成多级
    stages = []
    while speed < 0.5:
        stages.append(0.5)
        speed /= 0.5
    stages.append(speed)
    return ','.join(f'atempo={s:g}' for s in stages)


def ffmpeg_command(source_rate, response_format, sample_rate, speed):
    """把单声道 float32 PCM 转成目标格式的 ffmpeg 命令，调用方在末尾补上输出位置。"""
    command = [
        'ffmpeg', '-v', 'error', '-nostdin',
        # 输入是逐段产生的裸 PCM：关闭探测，收到数据立即编码
        '-probesize', '32', '-analyzeduration', '0',
        '-f', 'f32le', '-ar', str(source_rate), '-ac', '1', '-i', 'pipe:0',
    ]
    if speed != 1:
        command += ['-af', atempo_filter(speed)]
    return command + ['-ar', str(sample_rate), *AUDIO_FORMATS[response_format][0], '-flush_packets', '1']


def encode_audio(wav, command):
    # 写到临时文件而不是管道，ffmpeg 才能回填 WAV/FLAC 文件头里的总长度
    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, 'audio')
        subprocess.run([*command, output], input=wav.astype('<f4').tobytes(), capture_output=True, check=True)
        return Path(output).read_bytes()


async def encode_stream(results, first, command):
    proc = await asyncio.create_subprocess_exec(
        *command, 'pipe:1', stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )

    async def feed():
        try:
            proc.stdin.write(first.astype('<f4').tobytes())
            async for _, wav in results:
                await proc.stdin.drain()
                proc.stdin.write(wav.astype('<f4').tobytes())
            await proc.stdin.drain()
        finally:
            proc.stdin.close()

    feeder = asyncio.create_task(feed())
    try:
        while data := await proc.stdout.read(65536):
            yield data
        await feeder  # 生成过程中的异常在这里抛出
        if await proc.wait() != 0:
            raise RuntimeError('ffmpeg 编码失败')
    finally:
        # 客户端断开时停止喂数据、结束 ffmpeg，并通知 GPU 线程停止生成
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        await results.aclose()


def sse_event(event_type, **data):
    payload = json.dumps({'type': event_type, **data}, separators=(',', ':'))
    return f'event: {event_type}\ndata: {payload}\n\n'


async def generate_audio(text, control, reference_audio, inference_timesteps, streaming):
    text = (text or '').strip()
    if not text:
        raise gr.Error('请输入要合成的文本')
    kwargs = build_kwargs(text, control, reference_audio, inference_timesteps)
    chunks = []
    async with aclosing(run_on_gpu(kwargs, streaming)) as results:
        async for sample_rate, wav in results:
            chunks.append(wav)
            yield wav_bytes(wav, sample_rate), gr.skip()
    output = OUTPUT_DIR / 'latest.wav'
    sf.write(output, np.concatenate(chunks), sample_rate)
    yield gr.skip(), str(output)


def gpu_status():
    if not torch.cuda.is_available():
        return 'CUDA 不可用，将无法使用当前 GPU 配置。'
    free, total = torch.cuda.mem_get_info()
    return f'{torch.cuda.get_device_name(0)} · 显存 {free / 1024**3:.1f} / {total / 1024**3:.1f} GiB 可用'


with gr.Blocks(title='VoxCPM2 本地语音测试') as demo:
    gr.Markdown('# VoxCPM2 本地语音测试\n在本机运行中文文本转语音、音色设计和参考音频克隆。')
    gr.Markdown('[打开连续播放页](/player)：长文本自动分句，缓冲后连续播放，可随时停止。')
    status = gr.Markdown(gpu_status())
    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(
                label='合成文本',
                value='你好，这是 VoxCPM2 的本地中文语音测试。',
                lines=5,
            )
            control = gr.Textbox(
                label='音色 / 风格描述（可选）',
                placeholder='例如：年轻女性，温柔自然，语速稍慢，带有亲切感',
            )
            reference = gr.Audio(label='参考音频（可选，用于音色克隆）', type='filepath')
            steps = gr.Slider(5, 20, value=10, step=1, label='推理步数（越高越慢，音质可能更稳定）')
            stream = gr.Checkbox(
                value=True,
                label='流式播放',
                info='边生成边播放。关闭后整段生成再播放，音频长度异常时会自动重试。',
            )
            generate = gr.Button('生成语音', variant='primary')
        with gr.Column(scale=1):
            audio = gr.Audio(label='输出音频', streaming=True, autoplay=True)
            download = gr.File(label='完整音频（WAV）')
            gr.Markdown('提示：首次生成会加载模型，后续生成会更快。参考音频建议使用清晰的 16kHz 人声片段。')
    generate.click(generate_audio, [text, control, reference, steps, stream], [audio, download])

api = FastAPI()
api.mount('/player-assets', StaticFiles(directory=ROOT / 'static'), name='player-assets')


@api.get('/player', include_in_schema=False)
def player():
    return FileResponse(ROOT / 'static' / 'player.html')


class SpeechRequest(BaseModel):
    input: str
    model: str = 'openbmb/VoxCPM2'
    voice: str | None = None  # 预置音色名；未知或未填时使用 default 预置音色
    instructions: str | None = None  # 音色 / 风格描述
    response_format: Literal['wav', 'pcm', 'mp3', 'opus', 'aac', 'flac'] = 'wav'
    sample_rate: int | None = None  # 输出采样率，默认用模型原生的 48000
    speed: float = Field(1.0, ge=0.25, le=4.0)  # 变速不变调
    stream: bool = False  # 流式返回音频字节
    stream_format: Literal['audio', 'sse'] | None = None  # 同 OpenAI；设置后即为流式
    inference_timesteps: int = Field(10, ge=5, le=20)
    reference_audio: str | None = None  # base64 编码的参考音频，可带 data: URL 前缀
    reference_text: str | None = None  # reference_audio 的逐字文本；提供后音色还原更稳定
    split_sentences: bool = True  # 分句合成，固定参考音色


def check_api_key(authorization, x_api_key):
    if not API_KEYS:
        return
    key = x_api_key or ''
    if authorization and authorization.lower().startswith('bearer '):
        key = authorization[7:].strip()
    if not any(secrets.compare_digest(key.encode(), k.encode()) for k in API_KEYS):
        raise HTTPException(401, 'API key 无效', headers={'WWW-Authenticate': 'Bearer'})


def check_sample_rate(sample_rate, response_format):
    if sample_rate is None:
        return
    allowed = OPUS_SAMPLE_RATES if response_format == 'opus' else SAMPLE_RATES
    if sample_rate not in allowed:
        raise HTTPException(
            400, f'{response_format} 的 sample_rate 只支持 {", ".join(map(str, allowed))}'
        )


def save_reference_audio(data):
    if data.startswith('data:'):
        data = data.split(',', 1)[-1]
    try:
        audio_bytes = base64.b64decode(data, validate=True)
    except binascii.Error:
        raise HTTPException(400, 'reference_audio 不是有效的 base64')
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(audio_bytes)
    return f.name


def preset_voices():
    """音色名 -> 参考音频路径。每次请求重新扫描，增删文件不需要重启服务。"""
    if not VOICES_DIR.is_dir():
        return {}
    return {
        path.stem: path for path in sorted(VOICES_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in VOICE_EXTENSIONS
    }


def preset_transcript(audio_path):
    """预置音色同名的 .txt 是参考音频的逐字文本；缺失或为空时只按参考音频克隆。"""
    path = Path(audio_path).with_suffix('.txt')
    try:
        return path.read_text(encoding='utf-8').strip() or None
    except FileNotFoundError:
        return None


def resolve_voice(name, instructions=None):
    """返回 (音色名, 参考音频路径, 参考文本)。

    voice 匹配预置文件时使用该音色。未匹配时：有 instructions 则按描述设计音色（不用参考）；
    否则回落到 default 预置音色。都没有时返回 (None, None, None)，此时每个请求的音色是随机的。
    """
    voices = preset_voices()
    for candidate in (name, None if instructions and instructions.strip() else DEFAULT_VOICE):
        if candidate in voices:
            return candidate, str(voices[candidate]), preset_transcript(voices[candidate])
    return None, None, None


@api.get('/health')
def health():
    return {'status': 'ok'}


@api.get('/v1/models')
def models(authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    check_api_key(authorization, x_api_key)
    return {'object': 'list', 'data': [
        {'id': 'openbmb/VoxCPM2', 'object': 'model', 'created': 0, 'owned_by': 'openbmb'},
    ]}


@api.get('/v1/voices')
def voices(authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    check_api_key(authorization, x_api_key)
    return {'object': 'list', 'data': [{'id': name, 'object': 'voice'} for name in preset_voices()]}


@api.post('/v1/audio/speech')
async def speech(
    req: SpeechRequest,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None),
):
    check_api_key(authorization, x_api_key)
    text = req.input.strip()
    if not text:
        raise HTTPException(400, 'input 不能为空')
    check_sample_rate(req.sample_rate, req.response_format)
    streaming = req.stream or req.stream_format is not None
    # 上传的参考音频优先于预置音色；只有上传的临时文件需要在生成后删除
    if req.reference_audio:
        voice_name, voice_path, reference_text = 'reference_audio', None, req.reference_text
        reference_path = save_reference_audio(req.reference_audio)
    else:
        voice_name, voice_path, reference_text = resolve_voice(req.voice, req.instructions)
        reference_path = None
    kwargs = build_kwargs(
        text, req.instructions, reference_path or voice_path, req.inference_timesteps, req.split_sentences,
        reference_text,
    )
    results = run_on_gpu(kwargs, streaming, cleanup_reference=reference_path)
    try:
        # 等到第一段音频再发响应头，生成失败时还能返回正常的错误状态码
        model_rate, first = await anext(results)
    except BaseException:
        await results.aclose()
        raise
    sample_rate = req.sample_rate or model_rate
    command = ffmpeg_command(model_rate, req.response_format, sample_rate, req.speed)
    media_type = AUDIO_FORMATS[req.response_format][1]
    # 响应头只能是 latin-1，中文音色名按 URL 编码
    headers = {'X-Sample-Rate': str(sample_rate), 'X-Voice': quote(voice_name or 'none')}
    if not streaming:
        await results.aclose()
        data = await asyncio.to_thread(encode_audio, first, command)
        return Response(data, media_type=media_type, headers=headers)

    audio = encode_stream(results, first, command)
    if req.stream_format != 'sse':
        return StreamingResponse(audio, media_type=media_type, headers=headers)

    async def events():
        try:
            async with aclosing(audio) as chunks:
                async for data in chunks:
                    yield sse_event('speech.audio.delta', audio=base64.b64encode(data).decode())
        except Exception as e:
            yield sse_event('error', error={'message': str(e)})
            return
        yield sse_event('speech.audio.done')

    return StreamingResponse(events(), media_type='text/event-stream', headers=headers)


if __name__ == '__main__':
    password = os.environ.get('VOXCPM_PASSWORD')
    # API 路由先注册，Gradio 挂在 / 上兜底，所以 /v1 和 /health 不走网页登录
    app = gr.mount_gradio_app(
        api,
        demo,
        path='/',
        show_error=True,
        auth=(os.environ.get('VOXCPM_USER', 'admin'), password) if password else None,
        auth_message='VoxCPM2 本地语音测试',
    )
    uvicorn.run(app, host=os.environ.get('GRADIO_SERVER_NAME', '127.0.0.1'), port=7860)
