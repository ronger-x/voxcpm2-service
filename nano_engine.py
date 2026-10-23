"""Nano-vLLM engine for VoxCPM2 (https://github.com/a710128/nanovllm-voxcpm).

The engine runs in its own worker process with paged KV cache and CUDA graphs. On an RTX 4060 Ti
sharing the card with an Ollama 9B model it cut first audio from 0.37 s to 0.09 s and RTF from
0.42 to 0.26 against the PyTorch path, with the same speaker similarity and transcription accuracy.

This module adapts it to `synthesis.generate_segments`: a blocking per-sentence generator, a
reference voice encoded once and reused, and cancellation that closes the worker stream.
"""

import asyncio
import hashlib
import threading
from collections import OrderedDict
from contextlib import closing, suppress
from pathlib import Path

import numpy as np

# Same guard as upstream `retry_badcase`: a sentence longer than ~6 audio patches (~1 s) per
# character means the model failed to stop. Each streamed chunk is one patch.
PATCHES_PER_CHAR = 6
RETRIES = 3
MAX_VOICES = 32


class NanoVoice:
    """Latents of one reference clip; `prompt_id` adds continuation cloning from its transcript."""

    def __init__(self, reference_latents, prompt_id=None):
        self.reference_latents = reference_latents
        self.prompt_id = prompt_id


class NanoEngine:
    def __init__(self, model_path, gpu_memory_utilization, inference_timesteps=10, max_model_len=4096,
                 max_num_seqs=4, server_factory=None):
        # Nano-vLLM's async pool must live on one event loop; run it on a private thread and
        # call it from the service's single GPU worker thread.
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name='nano-loop', daemon=True)
        self._thread.start()
        self._voices = OrderedDict()
        self._voice_lock = threading.Lock()

        async def start():
            if server_factory is None:
                from nanovllm_voxcpm import VoxCPM
                factory = VoxCPM.from_pretrained
            else:
                factory = server_factory
            pool = factory(
                model=model_path, devices=[0], inference_timesteps=inference_timesteps,
                max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_seqs * max_model_len // 2,
                max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
            )
            await pool.wait_for_ready()
            return pool, await pool.get_model_info()

        try:
            self.pool, info = self._run(start())
        except BaseException:
            self.loop.call_soon_threadsafe(self.loop.stop)
            raise
        self.sample_rate = int(info['sample_rate'])

    def _run(self, coroutine, timeout=None):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    @staticmethod
    async def _next(stream):
        # run_coroutine_threadsafe needs a coroutine; __anext__() returns a bare awaitable
        return await anext(stream)

    @staticmethod
    async def _close(stream):
        await stream.aclose()

    def voice(self, path, prompt_text=None):
        """Encode a reference clip once per (content, transcript). Uploaded clips are temporary files
        with new names each request, so the cache is keyed by content and bounded."""
        data = Path(path).read_bytes()
        prompt_text = (prompt_text or '').strip()
        key = (hashlib.sha256(data).hexdigest(), prompt_text)
        with self._voice_lock:
            if key in self._voices:
                self._voices.move_to_end(key)
                return self._voices[key]
        fmt = Path(path).suffix.lstrip('.').lower() or 'wav'

        async def encode():
            latents = await self.pool.encode_latents(data, fmt)
            prompt_id = await self.pool.add_prompt(data, fmt, prompt_text) if prompt_text else None
            return NanoVoice(latents, prompt_id)

        voice = self._run(encode())
        evicted = []
        with self._voice_lock:
            self._voices[key] = voice
            while len(self._voices) > MAX_VOICES:
                evicted.append(self._voices.popitem(last=False)[1])
        for old in evicted:
            if old.prompt_id:
                with suppress(Exception):
                    self._run(self.pool.remove_prompt(old.prompt_id))
        return voice

    def _stream(self, text, voice, cancelled, cfg_value, limit):
        kwargs = {'target_text': text, 'cfg_value': cfg_value, 'max_generate_length': limit}
        if voice is not None:
            kwargs['ref_audio_latents'] = voice.reference_latents
            if voice.prompt_id:
                kwargs['prompt_id'] = voice.prompt_id
        stream = self.pool.generate(**kwargs)
        try:
            while not cancelled.is_set():
                try:
                    chunk = self._run(self._next(stream))
                except StopAsyncIteration:
                    return
                if chunk.size:
                    yield np.asarray(chunk, dtype=np.float32)
        finally:
            # Closing the async generator cancels the sequence in the worker and frees its KV blocks.
            with suppress(Exception):
                self._run(self._close(stream), timeout=10)

    def generate(self, text, voice, cancelled, streaming=True, cfg_value=2.0, max_len=2000):
        """Yield float32 PCM chunks for one sentence; stops within one chunk after `cancelled` is set.

        Non-streaming requests retry a sentence that ran into the length guard, like upstream.
        """
        limit = max(1, min(len(text) * PATCHES_PER_CHAR + 10, max_len))
        attempts = 1 if streaming else RETRIES
        for attempt in range(attempts):
            chunks = []
            with closing(self._stream(text, voice, cancelled, cfg_value, limit)) as stream:
                for chunk in stream:
                    if streaming:
                        yield chunk
                    else:
                        chunks.append(chunk)
            if streaming or cancelled.is_set():
                return
            if len(chunks) < limit or attempt == attempts - 1:
                yield from chunks
                return

    def close(self):
        with suppress(Exception):
            self._run(self.pool.stop(), timeout=30)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
