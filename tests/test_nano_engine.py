import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nano_engine import NanoEngine
from synthesis import generate_segments


class FakePool:
    """Stands in for nanovllm_voxcpm's AsyncVoxCPM2ServerPool."""

    def __init__(self, chunks=3, chunk_size=80, **config):
        self.config = config
        self.chunks = chunks
        self.chunk_size = chunk_size
        self.encoded, self.prompts, self.removed, self.requests = [], [], [], []
        self.closed = 0
        self.stopped = False

    async def wait_for_ready(self):
        pass

    async def get_model_info(self):
        return {'sample_rate': 48000}

    async def encode_latents(self, wav, wav_format):
        self.encoded.append(wav)
        return b'latents:' + wav

    async def add_prompt(self, wav, wav_format, prompt_text):
        self.prompts.append(prompt_text)
        return f'prompt-{len(self.prompts)}'

    async def remove_prompt(self, prompt_id):
        self.removed.append(prompt_id)

    async def generate(self, **kwargs):
        self.requests.append(kwargs)
        try:
            for index in range(min(self.chunks, kwargs['max_generate_length'])):
                yield np.full(self.chunk_size, 0.25 + index, dtype=np.float32)
        finally:
            self.closed += 1

    async def stop(self):
        self.stopped = True


def start(pool):
    def factory(**config):
        pool.config = config
        return pool

    return NanoEngine('/model', gpu_memory_utilization=0.5, server_factory=factory)


class NanoEngineTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.reference = Path(self.dir.name, 'voice.wav')
        self.reference.write_bytes(b'RIFF-voice')
        self.pool = FakePool()
        self.engine = start(self.pool)

    def tearDown(self):
        self.engine.close()
        self.dir.cleanup()

    def test_engine_settings_and_voice_encoded_once(self):
        self.assertEqual(self.engine.sample_rate, 48000)
        self.assertEqual(self.pool.config['gpu_memory_utilization'], 0.5)
        self.assertEqual(self.pool.config['max_num_batched_tokens'], 4 * 4096 // 2)
        first = self.engine.voice(str(self.reference), ' 参考文本。 ')
        again = self.engine.voice(str(self.reference), '参考文本。')
        plain = self.engine.voice(str(self.reference))
        self.assertIs(first, again)
        self.assertEqual(first.prompt_id, 'prompt-1')
        self.assertIsNone(plain.prompt_id)
        self.assertEqual(self.pool.prompts, ['参考文本。'])
        self.assertEqual(len(self.pool.encoded), 2)

    def test_generate_streams_with_reference_and_transcript(self):
        voice = self.engine.voice(str(self.reference), '参考文本。')
        chunks = list(self.engine.generate('你好。', voice, threading.Event()))
        self.assertEqual([len(c) for c in chunks], [80, 80, 80])
        request = self.pool.requests[0]
        self.assertEqual(request['ref_audio_latents'], b'latents:RIFF-voice')
        self.assertEqual(request['prompt_id'], 'prompt-1')
        self.assertEqual(request['max_generate_length'], 3 * 6 + 10)
        self.assertEqual(self.pool.closed, 1)

        list(self.engine.generate('你好。', None, threading.Event()))
        self.assertNotIn('ref_audio_latents', self.pool.requests[1])

    def test_cancel_closes_the_worker_stream(self):
        pool = FakePool(chunks=1000)
        engine = start(pool)
        try:
            cancelled = threading.Event()
            stream = engine.generate('很长的一句话，要生成很久很久很久很久。', None, cancelled)
            next(stream)
            cancelled.set()
            self.assertEqual(list(stream), [])
            self.assertEqual(pool.closed, 1)
        finally:
            engine.close()

    def test_non_streaming_retries_a_sentence_that_never_stops(self):
        class Runaway(FakePool):
            async def generate(self, **kwargs):
                self.requests.append(kwargs)
                count = kwargs['max_generate_length'] if len(self.requests) < 3 else 2
                for _ in range(count):
                    yield np.zeros(10, dtype=np.float32)

        pool = Runaway()
        engine = start(pool)
        try:
            chunks = list(engine.generate('好。', None, threading.Event(), streaming=False))
            self.assertEqual(len(pool.requests), 3)
            self.assertEqual(len(chunks), 2)
            # Streaming cannot take back audio already sent, so it does not retry
            pool.requests.clear()
            list(engine.generate('好。', None, threading.Event(), streaming=True))
            self.assertEqual(len(pool.requests), 1)
        finally:
            engine.close()

    def test_voice_cache_is_bounded_and_releases_prompts(self):
        with patch('nano_engine.MAX_VOICES', 2):
            for index in range(3):
                path = Path(self.dir.name, f'{index}.wav')
                path.write_bytes(f'clip-{index}'.encode())
                self.engine.voice(str(path), f'文本{index}')
        self.assertEqual(self.pool.removed, ['prompt-1'])

    def test_generate_segments_uses_the_engine(self):
        result = list(generate_segments(self.engine, {
            'text': '一。二。', 'control': '温柔', 'reference_wav_path': str(self.reference),
            'reference_text': '参考文本。', 'cfg_value': 2.0, 'max_len': 4096,
        }, True, threading.Event()))
        self.assertEqual(len(result), 6)
        self.assertEqual([r['target_text'] for r in self.pool.requests], ['(温柔)一。', '(温柔)二。'])
        self.assertTrue(all(r['prompt_id'] == 'prompt-1' for r in self.pool.requests))

        # No reference: the first sentence becomes the fixed anchor for the rest, without a transcript
        self.pool.requests.clear()
        list(generate_segments(self.engine, {'text': '一。二。三。'}, True, threading.Event()))
        self.assertNotIn('ref_audio_latents', self.pool.requests[0])
        anchor = self.pool.requests[1]['ref_audio_latents']
        self.assertTrue(anchor.startswith(b'latents:RIFF'))
        self.assertEqual(self.pool.requests[2]['ref_audio_latents'], anchor)
        self.assertNotIn('prompt_id', self.pool.requests[1])

    def test_close_stops_the_pool(self):
        self.engine.close()
        self.assertTrue(self.pool.stopped)
        self.engine = start(FakePool())


if __name__ == '__main__':
    unittest.main()
