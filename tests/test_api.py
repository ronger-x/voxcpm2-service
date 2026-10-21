import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

import app


class FakeTTS:
    sample_rate = 48000

    def build_prompt_cache(self, reference_wav_path):
        return {'ref': str(reference_wav_path)}

    def _generate_with_prompt_cache(self, **kwargs):
        for _ in range(3):
            yield torch.full((1, 4800), 0.125), None, None


class APITests(unittest.TestCase):
    def setUp(self):
        self.keys = patch.object(app, 'API_KEYS', ['test-key'])
        self.model = patch.object(app, 'MODEL', SimpleNamespace(tts_model=FakeTTS()))
        self.keys.start()
        self.model.start()
        self.client = TestClient(app.api)
        self.headers = {'Authorization': 'Bearer test-key'}

    def tearDown(self):
        app.JOBS.join()
        self.client.close()
        self.keys.stop()
        self.model.stop()

    def test_models_requires_auth_and_has_openai_shape(self):
        self.assertEqual(self.client.get('/v1/models').status_code, 401)
        data = self.client.get('/v1/models', headers=self.headers).json()
        self.assertEqual(data['object'], 'list')
        self.assertEqual(data['data'][0]['id'], 'openbmb/VoxCPM2')
        self.assertEqual(self.client.get('/v1/models', headers={'X-API-Key': 'test-key'}).status_code, 200)

    def test_pcm_stream_and_nonstream_include_all_sentences(self):
        for streaming in (False, True):
            response = self.client.post('/v1/audio/speech', headers=self.headers,
                json={'input': '第一句。第二句。', 'stream': streaming, 'response_format': 'pcm'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['X-Sample-Rate'], '48000')
            self.assertEqual(len(response.content), 2 * 3 * 4800 * 2)

    def test_sse_finishes_and_player_assets_load(self):
        response = self.client.post('/v1/audio/speech', headers=self.headers,
            json={'input': '你好。', 'stream_format': 'sse', 'response_format': 'wav'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('speech.audio.delta', response.text)
        self.assertTrue(response.text.endswith('data: {"type":"speech.audio.done"}\n\n'))
        self.assertEqual(self.client.get('/player').status_code, 200)
        self.assertIn('javascript', self.client.get('/player-assets/pcm-ring.mjs').headers['Content-Type'])

    def test_cancel_releases_full_queue_and_cleans_reference(self):
        closed = threading.Event()

        class EndlessTTS(FakeTTS):
            def _generate_with_prompt_cache(self, **kwargs):
                try:
                    while True:
                        yield torch.zeros(1, 80), None, None
                finally:
                    closed.set()

        async def cancel():
            with tempfile.NamedTemporaryFile(delete=False) as reference:
                path = reference.name
            with patch.object(app, 'MODEL', SimpleNamespace(tts_model=EndlessTTS())):
                results = app.run_on_gpu(app.build_kwargs('你好。', reference_audio=path), True, path)
                await anext(results)
                await asyncio.sleep(0.15)  # Let the producer fill its bounded queue.
                await results.aclose()
                self.assertTrue(await asyncio.to_thread(closed.wait, 2))
                await asyncio.to_thread(app.JOBS.join)
                self.assertFalse(Path(path).exists())

        asyncio.run(cancel())


if __name__ == '__main__':
    unittest.main()
