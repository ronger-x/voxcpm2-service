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

    def build_prompt_cache(self, reference_wav_path, prompt_text=None, prompt_wav_path=None):
        return {'ref': str(reference_wav_path)}

    def _generate_with_prompt_cache(self, **kwargs):
        for _ in range(3):
            yield torch.full((1, 4800), 0.125), None, None


class RecordingTTS(FakeTTS):
    def __init__(self):
        self.references = []
        self.transcripts = []
        self.prompts = []

    def build_prompt_cache(self, reference_wav_path, prompt_text=None, prompt_wav_path=None):
        self.references.append(reference_wav_path)
        self.transcripts.append(prompt_text)
        return super().build_prompt_cache(reference_wav_path)

    def _generate_with_prompt_cache(self, **kwargs):
        self.prompts.append(kwargs['target_text'])
        return super()._generate_with_prompt_cache(**kwargs)


class APITests(unittest.TestCase):
    def setUp(self):
        self.voices_dir = tempfile.TemporaryDirectory()
        self.voices = patch.object(app, 'VOICES_DIR', Path(self.voices_dir.name))
        self.keys = patch.object(app, 'API_KEYS', ['test-key'])
        self.model = patch.object(app, 'MODEL', SimpleNamespace(tts_model=FakeTTS()))
        self.voices.start()
        self.keys.start()
        self.model.start()
        self.client = TestClient(app.api)
        self.headers = {'Authorization': 'Bearer test-key'}

    def tearDown(self):
        app.JOBS.join()
        self.client.close()
        self.keys.stop()
        self.model.stop()
        self.voices.stop()
        self.voices_dir.cleanup()

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

    def test_preset_voices_are_listed_and_selected_by_voice(self):
        for name in ('default', 'alice', '小美'):
            Path(self.voices_dir.name, f'{name}.wav').write_bytes(b'RIFF')
        Path(self.voices_dir.name, 'notes.txt').write_text('ignored')
        self.assertEqual(self.client.get('/v1/voices').status_code, 401)
        listed = self.client.get('/v1/voices', headers=self.headers).json()
        self.assertEqual([voice['id'] for voice in listed['data']], ['alice', 'default', '小美'])

        tts = RecordingTTS()
        preset = lambda name: str(Path(self.voices_dir.name, f'{name}.wav'))
        cases = [
            ({'voice': 'alice'}, 'alice', preset('alice')),
            ({'voice': '小美'}, '%E5%B0%8F%E7%BE%8E', preset('小美')),
            ({'voice': 'nova'}, 'default', preset('default')),
            ({}, 'default', preset('default')),
            # 未知音色加 instructions：按描述设计音色，不套 default 参考
            ({'voice': 'nova', 'instructions': '温柔'}, 'none', None),
            ({'voice': 'alice', 'instructions': '温柔'}, 'alice', preset('alice')),
        ]
        with patch.object(app, 'MODEL', SimpleNamespace(tts_model=tts)):
            for body, header, reference in cases:
                tts.references.clear()
                response = self.client.post('/v1/audio/speech', headers=self.headers,
                    json={'input': '你好。', 'response_format': 'pcm', **body})
                self.assertEqual(response.status_code, 200, body)
                self.assertEqual(response.headers['X-Voice'], header, body)
                self.assertEqual(tts.references, [reference] if reference else [], body)

            tts.references.clear()
            response = self.client.post('/v1/audio/speech', headers=self.headers,
                json={'input': '你好。', 'voice': 'alice', 'reference_audio': 'UklGRg==', 'response_format': 'pcm'})
            self.assertEqual(response.headers['X-Voice'], 'reference_audio')
            self.assertNotEqual(tts.references, [preset('alice')])
            app.JOBS.join()
            self.assertFalse(Path(tts.references[0]).exists())
        # 预置文件只读使用，生成后不能被当作临时参考删除
        self.assertTrue(all(Path(preset(name)).exists() for name in ('default', 'alice', '小美')))

    def test_preset_transcript_and_uploaded_reference_text_are_used(self):
        Path(self.voices_dir.name, 'default.wav').write_bytes(b'RIFF')
        Path(self.voices_dir.name, 'default.txt').write_text(' 参考文本。\n', encoding='utf-8')
        Path(self.voices_dir.name, 'plain.wav').write_bytes(b'RIFF')
        tts = RecordingTTS()
        with patch.object(app, 'MODEL', SimpleNamespace(tts_model=tts)):
            for body, transcript in [
                ({'voice': 'default'}, '参考文本。'),
                ({'voice': 'plain'}, None),
                ({'reference_audio': 'UklGRg==', 'reference_text': '上传的文本'}, '上传的文本'),
                ({'reference_audio': 'UklGRg=='}, None),
            ]:
                tts.transcripts.clear()
                response = self.client.post('/v1/audio/speech', headers=self.headers,
                    json={'input': '你好。', 'response_format': 'pcm', **body})
                self.assertEqual(response.status_code, 200, body)
                self.assertEqual(tts.transcripts, [transcript], body)
        # .txt 不是音色，不应出现在列表里
        listed = self.client.get('/v1/voices', headers=self.headers).json()
        self.assertEqual([voice['id'] for voice in listed['data']], ['default', 'plain'])

    def test_segments_apply_their_style_and_keep_the_preset_voice(self):
        Path(self.voices_dir.name, 'default.wav').write_bytes(b'RIFF')
        Path(self.voices_dir.name, 'default.txt').write_text('参考文本。', encoding='utf-8')
        tts = RecordingTTS()
        with patch.object(app, 'MODEL', SimpleNamespace(tts_model=tts)):
            response = self.client.post('/v1/audio/speech', headers=self.headers, json={
                'voice': 'alloy',
                'response_format': 'pcm',
                'segments': [
                    {'text': '好一则趣话。', 'instructions': '带着笑意'},
                    {'text': '   '},
                    {'text': '你且说说。', 'instructions': '轻声'},
                ],
            })
        self.assertEqual(response.status_code, 200)
        # 段落风格不参与音色选择：仍用 default 预置音色，但改为只按参考音频克隆
        self.assertEqual(response.headers['X-Voice'], 'default')
        self.assertEqual(tts.transcripts, [None])
        self.assertEqual(tts.prompts, ['(带着笑意)好一则趣话。', '(轻声)你且说说。'])
        self.assertEqual(len(response.content), 2 * 3 * 4800 * 2)

        for body in ({'segments': [{'text': ' ', 'instructions': '轻声'}]}, {'segments': []}, {}):
            response = self.client.post('/v1/audio/speech', headers=self.headers, json=body)
            self.assertEqual(response.status_code, 400, body)

    def test_segment_prosody_is_validated_and_forwarded(self):
        captured = []
        original = app.generate_segments

        def recording(model, kwargs, streaming, cancelled):
            captured.append(kwargs.get('segments'))
            return original(model, kwargs, streaming, cancelled)

        with patch.object(app, 'generate_segments', recording):
            response = self.client.post('/v1/audio/speech', headers=self.headers, json={
                'response_format': 'pcm',
                'segments': [
                    {'text': '一。', 'prosody': {'volume': -3, 'pause': 0.25}},
                    {'text': '二。', 'prosody': {}},
                ],
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured[0], [
            {'text': '一。', 'instructions': None, 'prosody': {'volume': -3.0, 'pause': 0.25}},
            {'text': '二。', 'instructions': None, 'prosody': None},
        ])
        # 0.25 s 静音 + 两句各 3 块
        self.assertEqual(len(response.content), (12000 + 2 * 3 * 4800) * 2)

        for prosody in ({'rate': 3}, {'pitch': -13}, {'volume': 20}, {'pause': -1}):
            response = self.client.post('/v1/audio/speech', headers=self.headers,
                json={'segments': [{'text': '一。', 'prosody': prosody}]})
            self.assertEqual(response.status_code, 422, prosody)

    def test_without_presets_voice_is_ignored(self):
        tts = RecordingTTS()
        with patch.object(app, 'MODEL', SimpleNamespace(tts_model=tts)):
            response = self.client.post('/v1/audio/speech', headers=self.headers,
                json={'input': '你好。', 'voice': 'default', 'response_format': 'pcm'})
        self.assertEqual(response.headers['X-Voice'], 'none')
        self.assertEqual(tts.references, [])
        self.assertEqual(self.client.get('/v1/voices', headers=self.headers).json()['data'], [])

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
