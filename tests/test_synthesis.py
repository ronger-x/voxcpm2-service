import shutil
import subprocess
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from synthesis import apply_prosody, generate_segments, plan_sentences, prosody_filter, split_sentences


class FakeTTS:
    sample_rate = 48000

    def __init__(self):
        self.calls, self.references = [], []
        self.closed = 0

    def build_prompt_cache(self, reference_wav_path, prompt_text=None, prompt_wav_path=None):
        self.references.append(reference_wav_path)
        if 'anchor' in reference_wav_path:
            assert Path(reference_wav_path).is_file()
        return {'reference': reference_wav_path, 'prompt_text': prompt_text, 'prompt_wav': prompt_wav_path}

    def _generate_with_prompt_cache(self, **kwargs):
        self.calls.append(kwargs)
        try:
            for _ in range(2):
                yield torch.zeros(1, 80), None, None
        finally:
            self.closed += 1


class SynthesisTests(unittest.TestCase):
    def test_chinese_english_quotes_and_decimal(self):
        self.assertEqual(split_sentences('你好！世界。'), ['你好！', '世界。'])
        self.assertEqual(split_sentences('他说：“你好！”然后走开。'), ['他说：“你好！”', '然后走开。'])
        self.assertEqual(split_sentences('版本 3.14 正常。Hello world! Next sentence.'),
                         ['版本 3.14 正常。', 'Hello world!', 'Next sentence.'])
        self.assertEqual(split_sentences('  第一行\n第二行  '), ['第一行', '第二行'])

    def test_long_text_retains_all_nonspace_characters(self):
        text = '长句没有句号' * 55
        parts = split_sentences(text)
        self.assertEqual(''.join(parts), text)
        self.assertTrue(all(0 < len(part) <= 100 for part in parts))

    def test_fixed_first_sentence_reference_is_reused_and_deleted(self):
        tts = FakeTTS()
        result = list(generate_segments(SimpleNamespace(tts_model=tts),
                      {'text': '第一句。第二句。第三句。', 'control': '温柔'}, True, threading.Event()))
        self.assertEqual(len(result), 6)
        self.assertIsNone(tts.calls[0]['prompt_cache'])
        self.assertIs(tts.calls[1]['prompt_cache'], tts.calls[2]['prompt_cache'])
        self.assertEqual(len(tts.references), 1)
        self.assertFalse(Path(tts.references[0]).exists())
        self.assertEqual(tts.calls[1]['target_text'], '(温柔)第二句。')
        self.assertFalse(tts.calls[1]['retry_badcase'])

    def test_uploaded_reference_wins_and_is_encoded_once(self):
        tts = FakeTTS()
        list(generate_segments(SimpleNamespace(tts_model=tts),
             {'text': '一。二。三。', 'reference_wav_path': 'uploaded.wav'}, False, threading.Event()))
        self.assertEqual(tts.references, ['uploaded.wav'])
        self.assertTrue(all(call['prompt_cache'] is tts.calls[0]['prompt_cache'] for call in tts.calls))
        self.assertTrue(tts.calls[0]['retry_badcase'])

    def test_reference_transcript_enables_continuation_cloning(self):
        tts = FakeTTS()
        list(generate_segments(SimpleNamespace(tts_model=tts),
             {'text': '一。二。', 'reference_wav_path': 'voice.wav', 'reference_text': ' 参考文本。 '},
             True, threading.Event()))
        cache = tts.calls[0]['prompt_cache']
        self.assertEqual(cache, {'reference': 'voice.wav', 'prompt_text': '参考文本。', 'prompt_wav': 'voice.wav'})
        self.assertTrue(all(call['prompt_cache'] is cache for call in tts.calls))
        self.assertNotIn('reference_text', tts.calls[0])

        tts = FakeTTS()
        list(generate_segments(SimpleNamespace(tts_model=tts),
             {'text': '一。', 'reference_wav_path': 'voice.wav', 'reference_text': '  '}, True, threading.Event()))
        self.assertIsNone(tts.calls[0]['prompt_cache']['prompt_text'])
        self.assertIsNone(tts.calls[0]['prompt_cache']['prompt_wav'])

    def test_segments_carry_their_own_style_on_top_of_request_control(self):
        plan = plan_sentences('', '年轻女性', [
            {'text': '好一则趣话。倒是新颖。', 'instructions': '带着笑意'},
            {'text': ' 我偏是不做。 '},
            {'text': '你且说说。', 'instructions': ' 轻声，语速放慢 '},
        ])
        self.assertEqual(plan, [
            ('好一则趣话。', '年轻女性，带着笑意', None),
            ('倒是新颖。', '年轻女性，带着笑意', None),
            ('我偏是不做。', '年轻女性', None),
            ('你且说说。', '年轻女性，轻声，语速放慢', None),
        ])
        self.assertEqual(plan_sentences('一。二。', None, None), [('一。', None, None), ('二。', None, None)])
        self.assertEqual(plan_sentences('', None, [{'text': '一。二。', 'instructions': '平淡'}], split=False),
                         [('一。二。', '平淡', None)])

    def test_segment_prosody_applies_to_each_sentence_and_pauses_once(self):
        plan = plan_sentences('', None, [
            {'text': '一。二。', 'prosody': {'rate': 0.8, 'pause': 0.5}},
            {'text': '三。', 'prosody': {'pause': 0.3}},
        ])
        self.assertEqual(plan, [
            ('一。', None, {'rate': 0.8, 'pause': 0.5}),
            ('二。', None, {'rate': 0.8}),
            ('三。', None, {'pause': 0.3}),
        ])
        self.assertIsNone(prosody_filter({'volume': -6, 'pause': 0.5}))
        # 只变速用 atempo，音色变化最小；变调才用 rubberband
        self.assertEqual(prosody_filter({'rate': 0.9}), 'atempo=0.9')
        self.assertEqual(prosody_filter({'rate': 0.8, 'pitch': 12}),
                         'rubberband=tempo=0.8:pitch=2.000000:formant=preserved:pitchq=quality')

    def test_prosody_reshapes_whole_sentences_and_inserts_pauses(self):
        tts = FakeTTS()
        calls = []

        def fake_apply(pcm, rate, prosody):
            calls.append((len(pcm), rate, prosody))
            return np.ones(len(pcm) // 2, dtype=np.float32)

        with patch('synthesis.apply_prosody', fake_apply):
            result = list(generate_segments(SimpleNamespace(tts_model=tts), {
                'text': '',
                'segments': [
                    {'text': '一。', 'prosody': {'rate': 1.25, 'pause': 0.5}},
                    {'text': '二。', 'prosody': {'volume': -6.0206}},
                    {'text': '三。'},
                ],
            }, True, threading.Event()))
        # 变速 / 变调的句子攒成整句后处理一次；只调音量的句子仍按块流式输出
        self.assertEqual(calls, [(160, 48000, {'rate': 1.25, 'pause': 0.5})])
        self.assertEqual([len(chunk) for chunk in result], [24000, 80, 80, 80, 80, 80])
        self.assertTrue(np.all(result[0] == 0))
        self.assertTrue(np.allclose(result[1], 1.0))

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg 不可用')
    def test_apply_prosody_rate_keeps_pitch(self):
        rate = 48000
        tone = (0.5 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)).astype(np.float32)
        slower = apply_prosody(tone, rate, {'rate': 0.8, 'volume': -6.0206})
        self.assertAlmostEqual(len(slower) / rate, 1.25, delta=0.05)
        spectrum = np.abs(np.fft.rfft(slower * np.hanning(len(slower))))
        self.assertAlmostEqual(np.fft.rfftfreq(len(slower), 1 / rate)[spectrum.argmax()], 220, delta=3)
        self.assertAlmostEqual(np.abs(slower[rate // 4:-rate // 4]).max(), 0.25, delta=0.02)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg 不可用')
    def test_apply_prosody_changes_duration_and_pitch(self):
        rate = 48000
        tone = (0.5 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)).astype(np.float32)
        try:
            shifted = apply_prosody(tone, rate, {'rate': 0.8, 'pitch': 2})
        except subprocess.CalledProcessError as error:
            if b'rubberband' in error.stderr or b'No such filter' in error.stderr:
                self.skipTest('ffmpeg 未编译 rubberband')
            raise
        self.assertAlmostEqual(len(shifted) / rate, 1.25, delta=0.05)
        spectrum = np.abs(np.fft.rfft(shifted * np.hanning(len(shifted))))
        peak = np.fft.rfftfreq(len(shifted), 1 / rate)[spectrum.argmax()]
        self.assertAlmostEqual(peak, 220 * 2 ** (2 / 12), delta=5)

    def test_segment_styles_use_one_reference_and_drop_transcript_cloning(self):
        tts = FakeTTS()
        list(generate_segments(SimpleNamespace(tts_model=tts), {
            'text': '',
            'segments': [{'text': '好一则趣话。', 'instructions': '带着笑意'}, {'text': '我偏是不做。'}],
            'reference_wav_path': 'voice.wav',
            'reference_text': '参考文本。',
        }, True, threading.Event()))
        # 上游在续写式克隆下忽略括号指令，带风格时只按参考音频克隆
        self.assertEqual(tts.references, ['voice.wav'])
        self.assertIsNone(tts.calls[0]['prompt_cache']['prompt_text'])
        self.assertEqual([call['target_text'] for call in tts.calls], ['(带着笑意)好一则趣话。', '我偏是不做。'])
        self.assertTrue(all(call['prompt_cache'] is tts.calls[0]['prompt_cache'] for call in tts.calls))

        # 没有分段语气时保持原行为：只有整段描述也继续用续写式克隆
        for options in ({'text': '', 'segments': [{'text': '一。'}, {'text': '二。'}]},
                        {'text': '一。', 'control': '温柔'}):
            tts = FakeTTS()
            list(generate_segments(SimpleNamespace(tts_model=tts), {
                **options, 'reference_wav_path': 'voice.wav', 'reference_text': '参考文本。',
            }, True, threading.Event()))
            self.assertEqual(tts.calls[0]['prompt_cache']['prompt_text'], '参考文本。', options)

    def test_cancel_closes_active_generator_without_starting_next_sentence(self):
        tts, cancelled = FakeTTS(), threading.Event()
        result = generate_segments(SimpleNamespace(tts_model=tts), {'text': '第一句。第二句。'}, True, cancelled)
        next(result)
        cancelled.set()
        self.assertEqual(list(result), [])
        self.assertEqual(len(tts.calls), 1)
        self.assertEqual(tts.closed, 1)
        self.assertEqual(tts.references, [])

    def test_split_can_be_disabled(self):
        tts = FakeTTS()
        list(generate_segments(SimpleNamespace(tts_model=tts),
             {'text': '第一句。\n第二句。', 'split_sentences': False}, True, threading.Event()))
        self.assertEqual(len(tts.calls), 1)
        self.assertEqual(tts.calls[0]['target_text'], '第一句。 第二句。')


if __name__ == '__main__':
    unittest.main()
