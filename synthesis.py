"""Sentence segmentation and a fixed voice reference for each request."""

import re
import subprocess
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import soundfile as sf


def split_sentences(text, max_chars=100):
    """Keep punctuation/closing quotes; preserve decimals and English words."""
    if max_chars < 1:
        raise ValueError('max_chars must be positive')
    text = text.strip()
    # A period is a boundary only before whitespace/end, excluding decimals,
    # URLs and abbreviations without a following space.
    parts = re.split(r'(?<=[。！？!?；;])(?=[^。！？!?；;”’"\')）】」』])|(?<=\.)\s+|\n+', text)
    sentences = []
    for part in parts:
        part = part.strip()
        # Separate a quoted sentence after its closing punctuation as well.
        quoted = re.split(r'(?<=[”’"\')）】」』])(?<=[。！？!?][”’"\')）】」』])\s*', part)
        for piece in quoted:
            piece = piece.strip()
            while len(piece) > max_chars:
                candidates = list(re.finditer(r'[，,、：:]|\s+', piece[:max_chars]))
                cut = candidates[-1].end() if candidates else max_chars
                sentences.append(piece[:cut].strip())
                piece = piece[cut:].strip()
            if piece:
                sentences.append(piece)
    return sentences


def merge_controls(*controls):
    """Join the request-wide description with a segment's style into one parenthesised prefix."""
    parts = [control.strip() for control in controls if control and control.strip()]
    return '，'.join(parts) or None


def plan_sentences(text, control=None, segments=None, split=True):
    """Return [(sentence, control, prosody)].

    Each segment may add its own style to the request-wide control and carry a prosody;
    only the segment's first sentence keeps the leading pause. A segment `fade` (dB across the
    whole segment) becomes a per-sentence `ramp` (start dB, end dB) split by character count,
    so sentences still stream one by one.
    """
    pieces = segments if segments else [{'text': text}]
    plan = []
    for piece in pieces:
        piece_control = merge_controls(control, piece.get('instructions'))
        piece_text = piece['text'].strip()
        sentences = [s for s in (split_sentences(piece_text) if split else [piece_text]) if s]
        fade = (piece.get('prosody') or {}).get('fade', 0)
        total = sum(len(sentence) for sentence in sentences) or 1
        done = 0
        for index, sentence in enumerate(sentences):
            prosody = dict(piece.get('prosody') or {})
            prosody.pop('fade', None)
            if index:
                prosody.pop('pause', None)
            if fade:
                prosody['ramp'] = (round(fade * done / total, 3), round(fade * (done + len(sentence)) / total, 3))
            done += len(sentence)
            plan.append((sentence, piece_control, prosody or None))
    return plan


def prosody_filter(prosody):
    """ffmpeg filter for rate (tempo factor) and pitch (semitones). None when neither changes.

    Rate alone uses atempo: on the same take, rubberband at 0.85x brightened the voice (spectral
    centroid +80-120 Hz, speaker similarity 0.97) while atempo kept it (0.985). rubberband is only
    used when the pitch moves, with formants preserved.
    """
    rate = prosody.get('rate', 1.0)
    pitch = prosody.get('pitch', 0.0)
    if pitch:
        return f'rubberband=tempo={rate:g}:pitch={2 ** (pitch / 12):.6f}:formant=preserved:pitchq=quality'
    if rate != 1:
        return f'atempo={rate:g}'
    return None


def needs_whole_sentence(prosody):
    """Tempo/pitch filters and gain ramps need the whole sentence before any of it is released."""
    return bool(prosody and (prosody_filter(prosody) or prosody.get('ramp')))


def apply_gain(pcm, prosody):
    """Constant `volume` plus an optional linear-in-dB `ramp` across the given audio."""
    prosody = prosody or {}
    gain = prosody.get('volume', 0.0)
    start, end = prosody.get('ramp') or (0.0, 0.0)
    if not gain and not start and not end:
        return pcm
    db = gain + np.linspace(start, end, len(pcm), dtype=np.float32) if (start or end) else gain
    return np.clip(pcm * 10 ** (db / 20), -1.0, 1.0).astype(np.float32)


def apply_prosody(pcm, sample_rate, prosody):
    """Reshape one whole sentence: tempo, pitch and the gain ramp."""
    af = prosody_filter(prosody)
    if af:
        result = subprocess.run(
            ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'f32le', '-ar', str(sample_rate), '-ac', '1',
             '-i', 'pipe:0', '-af', af, '-f', 'f32le', 'pipe:1'],
            input=pcm.astype('<f4').tobytes(), capture_output=True, check=True,
        )
        pcm = np.frombuffer(result.stdout, '<f4').copy()
    return apply_gain(pcm, prosody)


def build_reference_cache(tts, reference, prompt_text=None):
    """With the transcript, also continue from the same clip (upstream "ultimate cloning"),
    which upstream recommends for the closest timbre; without it, clone from audio alone."""
    if prompt_text and prompt_text.strip():
        return tts.build_prompt_cache(
            prompt_text=prompt_text.strip(), prompt_wav_path=reference, reference_wav_path=reference
        )
    return tts.build_prompt_cache(reference_wav_path=reference)


class TorchBackend:
    """The upstream `voxcpm` PyTorch path (`VOXCPM_ENGINE=torch`)."""

    def __init__(self, tts, options, streaming):
        self.tts, self.options, self.streaming = tts, options, streaming
        self.sample_rate = tts.sample_rate

    def voice(self, path, prompt_text=None):
        return build_reference_cache(self.tts, path, prompt_text)

    def generate(self, text, voice, cancelled):
        with closing(self.tts._generate_with_prompt_cache(
            target_text=text,
            prompt_cache=voice,
            streaming=self.streaming,
            retry_badcase=not self.streaming,
            **self.options,
        )) as chunks:
            for wav, _, _ in chunks:
                if cancelled.is_set():
                    return
                yield wav.squeeze(0).cpu().numpy().copy()


class NanoBackend:
    """`nano_engine.NanoEngine`; diffusion steps and CFG are fixed when the engine starts."""

    def __init__(self, engine, options, streaming):
        self.engine, self.streaming = engine, streaming
        self.cfg_value = options.get('cfg_value', 2.0)
        self.max_len = options.get('max_len', 2000)
        self.sample_rate = engine.sample_rate

    def voice(self, path, prompt_text=None):
        return self.engine.voice(path, prompt_text)

    def generate(self, text, voice, cancelled):
        return self.engine.generate(text, voice, cancelled, streaming=self.streaming,
                                    cfg_value=self.cfg_value, max_len=self.max_len)


def generate_segments(model, kwargs, streaming, cancelled):
    """Yield native PCM. Encode an uploaded/first-sentence reference only once."""
    options = dict(kwargs)
    segments = options.pop('segments', None)
    plan = plan_sentences(
        options.pop('text'), options.pop('control', None), segments, options.pop('split_sentences', True)
    )
    reference = options.pop('reference_wav_path', None)
    reference_text = options.pop('reference_text', None)
    # Upstream ignores the parenthesised control in transcript ("Hi-Fi") cloning, so segment
    # styles switch the whole request to reference-only cloning, which honours them.
    if any(segment.get('instructions') for segment in segments or []):
        reference_text = None
    backend = (TorchBackend(model.tts_model, options, streaming) if hasattr(model, 'tts_model')
               else NanoBackend(model, options, streaming))
    if cancelled.is_set():
        return
    voice = backend.voice(reference, reference_text) if reference else None
    for index, (sentence, control, prosody) in enumerate(plan):
        if cancelled.is_set():
            return
        sentence = re.sub(r'\s+', ' ', sentence)
        prompt = f'({control}){sentence}' if control else sentence
        first_chunks = []
        held = [] if needs_whole_sentence(prosody) else None
        needs_anchor = index == 0 and voice is None and len(plan) > 1
        pause = (prosody or {}).get('pause', 0)
        if pause:
            yield np.zeros(int(pause * backend.sample_rate), dtype=np.float32)
        with closing(backend.generate(prompt, voice, cancelled)) as chunks:
            for pcm in chunks:
                if cancelled.is_set():
                    return
                if needs_anchor:
                    first_chunks.append(pcm)
                if held is None:
                    yield apply_gain(pcm, prosody)
                else:
                    held.append(pcm)
        if held and not cancelled.is_set():
            yield apply_prosody(np.concatenate(held), backend.sample_rate, prosody)
        if needs_anchor and first_chunks and not cancelled.is_set():
            # Keep this reference fixed; using the previous sentence accumulates
            # voice drift. The uploaded reference, when present, takes priority.
            with tempfile.TemporaryDirectory(prefix='voxcpm-anchor-') as directory:
                path = Path(directory) / 'anchor.wav'
                sf.write(path, np.concatenate(first_chunks), backend.sample_rate)
                voice = backend.voice(str(path))
