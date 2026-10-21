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
    only the segment's first sentence keeps the leading pause.
    """
    pieces = segments if segments else [{'text': text}]
    plan = []
    for piece in pieces:
        piece_control = merge_controls(control, piece.get('instructions'))
        piece_text = piece['text'].strip()
        sentences = [s for s in (split_sentences(piece_text) if split else [piece_text]) if s]
        for index, sentence in enumerate(sentences):
            prosody = dict(piece.get('prosody') or {})
            if index:
                prosody.pop('pause', None)
            plan.append((sentence, piece_control, prosody or None))
    return plan


def prosody_filter(prosody):
    """ffmpeg rubberband filter for rate (tempo factor) and pitch (semitones); formants are kept
    so the cloned timbre survives the shift. None when neither changes."""
    rate = prosody.get('rate', 1.0)
    pitch = prosody.get('pitch', 0.0)
    if rate == 1 and pitch == 0:
        return None
    return f'rubberband=tempo={rate:g}:pitch={2 ** (pitch / 12):.6f}:formant=preserved:pitchq=quality'


def apply_gain(pcm, prosody):
    gain = (prosody or {}).get('volume', 0.0)
    return np.clip(pcm * 10 ** (gain / 20), -1.0, 1.0).astype(np.float32) if gain else pcm


def apply_prosody(pcm, sample_rate, prosody):
    """Reshape one whole sentence: rubberband needs the full sentence, so it cannot stream."""
    result = subprocess.run(
        ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'f32le', '-ar', str(sample_rate), '-ac', '1',
         '-i', 'pipe:0', '-af', prosody_filter(prosody), '-f', 'f32le', 'pipe:1'],
        input=pcm.astype('<f4').tobytes(), capture_output=True, check=True,
    )
    return apply_gain(np.frombuffer(result.stdout, '<f4').copy(), prosody)


def build_reference_cache(tts, reference, prompt_text=None):
    """With the transcript, also continue from the same clip (upstream "ultimate cloning"),
    which upstream recommends for the closest timbre; without it, clone from audio alone."""
    if prompt_text and prompt_text.strip():
        return tts.build_prompt_cache(
            prompt_text=prompt_text.strip(), prompt_wav_path=reference, reference_wav_path=reference
        )
    return tts.build_prompt_cache(reference_wav_path=reference)


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
    tts = model.tts_model
    if cancelled.is_set():
        return
    prompt_cache = build_reference_cache(tts, reference, reference_text) if reference else None
    for index, (sentence, control, prosody) in enumerate(plan):
        if cancelled.is_set():
            return
        sentence = re.sub(r'\s+', ' ', sentence)
        prompt = f'({control}){sentence}' if control else sentence
        first_chunks = []
        held = [] if prosody and prosody_filter(prosody) else None
        needs_anchor = index == 0 and prompt_cache is None and len(plan) > 1
        pause = (prosody or {}).get('pause', 0)
        if pause:
            yield np.zeros(int(pause * tts.sample_rate), dtype=np.float32)
        with closing(tts._generate_with_prompt_cache(
            target_text=prompt,
            prompt_cache=prompt_cache,
            streaming=streaming,
            retry_badcase=not streaming,
            **options,
        )) as chunks:
            for wav, _, _ in chunks:
                if cancelled.is_set():
                    return
                pcm = wav.squeeze(0).cpu().numpy().copy()
                if needs_anchor:
                    first_chunks.append(pcm)
                if held is None:
                    yield apply_gain(pcm, prosody)
                else:
                    held.append(pcm)
        if held and not cancelled.is_set():
            yield apply_prosody(np.concatenate(held), tts.sample_rate, prosody)
        if needs_anchor and not cancelled.is_set():
            # Keep this reference fixed; using the previous sentence accumulates
            # voice drift. The uploaded reference, when present, takes priority.
            with tempfile.TemporaryDirectory(prefix='voxcpm-anchor-') as directory:
                path = Path(directory) / 'anchor.wav'
                sf.write(path, np.concatenate(first_chunks), tts.sample_rate)
                prompt_cache = tts.build_prompt_cache(reference_wav_path=str(path))
