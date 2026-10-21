"""Sentence segmentation and a fixed voice reference for each request."""

import re
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


def generate_segments(model, kwargs, streaming, cancelled):
    """Yield native PCM. Encode an uploaded/first-sentence reference only once."""
    options = dict(kwargs)
    text = options.pop('text')
    control = options.pop('control', None)
    split = options.pop('split_sentences', True)
    reference = options.pop('reference_wav_path', None)
    sentences = split_sentences(text) if split else [text]
    tts = model.tts_model
    if cancelled.is_set():
        return
    prompt_cache = tts.build_prompt_cache(reference_wav_path=reference) if reference else None
    for index, sentence in enumerate(sentences):
        if cancelled.is_set():
            return
        sentence = re.sub(r'\s+', ' ', sentence)
        prompt = f'({control.strip()}){sentence}' if control and control.strip() else sentence
        first_chunks = []
        needs_anchor = index == 0 and prompt_cache is None and len(sentences) > 1
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
                yield pcm
        if needs_anchor and not cancelled.is_set():
            # Keep this reference fixed; using the previous sentence accumulates
            # voice drift. The uploaded reference, when present, takes priority.
            with tempfile.TemporaryDirectory(prefix='voxcpm-anchor-') as directory:
                path = Path(directory) / 'anchor.wav'
                sf.write(path, np.concatenate(first_chunks), tts.sample_rate)
                prompt_cache = tts.build_prompt_cache(reference_wav_path=str(path))
