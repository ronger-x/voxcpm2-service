"""Measure PCM arrival latency/RTF with the same 0.6-second playback prefill."""

import argparse
import json
import os
import time
import urllib.request


def measure(url, key, text, split=True):
    request = urllib.request.Request(
        url.rstrip('/') + '/v1/audio/speech',
        data=json.dumps({'input': text, 'stream': True, 'response_format': 'pcm',
                         'split_sentences': split, 'inference_timesteps': 10}).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'},
    )
    start = time.perf_counter()
    arrivals = []
    with urllib.request.urlopen(request, timeout=900) as response:
        rate = int(response.headers['X-Sample-Rate'])
        while data := response.read1(65536):
            arrivals.append((time.perf_counter() - start, len(data) / (2 * rate)))
    wall = time.perf_counter() - start
    duration = sum(duration for _, duration in arrivals)
    if not duration:
        raise RuntimeError('No audio received')
    playing = False
    buffered = 0
    previous = 0
    underruns = 0
    for when, audio in arrivals:
        if playing:
            buffered -= when - previous
            if buffered < 0:
                underruns += 1
                buffered = 0
                playing = False
        buffered += audio
        if buffered >= 0.6:
            playing = True
        previous = when
    return {'chars': len(text), 'split': split, 'audio_seconds': round(duration, 3),
            'wall_seconds': round(wall, 3), 'rtf': round(wall / duration, 3),
            'first_seconds': round(arrivals[0][0], 3), 'underruns_600ms': underruns}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--no-split', action='store_true')
    args = parser.parse_args()
    key = os.environ.get('VOXCPM_API_KEY', '')
    text = ('今天天气不错，我们一起去公园散步吧。路上可以看到很多花，空气也特别清新。'
            '走累了就在湖边坐一会儿，看看水里的鱼。')
    for run in range(args.runs):
        print(json.dumps({'run': run, **measure(args.url, key, text, not args.no_split)}, ensure_ascii=False), flush=True)
