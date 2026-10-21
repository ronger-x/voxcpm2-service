"""Design a voice from a description and save it as a preset reference in voices/.

Calls the running service with instructions (no reference), so the result is a new
random speaker matching the description. Listen to the output and rerun until it fits.
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

DEFAULT_TEXT = '你好，很高兴认识你。今天的天气不错，我们一起出去走走吧。有什么需要帮忙的，随时告诉我。'


def design(url, key, instructions, text, timesteps):
    request = urllib.request.Request(
        url.rstrip('/') + '/v1/audio/speech',
        data=json.dumps({'input': text, 'instructions': instructions, 'response_format': 'wav',
                         'inference_timesteps': timesteps}).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', help='音色名，即请求中的 voice；default 为未指定 voice 时的默认音色')
    parser.add_argument('instructions', help='音色描述，例如：年轻女性，温柔自然，语速适中')
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    parser.add_argument('--text', default=DEFAULT_TEXT, help='参考音频朗读的文本，建议 5–15 秒')
    parser.add_argument('--steps', type=int, default=16, help='推理步数，参考音频只生成一次，可以用较高步数')
    parser.add_argument('--dir', default=os.environ.get('VOXCPM_VOICES_DIR', 'voices'))
    parser.add_argument('--force', action='store_true', help='覆盖已存在的同名音色')
    args = parser.parse_args()

    target = Path(args.dir) / f'{args.name}.wav'
    if target.exists() and not args.force:
        parser.error(f'{target} 已存在，确认替换时加 --force')
    audio = design(args.url, os.environ.get('VOXCPM_API_KEY', ''), args.instructions, args.text, args.steps)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(audio)
    # 同名 .txt 记录参考文本，服务端据此改用续写式克隆
    target.with_suffix('.txt').write_text(args.text.strip() + '\n', encoding='utf-8')
    print(f'已保存 {target}（{len(audio) / 1024:.0f} KiB），试听满意后即可在请求中使用 voice={args.name!r}')
