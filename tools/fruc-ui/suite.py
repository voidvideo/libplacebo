"""Deterministic menu benchmark data; no interpolation implementation.

Run with system Python (NumPy and Pillow). All outputs are lossless RGB PNGs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, COUNT = 960, 540, 25
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
SMALL = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
SCENES = ('scroll_slow', 'scroll_fast', 'vertical', 'focus_glide', 'crossing', 'static')


def scene(n, kind):
    im = Image.new('RGB', (W, H), (22, 26, 35))
    d = ImageDraw.Draw(im)
    d.text((28, 16), 'VIDEO LIBRARY     Movies     Shows     Settings', font=FONT, fill='white')
    d.rectangle((24, 64, 936, 500), fill=(42, 48, 62))
    if kind in ('scroll_slow', 'scroll_fast', 'static'):
        speed = {'scroll_slow': 8, 'scroll_fast': 48, 'static': 0}[kind]
        for k in range(16):
            x = round(50 + k * 160 - n * speed)
            if x > W or x + 140 < 0:
                continue
            d.rectangle((x, 110, x + 138, 355), fill=((70+k*31)%200, (80+k*43)%190, (100+k*23)%210), outline='white', width=2)
            for y in range(140, 330, 32):
                d.line((x+12, y, x+125, y+16), fill=(205, 205, 205), width=3)
            d.text((x+8, 370), f'MOVIE {k+1}', font=FONT, fill='white')
        d.rectangle((210, 100, 358, 410), outline=(60, 200, 255), width=5)
    elif kind in ('vertical', 'focus_glide'):
        titles = ['Picture settings', 'Audio output', 'Network connection', 'Display calibration', 'System information']
        for k in range(12):
            y = round(95 + k*65 - (n*12 if kind == 'vertical' else 0))
            if 70 <= y <= 450:
                d.rectangle((55, y, 585, y+57), fill=(55, 61, 74))
                d.text((78, y+14), titles[k%5], font=FONT, fill='white')
        y = round(95 + 260*(0.5-0.5*np.cos(n*np.pi/24))) if kind == 'focus_glide' else 160
        d.rectangle((53, y, 587, y+59), outline=(60, 200, 255), width=4)
        d.text((650, 180), 'FIXED PANEL', font=FONT, fill='white')
    else:
        for x, y, color, label in [(round(60+n*24), 180, (160, 60, 90), 'MOVING RIGHT'), (round(720-n*24), 260, (40, 150, 190), 'MOVING LEFT')]:
            d.rectangle((x, y, x+210, y+100), fill=color, outline='white', width=2)
            d.text((x+10, y+35), label, font=FONT, fill='white')
    d.rectangle((0, 510, W, H), fill=(10, 12, 18))
    d.text((25, 516), 'BACK     SELECT                         MENU INTERPOLATION BENCHMARK', font=SMALL, fill='white')
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    args = p.parse_args()
    manifest = {'schema': 1, 'width': W, 'height': H, 'source_fps': 24,
                'source_count': COUNT, 'scenes': list(SCENES), 'sha256': {},
                'tracks': {'midpoint': [(i+.5)/24 for i in range(24)],
                           '24to60': [i/60 for i in range(60)]}}
    for name in SCENES:
        root = args.output / name
        for track, times in [('input', [i/24 for i in range(COUNT)]), *manifest['tracks'].items()]:
            dest = root / track
            dest.mkdir(parents=True, exist_ok=True)
            for i, t in enumerate(times):
                path = dest / f'{i:04}.png'
                scene(t*24, name).save(path)
                manifest['sha256'][str(path.relative_to(args.output))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with (root/'input.rgba').open('wb') as f:
            for i in range(COUNT):
                f.write(scene(i, name).convert('RGBA').tobytes())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
