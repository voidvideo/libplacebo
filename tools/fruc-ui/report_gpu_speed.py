"""Summarize GPU-resident Vulkan and TensorRT throughput; exclude build time."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    runs = {str(p.parent.relative_to(args.results)): json.loads(p.read_text())
            for backend in ('ncnn', 'tensorrt')
            for p in sorted((args.results/backend).glob('*/run.json'))}
    rows = []
    summary = json.loads(json.dumps(runs))
    for name, record in summary.items():
        record.pop('milliseconds', None)
        record.pop('batch_wall_ms', None)
        rate = record.get('generated_fps')
        values = [name, record['status'], f'{rate:.2f}' if rate else '—',
                  'yes' if rate and rate >= 48 else 'no' if rate else '—']
        rows.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in values)+'</tr>')
    page = '''<!doctype html><meta charset="utf-8"><title>4K GPU-resident interpolation speed</title>
<style>body{font:16px system-ui;background:#171b22;color:#eee;margin:24px}td,th{padding:10px;text-align:left;border-bottom:1px solid #555}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{max-width:1050px}</style>
<h1>4K GPU-resident interpolation speed</h1><p>RTX 5070 · 3840×2160 · 24→60 · target: 48 generated frames/s.</p>
<p>Inputs and outputs remain on the GPU during timing. Model loading, engine compilation, input upload and output readback are excluded. Each run lasts at least fifteen seconds after warmup. Tests execute sequentially on an active desktop.</p>
<p>ncnn uses independent computation workers. TensorRT uses FP16, CUDA graphs and separate execution contexts/streams. TensorRT processes eight queued inferences per stream per measured batch. These are backend throughput measurements, not live libplacebo playback tests.</p>
<table><tr><th>Configuration</th><th>Status</th><th>Generated frames/s</th><th>Meets 48/s</th></tr>ROWS</table>
<h2>Settings and evidence</h2><pre>SUMMARY</pre>'''
    page = page.replace('ROWS', ''.join(rows)).replace('SUMMARY', html.escape(json.dumps(summary, indent=2)))
    (args.results/'index.html').write_text(page)
    for name, data in [('runs.json', runs), ('summary.json', summary)]:
        (args.results/name).write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()
