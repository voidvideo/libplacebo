"""Render GPU-resident and filter timings from the ncnn/Vulkan benchmark."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    runs = {str(p.parent.relative_to(args.results)): json.loads(p.read_text())
            for group in ('resident', 'timing')
            for p in sorted((args.results/group).glob('*/run.json'))}
    rows = []
    for name, run in runs.items():
        timing = run.get('timing', run)
        values = [name, run['status'], timing.get('generated_fps', '—')]
        rows.append('<tr>'+''.join('<td>'+html.escape(f'{v:.2f}' if isinstance(v, float) else str(v))+'</td>' for v in values)+'</tr>')
    summary = json.loads(json.dumps(runs))
    for run in summary.values():
        run.pop('board_memory_samples', None)
        run.pop('milliseconds', None)
        for key in ('request_latencies_ms', 'completion_seconds', 'output_indices'):
            run.get('timing', {}).pop(key, None)
    page = '''<!doctype html><meta charset="utf-8"><title>Native 4K ncnn/Vulkan speed test</title>
<style>body{background:#171b22;color:#eee;font:16px system-ui;margin:24px}td,th{padding:10px;text-align:left;border-bottom:1px solid #555}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{max-width:1050px}</style>
<h1>Native 4K ncnn/Vulkan speed test</h1>
<p>RTX 5070 · 3840×2160 · 24→60. Exact output timestamps require 48 generated frames per second. Twelve outputs coincide with source timestamps.</p>
<p><strong>Resident rows are the relevant deployment measurement:</strong> input and output stay on the GPU throughout the timed interval. They measure serial graph execution, command submission and synchronization. There is no timed CPU upload or readback. This is not an integrated libplacebo frame-queue test.</p>
<p>Secondary timing rows measure the VapourSynth filter including CPU packing, upload and readback. They do not decide GPU-resident feasibility. Failed runs have no throughput result. Quality evaluation is outside this speed test.</p>
<table><tr><th>Configuration / boundary</th><th>Status</th><th>Generated frames/s</th></tr>ROWS</table>
<h2>Recorded settings</h2><pre>SUMMARY</pre>'''
    page = page.replace('ROWS', ''.join(rows)).replace('SUMMARY', html.escape(json.dumps(summary, indent=2)))
    (args.results/'index.html').write_text(page)
    (args.results/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.results/'runs.json').write_text(json.dumps(runs, indent=2)+'\n')


if __name__ == '__main__':
    main()
