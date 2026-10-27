"""Build a local report from completed or failed native 4K RIFE runs."""
import argparse
import html
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    args = p.parse_args()
    runs = {f.parent.name: json.loads(f.read_text()) for f in sorted(args.results.glob('*/run.json'))}
    rows = []
    for name, run in runs.items():
        speed = run.get('resident_inference', {})
        values = [name, run['status'], speed.get('generated_fps'), speed.get('p95_ms'),
                  speed.get('max_ms'), run.get('inference_peak_allocated_mib')]
        rows.append('<tr>'+''.join('<td>'+html.escape(f'{v:.2f}' if isinstance(v, float) else str(v))+'</td>' for v in values)+'</tr>')
    catalog = {}
    for name, run in runs.items():
        if run['status'] != 'complete': continue
        for scene in run['scenes']:
            entry = catalog.setdefault(scene, {})
            for f in sorted((args.results/name/scene).glob('*-reference.png')):
                i = f.stem.split('-')[0]
                frame = entry.setdefault(i, {})
                frame.setdefault('Reference', str(f.resolve()))
                frame[name] = str(f.with_name(i+'.png').resolve())
    page = '''<!doctype html><meta charset="utf-8"><title>4K RIFE menu benchmark</title>
<style>body{background:#171b22;color:#eee;font:16px system-ui;margin:24px}td,th{padding:10px;text-align:left;border-bottom:1px solid #555}select{font:inherit;margin:8px}#frames{display:flex;flex-wrap:wrap;gap:20px}figure{margin:0;width:47%}img,canvas{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{max-width:1050px}</style>
<h1>4K RIFE menu benchmark — historical PyTorch results</h1><p><strong>Excluded from the ncnn/Vulkan feasibility decision.</strong> These measurements use a different runtime.</p><p>3840×2160 input and output on RTX 5070. Continuous resident CUDA inference must generate 48 frames/s for our 24→60 cadence. Twelve output timestamps per second coincide with source frames. This test excludes capture, upload, output readback, rendering and presentation; passing inference alone does not establish live playback.</p>
<table><tr><th>Configuration</th><th>Status</th><th>Generated FPS</th><th>p95 ms</th><th>Maximum ms</th><th>Peak tensor MiB</th></tr>ROWS</table>
<p>Scale 0.5 changes internal motion estimation; input and output remain 4K. Geometry and font glyphs are rasterized directly at 4K. Scores cover all 48 intermediate timestamps per scene. The viewer shows twelve sampled timestamps. The crop shows pixels at their native size before browser scaling.</p>
<select id="scene"></select><select id="frame"></select><div id="frames"></div>
<h2>Measured quality and settings</h2><pre>RUNS</pre>
<script>const c=CATALOG,s=document.querySelector('#scene'),f=document.querySelector('#frame');for(let k of Object.keys(c))s.add(new Option(k,k));function select(){f.innerHTML='';for(let k of Object.keys(c[s.value]))f.add(new Option('Output '+k,k));draw()}function draw(){document.querySelector('#frames').innerHTML=Object.entries(c[s.value][f.value]).map(([n,p])=>`<figure><figcaption>${n}</figcaption><img src="${p}"><canvas width="960" height="200"></canvas></figure>`).join('');for(let im of document.querySelectorAll('img')){im.onload=()=>{let ctx=im.nextElementSibling.getContext('2d');ctx.imageSmoothingEnabled=false;let scroll=s.value.startsWith('scroll')||s.value==='static';ctx.drawImage(im,scroll?680:200,scroll?1400:360,960,200,0,0,960,200)};if(im.complete&&im.naturalWidth)im.onload()}}s.onchange=select;f.onchange=draw;if(s.options.length)select();</script>'''
    compact = {}
    for name, run in runs.items():
        run = json.loads(json.dumps(run))
        run.get('resident_inference', {}).pop('samples_ms', None)
        for scene in run.get('scenes', {}).values(): scene.pop('frames', None)
        compact[name] = run
    page = page.replace('ROWS', ''.join(rows)).replace('RUNS', html.escape(json.dumps(compact, indent=2))).replace('CATALOG', json.dumps(catalog))
    (args.results/'index.html').write_text(page)
    (args.results/'summary.json').write_text(json.dumps(compact, indent=2)+'\n')


if __name__ == '__main__': main()
