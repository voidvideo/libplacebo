"""Score common timestamps and generate a lossless-frame comparison report."""
import argparse
import html
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def read(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)


def edge_mask(reference):
    gradient=np.zeros(reference.shape[:2],dtype=bool)
    gradient[1:] |= np.max(np.abs(reference[1:]-reference[:-1]),axis=2)>16
    gradient[:,1:] |= np.max(np.abs(reference[:,1:]-reference[:,:-1]),axis=2)>16
    return np.asarray(Image.fromarray(gradient).filter(ImageFilter.MaxFilter(5)))


def metrics(output, reference, scene, masks=None):
    errors=[]; edges=[]; text=[]; fixed=[]; temporal=[]
    previous=None
    if masks is None: masks=[edge_mask(b) for b in reference]
    for a,b,gradient in zip(output,reference,masks):
        delta=a-b
        error=np.abs(delta).mean(axis=2)
        errors.append(float(error.mean()))
        edges.append(float(error[gradient].mean()))
        roi=error[365:405] if scene.startswith('scroll') or scene=='static' else error[85:465,70:600]
        text.append(float(roi.mean()))
        fixed.append(float(error[:60].mean()))
        if previous is not None: temporal.append(float(np.abs(delta-previous).mean()))
        previous=delta
    return {'mae_8bit':float(np.mean(errors)), 'edge_mae_8bit':float(np.mean(edges)),
            'text_region_mae_8bit':float(np.mean(text)), 'fixed_header_mae_8bit':float(np.mean(fixed)),
            'worst_frame_mae_8bit':max(errors), 'temporal_error_change_8bit':float(np.mean(temporal))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('fixtures',type=Path);p.add_argument('results',type=Path)
    p.add_argument('--encode-video', action='store_true')
    args=p.parse_args()
    manifest=json.loads((args.fixtures/'manifest.json').read_text())
    runs={d.name:json.loads((d/'run.json').read_text()) for d in args.results.iterdir() if (d/'run.json').exists()}
    scores={};catalog={}
    for track in manifest['tracks']:
        usable=[name for name,r in runs.items() if r['track']==track and r['status'] in ('complete','partial_output')]
        scores[track]={};catalog[track]={}
        for scene in manifest['scenes']:
            names=[name for name in usable if scene in runs[name]['scenes']]
            if not names: continue
            indices=set(range(len(manifest['tracks'][track])))
            if track=='24to60': indices={i for i in indices if i%5!=0}
            for name in names:
                indices &= {int(f.stem) for f in (args.results/name/scene).glob('*.png')}
            indices=sorted(indices)
            if not indices: continue
            reference=[read(args.fixtures/scene/track/f'{i:04}.png') for i in indices]
            masks=[edge_mask(b) for b in reference]
            scores[track][scene]={'common_indices':indices,'candidates':{}}
            for name in names:
                output=[read(args.results/name/scene/f'{i:04}.png') for i in indices]
                scores[track][scene]['candidates'][name]=metrics(output,reference,scene,masks)
            # Use lossless PNGs for synchronized scrubbing; no video codec distortion.
            catalog[track][scene]={'indices':indices,'reference':[str((args.fixtures/scene/track/f'{i:04}.png').resolve()) for i in indices],
                'candidates':{name:[str((args.results/name/scene/f'{i:04}.png').resolve()) for i in indices] for name in names}}
            if args.encode_video and track=='24to60':
                videos={}
                for name,root in [('Reference',args.fixtures/scene/track), *[(n,args.results/n/scene) for n in names]]:
                    video=root/'preview.mp4'
                    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','60','-i',str(root/'%04d.png'),'-c:v','libx264','-crf','12','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
                    videos[name]=str(video.resolve())
                catalog[track][scene]['videos']=videos
    (args.results/'scores.json').write_text(json.dumps(scores,indent=2)+'\n')
    rows=[]
    for track, scenes in scores.items():
        for scene,data in scenes.items():
            for name,m in data['candidates'].items():
                rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [track,scene,name,len(data['common_indices']),round(m['edge_mae_8bit'],3),round(m['text_region_mae_8bit'],3),round(m['fixed_header_mae_8bit'],3),round(m['temporal_error_change_8bit'],3)])+'</tr>')
    page='''<!doctype html><meta charset="utf-8"><title>Menu interpolation comparison</title>
<style>body{background:#151922;color:#eceff5;font:16px system-ui;margin:24px}select,button,input{font:inherit;margin:8px}#frames{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0;width:46%;min-width:400px}img,canvas{width:100%;image-rendering:auto}figcaption{padding:6px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #465064;text-align:left}p{max-width:1000px}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#8ec8ff}</style>
<h1>Menu interpolation comparison</h1><p>960×540 synthetic menus with known reference images. Inspect the pictures before interpreting the scores. Lower error is better; these values are RGB code values, not percentages of visual quality. This is an offline quality benchmark, not application playback validation.</p>
<p>Every candidate is scored at the same available intermediate timestamps within each scene and track. Missing output is not padded. The reference follows the defined synthetic animation; real Android TV footage remains untested.</p>
<select id="track"></select><select id="scene"></select><input id="frame" type="range" min="0" value="0"><button id="play">Play / pause</button><span id="time"></span><p>The crop enlarges the left text region. Preview videos preserve each tool's actual output length and are compressed for viewing; scores use lossless PNGs.</p><div id="frames"></div>
<h2>Scores</h2><p>Text-region error includes the surrounding menu region. Edge error uses reference edges dilated by two pixels. Temporal error measures changes in reconstruction error between consecutive scored frames; the 24→60 track excludes source-coincident frames and therefore has unequal time intervals. Read it as a diagnostic, not a perceptual ranking.</p><table><tr><th>Track</th><th>Scene</th><th>Candidate</th><th>Frames</th><th>Edge error</th><th>Text region</th><th>Fixed header</th><th>Temporal error change</th></tr>ROWS</table>
<h2>Run status and settings</h2><pre>RUNS</pre>
<script>const catalog=CATALOG;const t=document.querySelector('#track'),s=document.querySelector('#scene'),f=document.querySelector('#frame');let playing=false;for(const k of Object.keys(catalog))t.add(new Option(k,k));function choose(){s.innerHTML='';for(const k of Object.keys(catalog[t.value]))s.add(new Option(k,k));draw()}function draw(){let d=catalog[t.value][s.value];if(!d)return;f.max=d.indices.length-1;let i=Math.min(+f.value,+f.max);document.querySelector('#time').textContent='Output index '+d.indices[i];document.querySelector('#frames').innerHTML=Object.entries({'Reference':d.reference,...d.candidates}).map(([name,paths])=>`<figure><figcaption>${name} ${d.videos?.[name]?`<a href="${d.videos[name]}">60 FPS preview</a>`:''}</figcaption><img src="${paths[i]}"><canvas width="960" height="140"></canvas></figure>`).join('');for(const img of document.querySelectorAll('figure img')){img.onload=()=>{let ctx=img.nextElementSibling.getContext('2d');ctx.imageSmoothingEnabled=false;const scroll=s.value.startsWith('scroll')||s.value==='static';ctx.drawImage(img,scroll?0:50,scroll?350:90,480,70,0,0,960,140)};if(img.complete&&img.naturalWidth)img.onload()}}t.onchange=choose;s.onchange=()=>{f.value=0;draw()};f.oninput=draw;document.querySelector('#play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){f.value=(+f.value+1)%(+f.max+1);draw()}},100);choose();</script>'''
    page=page.replace('ROWS',''.join(rows)).replace('RUNS',html.escape(json.dumps(runs,indent=2))).replace('CATALOG',json.dumps(catalog))
    (args.results/'index.html').write_text(page)


if __name__=='__main__':main()
