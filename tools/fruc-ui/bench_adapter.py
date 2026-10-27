"""Adapters invoke existing tools/models unchanged. Each run records its settings."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('backend', choices=['mvtools', 'ffmpeg', 'rife-ncnn', 'film', 'rife', 'bim', 'libplacebo'])
    p.add_argument('fixtures', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--track', choices=['midpoint', '24to60'], required=True)
    p.add_argument('--resource', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--scene', action='append')
    args = p.parse_args()
    if (args.output/'run.json').exists():
        raise ValueError('Choose a new output directory to preserve the existing run')
    manifest = json.loads((args.fixtures/'manifest.json').read_text())
    times = manifest['tracks'][args.track]
    W, H = manifest['width'], manifest['height']
    args.output.mkdir(parents=True, exist_ok=True)
    record = {'backend': args.backend, 'track': args.track, 'status': 'running',
              'settings': vars(args).copy(), 'scenes': {}}
    record['settings'] = {k: str(v) if isinstance(v, Path) else v for k,v in record['settings'].items()}
    record['artifact_sha256'] = {}
    for path in (args.resource, args.model):
        if path is None:
            continue
        files = [path] if path.is_file() else [f for f in path.rglob('*') if f.suffix in ('.pkl', '.pt', '.bin', '.param')]
        for file in files:
            with file.open('rb') as stream:
                record['artifact_sha256'][str(file)] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.is_dir() and (path/'.git').exists():
            record.setdefault('source_revisions', {})[str(path)] = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    timings = []
    start_all = time.perf_counter()
    try:
        if args.backend in ('film', 'rife', 'bim'):
            import torch
            import torch.nn.functional as F
            assert torch.cuda.is_available(), 'CUDA is required for this benchmark configuration'
            torch.set_grad_enabled(False)
            if args.backend == 'film':
                model = torch.jit.load(str(args.model), map_location='cpu').eval().cuda()
                predict = lambda a,b,t: model(a,b,a.new_full((1,1),t))
            elif args.backend == 'rife':
                import sys
                sys.path.insert(0, str(args.resource))
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(str(args.resource/'train_log'), -1)
                model.eval()
                model.device()
                predict = lambda a,b,t: model.inference(a,b,timestep=t)
            else:
                # Import the official inference component without executing
                # unrelated training/metrics package initializers.
                import sys
                import types
                for name in ('modules','utils'):
                    package=types.ModuleType(name)
                    package.__path__=[str(args.resource/name)]
                    sys.modules[name]=package
                from modules.components.bim_vfi.bim_vfi import BiMVFI
                model=BiMVFI(pyr_level=5,feat_channels=32).eval().cuda()
                checkpoint=torch.load(args.model,map_location='cpu',weights_only=False)
                model.load_state_dict(checkpoint['model'],strict=True)
                predict=lambda a,b,t: model(a,b,a.new_full((1,1,1,1),t),pyr_level=5,run_with_gt=False)['imgt_pred']
                record['pyramid_levels']=5
            record['torch'] = torch.__version__
            record['device'] = torch.cuda.get_device_name()
        if args.backend == 'mvtools':
            import vapoursynth as vs
            vs.core.std.LoadPlugin(path=str(args.resource))
            record['vapoursynth'] = str(vs.core)
        for scene in args.scene or manifest['scenes']:
            src = args.fixtures/scene
            dst = args.output/scene
            dst.mkdir(parents=True, exist_ok=True)
            begin = time.perf_counter()
            images = [np.asarray(Image.open(f).convert('RGB')) for f in sorted((src/'input').glob('*.png'))]
            commands = []
            if args.backend == 'libplacebo':
                timestamps=args.output/'timestamps.txt'
                timestamps.write_text(''.join(f'{t:.12f}\n' for t in times))
                rawpath=args.output/f'{scene}.rgba'
                command=[str(args.resource),str(src/'input.rgba'),str(rawpath),str(timestamps)]
                commands.append(command)
                with (args.output/'tool.log').open('a') as log:
                    subprocess.run(command,env=dict(os.environ,LD_PRELOAD=str(args.model)),stdout=log,stderr=log,check=True)
                frames=np.fromfile(rawpath,np.uint8).reshape(-1,H,W,4)
                for i,frame in enumerate(frames): Image.fromarray(frame[:,:,:3]).save(dst/f'{i:04}.png')
            elif args.backend == 'ffmpeg':
                rate = 48 if args.track == 'midpoint' else 60
                command = ['ffmpeg','-hide_banner','-loglevel','error','-framerate','24','-i',str(src/'input/%04d.png'),'-vf',f'minterpolate=fps={rate}:mi_mode=mci:mc_mode=aobmc:me_mode=bilat:me=umh:search_param=64:vsbmc=1:scd=none','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
                commands.append(command)
                raw = subprocess.check_output(command)
                frames = np.frombuffer(raw, np.uint8).reshape(-1,H,W,3)
                indices = range(1,len(frames),2) if args.track == 'midpoint' else range(len(frames))
                # EOF truncation is explicit. Never pad missing output.
                for i,j in enumerate(indices):
                    if i >= len(times): break
                    Image.fromarray(frames[j]).save(dst/f'{i:04}.png')
            elif args.backend == 'mvtools':
                clip = vs.core.std.BlankClip(width=W,height=H,format=vs.RGB24,length=len(images),fpsnum=24)
                def supply(n,f):
                    out=f.copy()
                    for c in range(3): np.copyto(np.asarray(out[c]), images[n][:,:,c])
                    return out
                clip=vs.core.std.ModifyFrame(clip,clip,supply)
                yuv=vs.core.resize.Point(clip,format=vs.YUV444P16,matrix_s='709',range_s='full')
                sup=vs.core.mv.Super(yuv,pel=2,hpad=64,vpad=64)
                opts=dict(blksize=16,overlap=8,search=3,searchparam=4,truemotion=True)
                fw=vs.core.mv.Analyse(sup,isb=False,**opts)
                bw=vs.core.mv.Analyse(sup,isb=True,**opts)
                rate=48 if args.track=='midpoint' else 60
                out=vs.core.mv.FlowFPS(yuv,sup,bw,fw,num=rate,den=1,mask=2)
                out=vs.core.resize.Point(out,format=vs.RGB24,matrix_in_s='709',range_in_s='full')
                record['mvtools_settings']={'analyse':opts,'pel':2,'padding':64,'mask':2,'format':'YUV444P16 full BT.709'}
                for i in range(len(times)):
                    frame=out.get_frame(2*i+1 if args.track=='midpoint' else i)
                    Image.fromarray(np.stack([np.asarray(frame[c]) for c in range(3)],axis=-1)).save(dst/f'{i:04}.png')
            else:
                for i,t in enumerate(times):
                    pos=t*24
                    j=int(math.floor(pos+1e-8)); fraction=pos-j
                    path=dst/f'{i:04}.png'
                    if abs(fraction)<1e-7:
                        Image.fromarray(images[j]).save(path)
                        continue
                    if args.backend=='rife-ncnn':
                        cmd=[str(args.resource),'-0',str(src/'input'/f'{j:04}.png'),'-1',str(src/'input'/f'{j+1:04}.png'),'-o',str(path),'-s',str(fraction),'-m',str(args.model),'-g','0','-j','1:1:1']
                        commands.append(cmd)
                        with (args.output/'tool.log').open('a') as log: subprocess.run(cmd,stdout=log,stderr=log,check=True)
                    else:
                        def tensor(image):
                            v=torch.from_numpy(image.copy()).permute(2,0,1).unsqueeze(0).float().cuda()/255
                            return F.pad(v,(0,(-W)%64,0,(-H)%64))
                        a,b=tensor(images[j]),tensor(images[j+1])
                        torch.cuda.synchronize(); begin_gpu=time.perf_counter()
                        value=predict(a,b,fraction)
                        torch.cuda.synchronize(); timings.append((time.perf_counter()-begin_gpu)*1000)
                        value=value[0,:,:H,:W].clamp(0,1).permute(1,2,0).cpu().numpy()
                        Image.fromarray(np.rint(value*255).astype(np.uint8)).save(path)
            record['scenes'][scene]={'wall_seconds':time.perf_counter()-begin,'output_count':len(list(dst.glob('*.png'))),'expected_count':len(times),'commands':commands}
        record['status']='complete' if all(s['output_count']==s['expected_count'] for s in record['scenes'].values()) else 'partial_output'
        if timings:
            record['first_inference_ms']=timings[0]
            record['subsequent_inference_ms']={'median':float(np.median(timings[1:])),'p95':float(np.percentile(timings[1:],95)),'samples':len(timings)-1}
    except BaseException as e:
        record['status']='failed';record['error']=repr(e)
        raise
    finally:
        record['total_wall_seconds']=time.perf_counter()-start_all
        (args.output/'run.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
