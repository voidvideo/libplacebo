"""Native 4K menu quality and resident CUDA inference throughput for RIFE.

Uses unmodified downloaded model code. No application integration or playback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from suite import scene, SCENES


def summary(values):
    return {'mean_ms': float(np.mean(values)), 'median_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)),
            'p99_ms': float(np.percentile(values, 99)), 'max_ms': max(values),
            'count': len(values)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('resource', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--scale', type=float, choices=[1.0, 0.5], required=True)
    p.add_argument('--seconds', type=float, default=15)
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'width': 3840, 'height': 2160,
              'source_fps': 24, 'output_fps': 60, 'scale': args.scale,
              'precision': args.precision, 'backend': 'PyTorch CUDA',
              'resource': str(args.resource.resolve()), 'torch': torch.__version__,
              'scenes': {}, 'artifacts': {}}
    record['framework_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ('bench_rife_4k.py', 'suite.py')}
    for root in (args.resource/'train_log', args.resource/'model'):
        for file in sorted(root.rglob('*')):
            if file.suffix in ('.py', '.pkl'):
                with file.open('rb') as f:
                    record['artifacts'][str(file.relative_to(args.resource))] = hashlib.file_digest(f, 'sha256').hexdigest()
    record['hardware'] = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv'], text=True)
    def save():
        (args.output/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    try:
        assert torch.cuda.is_available()
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        if args.precision == 'fp16':
            # Match upstream --fp16, including warp-grid creation dtype.
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        record['cudnn_benchmark'] = torch.backends.cudnn.benchmark
        sys.path.insert(0, str(args.resource.resolve()))
        from train_log.RIFE_HDv3 import Model
        model = Model()
        # Validate every weight, rather than accepting missing keys silently.
        weights = torch.load(args.resource/'train_log/flownet.pkl', map_location='cpu', weights_only=True)
        weights = {k.removeprefix('module.'): v for k, v in weights.items()}
        extra = set(weights)-set(model.flownet.state_dict())
        assert all(k.startswith(('teacher.', 'caltime.')) for k in extra), extra
        record['unused_training_weights'] = sorted(extra)
        weights = {k: v for k, v in weights.items() if k not in extra}
        model.flownet.load_state_dict(weights, strict=True)
        model.eval()
        if args.precision == 'fp16':
            model.flownet.half()
        align = max(128, int(128 / args.scale))
        record['padding_alignment'] = align
        def tensor(im):
            v = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).unsqueeze(0).cuda().float()/255
            v = F.pad(v, (0, (-3840) % align, 0, (-2160) % align))
            return v.half() if args.precision == 'fp16' else v
        def predict(a, b, fraction):
            return model.inference(a, b, timestep=fraction, scale=args.scale)

        # Continuous resident-pair test isolates inference from PNG I/O and
        # quality scoring. It is an upper bound, not a complete playback test.
        a, b = tensor(scene(10, 'scroll_fast', 4)), tensor(scene(11, 'scroll_fast', 4))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = predict(a, b, .4)
        torch.cuda.synchronize()
        record['cold_inference_ms'] = (time.perf_counter()-start)*1000
        del result
        for _ in range(8):
            result = predict(a, b, .4)
            del result
        torch.cuda.synchronize()
        timings = []
        start = time.perf_counter()
        while time.perf_counter()-start < args.seconds or len(timings) < 48:
            fraction = (.4, .8, .2, .6)[len(timings) % 4]
            begin = time.perf_counter()
            result = predict(a, b, fraction)
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-begin)*1000)
            del result
        elapsed = time.perf_counter()-start
        record['resident_inference'] = summary(timings)
        record['resident_inference']['wall_seconds'] = elapsed
        record['resident_inference']['generated_fps'] = len(timings)/elapsed
        record['resident_inference']['equivalent_24to60_output_fps'] = len(timings)/elapsed * 60/48
        record['resident_inference']['meets_48_generated_fps'] = len(timings)/elapsed >= 48
        record['resident_inference']['samples_ms'] = timings
        record['inference_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        record['inference_peak_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
        del a, b
        save()
        print('Resident inference:', record['resident_inference']['generated_fps'], 'generated FPS', flush=True)

        # Every non-source-coincident timestamp of the one-second 24->60 track.
        # Save representative images; calculate quality for all 48 timestamps.
        keep = {1, 2, 3, 4, 26, 27, 28, 29, 56, 57, 58, 59}
        for name in SCENES:
            directory = args.output/name
            directory.mkdir()
            rows = []
            pair_index = None
            for index in range(60):
                if index % 5 == 0:
                    continue
                numerator = index*2
                j, remainder = divmod(numerator, 5)
                fraction = remainder/5
                if pair_index != j:
                    if pair_index is not None: del a, b
                    a, b = tensor(scene(j, name, 4)), tensor(scene(j+1, name, 4))
                    pair_index = j
                torch.cuda.synchronize()
                begin = time.perf_counter()
                result = predict(a, b, fraction)
                torch.cuda.synchronize()
                ms = (time.perf_counter()-begin)*1000
                pixels = (result[0, :, :2160, :3840].float().clamp(0, 1)*255).round()
                del result
                reference = scene(numerator/5, name, 4)
                ref = torch.from_numpy(np.asarray(reference).copy()).permute(2, 0, 1).cuda().float()
                error = (pixels-ref).abs().mean(0)
                edges = torch.zeros((2160, 3840), device='cuda', dtype=torch.bool)
                edges[1:] |= (ref[:, 1:]-ref[:, :-1]).abs().amax(0)>16
                edges[:, 1:] |= (ref[:, :, 1:]-ref[:, :, :-1]).abs().amax(0)>16
                mask = F.max_pool2d(edges.float()[None, None], 17, stride=1, padding=8)[0, 0].bool()
                roi = error[1460:1620] if name.startswith('scroll') or name == 'static' else error[340:1860, 280:2400]
                rows.append({'output_index': index, 'inference_ms': ms,
                             'mae_8bit': error.mean().item(),
                             'edge_mae_8bit': error[mask].mean().item(),
                             'text_region_mae_8bit': roi.mean().item(),
                             'fixed_header_mae_8bit': error[:240].mean().item()})
                if index in keep:
                    Image.fromarray(pixels.byte().permute(1, 2, 0).cpu().numpy()).save(directory/f'{index:04}.png')
                    reference.save(directory/f'{index:04}-reference.png')
                del pixels, ref, error, edges, mask, roi
            del a, b
            record['scenes'][name] = {'frames': rows,
                'mean': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'output_index'}}
            save()
            print(name, record['scenes'][name]['mean'], flush=True)
        record['status'] = 'complete'
    except BaseException as e:
        record['status'] = 'failed'
        record['error'] = repr(e)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
