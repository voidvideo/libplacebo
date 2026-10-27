"""Persistent native 4K ncnn/Vulkan benchmark using the unmodified VS plugin."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import vapoursynth as vs

from suite import scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument('plugin', type=Path)
    p.add_argument('model', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--workers', type=int, choices=[1, 2, 3], required=True)
    p.add_argument('--mode', choices=['timing'], default='timing')
    p.add_argument('--seconds', type=float, default=15)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'backend': 'ncnn/Vulkan VapourSynth plugin',
              'mode': args.mode, 'pid': os.getpid(), 'width': 3840, 'height': 2160,
              'source_fps': 24, 'output_fps': 60, 'workers': args.workers,
              'plugin': str(args.plugin.resolve()), 'model': str(args.model.resolve()),
              'sc': False, 'skip': False, 'tta': False, 'uhd': False, 'scenes': {},
              'precision': 'Upstream fp16 storage/packing enabled; fp16 arithmetic disabled',
              'vapoursynth': str(vs.core), 'artifacts': {}}
    for file in [args.plugin, *args.model.glob('flownet.*'), Path(__file__), Path(__file__).with_name('suite.py')]:
        with file.open('rb') as f: record['artifacts'][str(file.resolve())] = hashlib.file_digest(f, 'sha256').hexdigest()
    record['hardware'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv'], text=True)
    def save():
        (args.output/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    memory = []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                values = result.stdout.strip().split(',')
                memory.append({'time': time.monotonic(), 'board_memory_mib': int(values[0]), 'gpu_utilization_percent': int(values[1])})
            stop.wait(.5)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        core = vs.core
        core.num_threads = 8
        core.max_cache_size = 1024
        core.std.LoadPlugin(path=str(args.plugin.resolve()))
        template = core.std.BlankClip(width=3840, height=2160, format=vs.RGBS, length=1, fpsnum=24)
        def make_frame(im):
            data = np.asarray(im).transpose(2, 0, 1).astype(np.float32)/255
            out = template.get_frame(0).copy()
            for c in range(3): np.copyto(np.asarray(out[c]), data[c])
            return out
        def make_clip(frames, length):
            base = core.std.BlankClip(template, length=length)
            source = core.std.ModifyFrame(base, base, lambda n, f: frames[n % len(frames)])
            core.std.SetVideoCache(source, mode=0)
            output = core.rife.RIFE(source, model_path=str(args.model.resolve()),
                fps_num=60, fps_den=1, gpu_id=0, gpu_thread=args.workers,
                sc=False, skip=False, tta=False, uhd=False)
            core.std.SetVideoCache(output, mode=0)
            assert output.width == 3840 and output.height == 2160
            assert output.fps_num == 60 and output.fps_den == 1
            return output
        def rgb(frame):
            data = np.stack([np.asarray(frame[c]) for c in range(3)], axis=2)
            assert np.isfinite(data).all(), 'Non-finite Vulkan output'
            return np.rint(np.clip(data, 0, 1)*255).astype(np.uint8)

        frames = [make_frame(scene(i, 'scroll_fast', 4)) for i in (10, 11)]
        output = make_clip(frames, 20000)
        begin = time.perf_counter()
        check = rgb(output.get_frame(1))
        record['cold_request_ms'] = (time.perf_counter()-begin)*1000
        assert not np.array_equal(check, np.asarray(scene(10, 'scroll_fast', 4))), 'Output repeats first source'
        for i in [2, 3, 4, 6, 7, 8, 9, 11]: output.get_frame(i)
        # Unique output indices and disabled VS caches prevent a cached
        # frame from being counted as another interpolation.
        queue = deque()
        latency = []
        completion_times = []
        indices = []
        index = 101
        depth = args.workers * 2
        start = time.perf_counter()
        def submit():
            nonlocal index
            while index % 5 == 0: index += 1
            submitted = time.perf_counter()
            future = output.get_frame_async(index)
            event = threading.Event()
            def completed(f):
                end = time.perf_counter()
                latency.append((end-submitted)*1000)
                completion_times.append(end-start)
                event.set()
            future.add_done_callback(completed)
            queue.append((future, event))
            indices.append(index)
            index += 1
        for _ in range(depth): submit()
        count = 0
        while queue:
            future, event = queue.popleft()
            frame = future.result()
            event.wait()
            count += 1
            del frame, future
            if time.perf_counter()-start < args.seconds or count < 48:
                submit()
        elapsed = time.perf_counter()-start
        assert len(set(indices)) == count and len(latency) == count
        record['timing'] = {'seconds': elapsed, 'generated_frames': count,
            'generated_fps': count/elapsed, 'equivalent_24to60_output_fps': count/elapsed*60/48,
            'meets_48_generated_fps': count/elapsed >= 48, 'in_flight_requests': depth,
            'request_latency_median_ms': float(np.median(latency)),
            'request_latency_p95_ms': float(np.percentile(latency, 95)),
            'request_latency_max_ms': max(latency), 'request_latencies_ms': latency,
            'completion_seconds': completion_times, 'output_indices': indices,
            'scope': 'Persistent Vulkan filter including CPU packing, GPU transfers, inference and readback; excludes image I/O, capture and presentation'}
        print(json.dumps({k: v for k, v in record['timing'].items() if not isinstance(v, list)}), flush=True)
        record['status'] = 'complete'
    except BaseException as e:
        record['status'] = 'failed'
        record['error'] = repr(e)
        raise
    finally:
        stop.set()
        monitor_thread.join()
        record['board_memory_samples'] = memory
        if memory: record['peak_board_memory_mib'] = max(r['board_memory_mib'] for r in memory)
        save()


if __name__ == '__main__': main()
