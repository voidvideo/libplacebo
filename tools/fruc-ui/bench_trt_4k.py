"""Native 4K TensorRT speed test. No host transfers inside the timed interval."""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda


def cu(result):
    if result[0] != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError(f'CUDA operation failed: {result[0]}')
    return result[1] if len(result) == 2 else result[1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', type=Path)
    p.add_argument('pair', type=Path)
    p.add_argument('engine', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--build', action='store_true')
    p.add_argument('--streams', type=int, choices=[1, 2, 3], default=1)
    p.add_argument('--seconds', type=float, default=15)
    args = p.parse_args()
    if args.seconds <= 0: p.error("--seconds must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    r = {'status': 'running', 'backend': 'TensorRT', 'version': trt.__version__,
         'model': str(args.model), 'streams': args.streams, 'width': 3840, 'height': 2160,
         'fp16': True, 'tf32': False, 'cuda_graph': True, 'source_fps': 24, 'output_fps': 60,
         'required_generated_fps': 48, 'builder_optimization_level': 3,
         'requested_seconds': args.seconds,
         'scope': 'GPU-resident inputs and output; no timed host transfers; concurrent CUDA streams',
         'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
         'framework_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv'], text=True)}
    def save(): (args.output/'run.json').write_text(json.dumps(r, indent=2)+'\n')
    save()
    try:
        cu(cuda.cudaSetDevice(0))
        logger = trt.Logger(trt.Logger.WARNING)
        if args.build:
            if args.engine.exists(): raise RuntimeError('Engine already exists; use a new path')
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(args.model.read_bytes()):
                raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            assert network.num_inputs == 1
            inp = network.get_input(0)
            assert inp.shape[1] == 7, f'Expected upstream V2 seven-channel input, got {inp.shape}'
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            config.clear_flag(trt.BuilderFlag.TF32)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            config.builder_optimization_level = 3
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
            # These FP32 constraints are specified by upstream vsmlrt.RIFEMerge
            # for V2 models. Do not modify model nodes or weights.
            names = ['/Cast_2', '/Cast_3', '/Cast_5', '/Cast_7', '/Reciprocal', '/Reciprocal_1',
                     '/Mul', '/Mul_1', '/Mul_8', '/Mul_10', '/Sub_5', '/Sub_6', 'ONNXTRT_Broadcast_*']
            constrained = []
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                floating = any(layer.get_output(j).dtype in (trt.float32, trt.float16)
                               and not layer.get_output(j).is_shape_tensor
                               for j in range(layer.num_outputs))
                if floating and any(fnmatch.fnmatchcase(layer.name, name) for name in names):
                    layer.precision = trt.float32
                    constrained.append(layer.name)
            profile = builder.create_optimization_profile()
            shape = (1, 7, 2160, 3840)
            profile.set_shape(inp.name, shape, shape, shape)
            config.add_optimization_profile(profile)
            r.update(operation='build', fp32_constraints=constrained)
            save()
            start = time.perf_counter()
            serialized = builder.build_serialized_network(network, config)
            if serialized is None: raise RuntimeError('TensorRT engine build failed')
            args.engine.parent.mkdir(parents=True, exist_ok=True)
            args.engine.write_bytes(bytes(serialized))
            r.update(status='complete', build_seconds=time.perf_counter()-start,
                     engine_sha256=hashlib.sha256(args.engine.read_bytes()).hexdigest())
            return

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
        if engine is None: raise RuntimeError('Engine deserialization failed')
        r['engine_sha256'] = hashlib.sha256(args.engine.read_bytes()).hexdigest()
        inputs = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                  if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        outputs = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        assert len(inputs) == len(outputs) == 1
        input_name, output_name = inputs[0], outputs[0]
        assert engine.get_tensor_dtype(input_name) == engine.get_tensor_dtype(output_name) == trt.float32
        pair = np.fromfile(args.pair, np.float32).reshape(2, 3, 2160, 3840)
        states = []
        for t in [.4, .8, .2, .6]:
            host = np.empty((1, 7, 2160, 3840), np.float32)
            host[0, :6] = pair.reshape(6, 2160, 3840)
            host[0, 6] = t
            ptr = cu(cuda.cudaMalloc(host.nbytes))
            cu(cuda.cudaMemcpy(ptr, host.ctypes.data, host.nbytes, cuda.cudaMemcpyKind.cudaMemcpyHostToDevice))
            states.append(ptr)
        del host, pair
        workers = []
        for i in range(args.streams):
            context = engine.create_execution_context()
            if context is None: raise RuntimeError('Execution context allocation failed')
            assert context.set_input_shape(input_name, (1, 7, 2160, 3840))
            assert tuple(context.get_tensor_shape(output_name)) == (1, 3, 2160, 3840)
            stream = cu(cuda.cudaStreamCreate())
            out = cu(cuda.cudaMalloc(3*2160*3840*4))
            assert context.set_tensor_address(output_name, out)
            graphs = []
            for ptr in states:
                assert context.set_tensor_address(input_name, ptr)
                assert context.execute_async_v3(stream)
                cu(cuda.cudaStreamSynchronize(stream))
                cu(cuda.cudaStreamBeginCapture(stream, cuda.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
                assert context.execute_async_v3(stream)
                graph = cu(cuda.cudaStreamEndCapture(stream))
                executable = cu(cuda.cudaGraphInstantiate(graph, 0))
                graphs.append((graph, executable))
            workers.append((context, stream, out, graphs))
        def batch():
            for _, stream, _, graphs in workers:
                for index in range(8): cu(cuda.cudaGraphLaunch(graphs[index % 4][1], stream))
            for _, stream, _, _ in workers: cu(cuda.cudaStreamSynchronize(stream))
        for _ in range(3): batch()
        samples = []
        start = time.perf_counter()
        count = 0
        while time.perf_counter()-start < args.seconds or count < 48:
            begin = time.perf_counter()
            batch()
            samples.append((time.perf_counter()-begin)*1000)
            count += args.streams*8
        elapsed = time.perf_counter()-start
        # Execution sanity only, after timing. No image-quality evaluation.
        out = np.empty((3, 2160, 3840), np.float32)
        cu(cuda.cudaMemcpy(out.ctypes.data, workers[0][2], out.nbytes, cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        assert np.isfinite(out).all() and float(out.std()) > .01
        r.update(status='complete', operation='inference', seconds=elapsed,
                 generated_frames=count, generated_fps=count/elapsed,
                 meets_required_rate=count/elapsed >= 48, batch_size_per_stream=8,
                 batch_wall_ms=samples, finite_nonconstant_output=True,
                 output_min=float(out.min()), output_max=float(out.max()))
        print(json.dumps({k:v for k,v in r.items() if k!='batch_wall_ms'}), flush=True)
        for context, stream, ptr, graphs in workers:
            for graph, executable in graphs:
                cu(cuda.cudaGraphExecDestroy(executable)); cu(cuda.cudaGraphDestroy(graph))
            cu(cuda.cudaStreamDestroy(stream)); cu(cuda.cudaFree(ptr))
        for ptr in states: cu(cuda.cudaFree(ptr))
    except BaseException as error:
        r.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__': main()
