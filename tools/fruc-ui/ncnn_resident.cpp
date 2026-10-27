// Benchmark glue only: use the released graph and upstream Warp unchanged.
#include <net.h>
#include <gpu.h>
#include <command.h>
#include "rife_ops.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <future>

DEFINE_LAYER_CREATOR(Warp)
using Clock = std::chrono::steady_clock;
static void check(int code) { if (code) throw std::runtime_error("ncnn operation failed: " + std::to_string(code)); }

int main(int argc, char** argv) {
    // model directory, planar RGB float32 pair, output float32 file, timings JSON
    if (argc != 5 && argc != 6) return 2;
    const int workers = argc == 6 ? std::stoi(argv[5]) : 1;
    if (workers < 1 || workers > 3) return 2;
    ncnn::create_gpu_instance();
    {
        auto* gpu = ncnn::get_gpu_device(0);
        ncnn::Net net;
        net.opt.num_threads = 1;
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_int8_storage = false;
        net.set_vulkan_device(gpu);
        net.register_custom_layer("rife.Warp", Warp_layer_creator);
        check(net.load_param((std::string(argv[1])+"/flownet.param").c_str()));
        check(net.load_model((std::string(argv[1])+"/flownet.bin").c_str()));
        int padding = std::string(argv[1]).find("4.25-lite") != std::string::npos ? 128 :
                      std::string(argv[1]).find("4.25") != std::string::npos ? 64 : 32;
        const int w = 3840, h = 2160, pw = (w+padding-1)/padding*padding, ph = (h+padding-1)/padding*padding;
        auto* blob = gpu->acquire_blob_allocator();
        auto* staging = gpu->acquire_staging_allocator();
        auto opt = net.opt;
        opt.blob_vkallocator = blob;
        opt.workspace_vkallocator = blob;
        opt.staging_vkallocator = staging;
        {
            ncnn::VkMat inputs[2], times[4];
            FILE* in = fopen(argv[2], "rb");
            if (!in) throw std::runtime_error("Cannot read input pair");
            for (int i=0; i<2; i++) {
                ncnn::Mat cpu(pw, ph, 3);
                cpu.fill(0.f);
                for (int c=0; c<3; c++) {
                    float* channel = cpu.channel(c);
                    for (int y=0; y<h; y++)
                        if (fread(channel+y*pw, sizeof(float), w, in) != size_t(w)) throw std::runtime_error("Short input");
                }
                ncnn::Mat half;
                ncnn::cast_float32_to_float16(cpu, half, opt);
                ncnn::VkCompute cmd(gpu);
                cmd.record_clone(half, inputs[i], opt);
                check(cmd.submit_and_wait());
            }
            fclose(in);
            const float fractions[4] = {.4f, .8f, .2f, .6f};
            for (int i=0; i<4; i++) {
                ncnn::Mat cpu(pw, ph, 1);
                cpu.fill(fractions[i]);
                ncnn::Mat half;
                ncnn::cast_float32_to_float16(cpu, half, opt);
                ncnn::VkCompute cmd(gpu);
                cmd.record_clone(half, times[i], opt);
                check(cmd.submit_and_wait());
            }
            auto infer = [&](int t, bool download, ncnn::VkAllocator* worker_blob, ncnn::VkAllocator* worker_staging) {
                ncnn::VkCompute cmd(gpu);
                auto ex = net.create_extractor();
                ex.set_blob_vkallocator(worker_blob);
                ex.set_workspace_vkallocator(worker_blob);
                ex.set_staging_vkallocator(worker_staging);
                check(ex.input("in0", inputs[0]));
                check(ex.input("in1", inputs[1]));
                check(ex.input("in2", times[t]));
                ncnn::VkMat out;
                check(ex.extract("out0", out, cmd));
                ncnn::Mat cpu;
                if (download) cmd.record_clone(out, cpu, opt);
                check(cmd.submit_and_wait());
                if (download) {
                    ncnn::Mat unpacked, fp32;
                    ncnn::convert_packing(cpu, unpacked, 1, opt);
                    if (unpacked.elemsize == 2) ncnn::cast_float16_to_float32(unpacked, fp32, opt);
                    else fp32 = unpacked;
                    if (fp32.w != pw || fp32.h != ph || fp32.c != 3) throw std::runtime_error("Wrong output shape");
                    FILE* f = fopen(argv[3], "wb");
                    if (!f) throw std::runtime_error("Cannot write output");
                    for (int c=0; c<3; c++) {
                        const float* channel = fp32.channel(c);
                        for (int y=0; y<h; y++)
                            if (fwrite(channel+y*pw, sizeof(float), w, f) != size_t(w)) throw std::runtime_error("Short output");
                    }
                    fclose(f);
                }
            };
            std::promise<void> release;
            auto gate = release.get_future().share();
            std::vector<std::promise<void>> ready(workers);
            std::vector<std::future<std::vector<double>>> tasks;
            Clock::time_point start;
            for (int worker=0; worker<workers; worker++) {
                tasks.push_back(std::async(std::launch::async, [&, worker] {
                    auto* worker_blob = gpu->acquire_blob_allocator();
                    auto* worker_staging = gpu->acquire_staging_allocator();
                    try {
                        for (int i=0; i<9; i++) infer(i%4, false, worker_blob, worker_staging);
                    } catch (...) {
                        ready[worker].set_exception(std::current_exception());
                        throw;
                    }
                    ready[worker].set_value();
                    gate.get();
                    std::vector<double> samples;
                    do {
                        auto begin = Clock::now();
                        infer(samples.size()%4, false, worker_blob, worker_staging);
                        samples.push_back(std::chrono::duration<double, std::milli>(Clock::now()-begin).count());
                    } while (Clock::now()-start < std::chrono::seconds(15) || samples.size()<48);
                    gpu->reclaim_blob_allocator(worker_blob);
                    gpu->reclaim_staging_allocator(worker_staging);
                    return samples;
                }));
            }
            try {
                for (auto& promise : ready) promise.get_future().get();
            } catch (...) {
                release.set_exception(std::current_exception());
                for (auto& task : tasks) { try { task.get(); } catch (...) {} }
                throw;
            }
            start = Clock::now();
            release.set_value();
            std::vector<double> ms;
            for (auto& task : tasks) {
                auto samples = task.get();
                ms.insert(ms.end(), samples.begin(), samples.end());
            }
            double seconds = std::chrono::duration<double>(Clock::now()-start).count();
            infer(0, true, blob, staging);
            FILE* f = fopen(argv[4], "w");
            if (!f) throw std::runtime_error("Cannot write timing record");
            fprintf(f, "{\"status\":\"complete\",\"backend\":\"ncnn/Vulkan GPU-resident graph\",\"width\":%d,\"height\":%d,\"padding\":%d,\"workers\":%d,\"cooperative_matrix_option\":%s,\"seconds\":%.9f,\"generated_fps\":%.9f,\"milliseconds\":[", w,h,padding,workers,opt.use_cooperative_matrix?"true":"false",seconds,ms.size()/seconds);
            for (size_t i=0; i<ms.size(); i++) fprintf(f, "%s%.9f", i?",":"",ms[i]);
            fputs("]}\n",f);
            fclose(f);
        }
        gpu->reclaim_blob_allocator(blob);
        gpu->reclaim_staging_allocator(staging);
    }
    ncnn::destroy_gpu_instance();
    return 0;
}
