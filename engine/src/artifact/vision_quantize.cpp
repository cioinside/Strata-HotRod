// src/artifact/vision_quantize.cpp - the `strata-vision-quantize` CLI.
// Reads a GGUF, quantizes every `clip.vision.*` tensor F32 -> Q8_0 with the engine's faithful
// `quantize_q8_0` (the same kernel that writes ggml's Q8_0 bytes), and dumps each tensor's
// quantized bytes + its shape to an output directory. A GGUF writer (Python) reassembles them.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/quantize_act.hpp"

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: strata-vision-quantize <in.gguf> <out_dir>\n"); return 2; }
    strata::GgufFile g(argv[1]);
    std::string out = argv[2];
    int n = 0;
    for (const auto& t : g.tensors()) {
        if (t.name.rfind("clip.vision.", 0) != 0) continue;
        int64_t ne = 1;
        for (auto d : t.shape) ne *= d;
        if (ne % 32) { std::printf("skip %s (%lld not mult of 32)\n", t.name.c_str(), (long long)ne); continue; }
        const float* src = (const float*) g.tensor_data(t);
        float* d_src = nullptr;
        if (cudaMalloc(&d_src, ne * 4) != cudaSuccess) { std::printf("cudaMalloc %s failed\n", t.name.c_str()); return 1; }
        cudaMemcpy(d_src, src, ne * 4, cudaMemcpyHostToDevice);
        int64_t nb = ne / 32;
        uint8_t* d_blk = nullptr;
        if (cudaMalloc(&d_blk, nb * 34) != cudaSuccess) { cudaFree(d_src); std::printf("cudaMalloc blk %s failed\n", t.name.c_str()); return 1; }
        strata::kernels::quantize_q8_0(d_src, d_blk, ne, nullptr);
        std::vector<uint8_t> ob((size_t) nb * 34);
        cudaMemcpy(ob.data(), d_blk, ob.size(), cudaMemcpyDeviceToHost);
        cudaFree(d_src); cudaFree(d_blk);
        std::string path = out + "/" + t.name + ".q8_0";
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) { std::printf("cannot write %s\n", path.c_str()); return 1; }
        fwrite(ob.data(), 1, ob.size(), f);
        fclose(f);
        std::printf("%s [%lld,%lld] -> %lld bytes Q8_0\n", t.name.c_str(),
                    (long long) t.shape[0], (long long) t.shape[1], (long long) ob.size());
        n++;
    }
    std::printf("quantized %d vision tensors\n", n);
    return 0;
}
