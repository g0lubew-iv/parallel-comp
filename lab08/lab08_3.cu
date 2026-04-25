#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

static void fill_pattern(uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)((i * 1315423911u + 12345u) & 0xFF);
}

static bool equal_buf(const uint8_t* a, const uint8_t* b, size_t n) {
    return std::memcmp(a, b, n) == 0;
}

static float gbps(size_t bytes, float ms) {
    return (float)(bytes / 1e9) / (ms / 1000.0f);
}

struct Results {
    float h2d_pageable_ms;
    float d2h_pageable_ms;
    float h2d_pinned_ms;
    float d2h_pinned_ms;
    float d2d_ms;

    int ok_h2d_d2h_pageable;
    int ok_h2d_d2h_pinned;
    int ok_d2d;
};

extern "C" Results run_memcpy_bench(size_t bytes, int iters, int do_warmup) {
    Results r{};
    r.ok_h2d_d2h_pageable = 0;
    r.ok_h2d_d2h_pinned   = 0;
    r.ok_d2d              = 0;

    uint8_t* h_src = (uint8_t*)std::malloc(bytes);
    uint8_t* h_dst = (uint8_t*)std::malloc(bytes);
    uint8_t* h_chk = (uint8_t*)std::malloc(bytes);
    if (!h_src || !h_dst || !h_chk) {
        std::printf("malloc failed\n");
        std::exit(1);
    }

    uint8_t *hp_src = nullptr, *hp_dst = nullptr, *hp_chk = nullptr;
    checkCuda(cudaMallocHost((void**)&hp_src, bytes));
    checkCuda(cudaMallocHost((void**)&hp_dst, bytes));
    checkCuda(cudaMallocHost((void**)&hp_chk, bytes));

    uint8_t *d_a = nullptr, *d_b = nullptr;
    checkCuda(cudaMalloc((void**)&d_a, bytes));
    checkCuda(cudaMalloc((void**)&d_b, bytes));

    fill_pattern(h_src, bytes);
    std::memset(h_dst, 0, bytes);
    std::memset(h_chk, 0, bytes);

    fill_pattern(hp_src, bytes);
    std::memset(hp_dst, 0, bytes);
    std::memset(hp_chk, 0, bytes);

    // events
    cudaEvent_t e0, e1;
    checkCuda(cudaEventCreate(&e0));
    checkCuda(cudaEventCreate(&e1));

    auto timeMemcpy = [&](cudaMemcpyKind kind, const void* src, void* dst, float* out_ms) {
        // warmup
        if (do_warmup) {
            checkCuda(cudaMemcpy(dst, src, bytes, kind));
            checkCuda(cudaDeviceSynchronize());
        }
        checkCuda(cudaEventRecord(e0));
        for (int i = 0; i < iters; i++) {
            checkCuda(cudaMemcpy(dst, src, bytes, kind));
        }
        checkCuda(cudaEventRecord(e1));
        checkCuda(cudaEventSynchronize(e1));

        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, e0, e1));
        *out_ms = ms / iters;
    };

    timeMemcpy(cudaMemcpyHostToDevice, h_src, d_a, &r.h2d_pageable_ms);
    timeMemcpy(cudaMemcpyDeviceToHost, d_a, h_dst, &r.d2h_pageable_ms);

    r.ok_h2d_d2h_pageable = equal_buf(h_src, h_dst, bytes) ? 1 : 0;

    timeMemcpy(cudaMemcpyHostToDevice, hp_src, d_a, &r.h2d_pinned_ms);
    timeMemcpy(cudaMemcpyDeviceToHost, d_a, hp_dst, &r.d2h_pinned_ms);

    r.ok_h2d_d2h_pinned = equal_buf(hp_src, hp_dst, bytes) ? 1 : 0;

    if (do_warmup) {
        checkCuda(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));
        checkCuda(cudaDeviceSynchronize());
    }
    checkCuda(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) {
        checkCuda(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));
    }
    checkCuda(cudaEventRecord(e1));
    checkCuda(cudaEventSynchronize(e1));
    float d2d_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&d2d_ms, e0, e1));
    r.d2d_ms = (d2d_ms / iters);

    checkCuda(cudaMemcpy(h_chk, d_b, bytes, cudaMemcpyDeviceToHost));
    r.ok_d2d = equal_buf(h_chk, h_src, bytes) ? 1 : 0;

    checkCuda(cudaEventDestroy(e0));
    checkCuda(cudaEventDestroy(e1));

    cudaFree(d_a);
    cudaFree(d_b);

    cudaFreeHost(hp_src);
    cudaFreeHost(hp_dst);
    cudaFreeHost(hp_chk);

    std::free(h_src);
    std::free(h_dst);
    std::free(h_chk);

    return r;
}
