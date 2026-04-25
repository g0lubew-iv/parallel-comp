#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <vector>
#include <cstdint>

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

extern "C" Results run_memcpy_bench(size_t bytes, int iters, int do_warmup);

static double gbps(size_t bytes, double ms) {
    return (double)(bytes / 1e9) / (ms / 1000.0);
}

static void cpu_memcpy_bench(size_t bytes, int iters, bool pinned_like, double& out_ms, bool& ok) {
    std::vector<uint8_t> a(bytes), b(bytes);
    for (size_t i = 0; i < bytes; i++) {
        a[i] = (uint8_t)((i * 17u + 3u) & 0xFF);
    }

    (void) pinned_like;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        std::memcpy(b.data(), a.data(), bytes);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dt = t1 - t0;
    out_ms = dt.count() / iters;
    ok = (std::memcmp(a.data(), b.data(), bytes) == 0);
}

int main(int argc, char** argv) {
    size_t bytes = 256ull * 1024 * 1024; // 256 MiB default

    int iters  = 20;
    int warmup = 1;

    if (argc >= 2) {
        bytes = (size_t)std::stoull(argv[1]); // bytes
    }
    if (argc >= 3) {
        iters = std::stoi(argv[2]); // iterations
    }

    std::cout << "Buffer size: " << bytes << " bytes (" << std::fixed << std::setprecision(2) << (bytes / (1024.0*1024.0)) << " MiB)" << std::endl;
    std::cout << "Iterations: "  << iters << std::endl;

    // CPU -> CPU
    double cpu_ms = 0.0;
    bool cpu_ok = false;
    cpu_memcpy_bench(bytes, iters, false, cpu_ms, cpu_ok);

    std::cout << "[CPU->CPU memcpy] avg: " << cpu_ms << " ms, " << gbps(bytes, cpu_ms) << " GB/s, ok=" << cpu_ok << std::endl;

    // GPU-related copies
    Results r = run_memcpy_bench(bytes, iters, warmup);

    auto pr = [&](const char* name, float ms) {
        std::cout << std::left << std::setw(28) << name
                  << " avg: " << std::right << std::setw(8) << std::fixed << std::setprecision(3)
                  << ms << " ms,  "
                  << std::setw(8) << std::setprecision(2) << gbps(bytes, ms)
                  << " GB/s" << std::endl;
    };

    pr("H->D pageable", r.h2d_pageable_ms);
    pr("D->H pageable", r.d2h_pageable_ms);
    std::cout << "  roundtrip ok=" << r.ok_h2d_d2h_pageable << std::endl;

    pr("H->D pinned", r.h2d_pinned_ms);
    pr("D->H pinned", r.d2h_pinned_ms);
    std::cout << "  roundtrip ok=" << r.ok_h2d_d2h_pinned << std::endl;

    pr("D->D", r.d2d_ms);
    std::cout << "  d2d ok=" << r.ok_d2d << std::endl;

    return 0;
}
