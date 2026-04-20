#include <immintrin.h>
#include <cpuid.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdlib>

constexpr int W = 800;
constexpr int H = 800;
constexpr int N = W * H;

static uint8_t src[N];


static void blur2x2_scalar(const uint8_t* s, uint8_t* d, int w, int h) {
    std::fill(d, d + w * h, 0);
    for (int y = 0; y < h - 1; ++y) {
        const uint8_t* r0 = s + y * w;
        const uint8_t* r1 = s + (y + 1) * w;
        uint8_t* out = d + y * w;

        for (int x = 0; x < w - 1; ++x) {
            int sum = r0[x] + r0[x + 1] + r1[x] + r1[x + 1];
            out[x] = (uint8_t)(sum / 4);
        }
    }
}


static void blur2x2_avx2(const uint8_t* s, uint8_t* d, int w, int h) {
    std::fill(d, d + w * h, 0);

    for (int y = 0; y < h - 1; ++y) {
        const uint8_t* r0 = s + y * w;
        const uint8_t* r1 = s + (y + 1) * w;
        uint8_t* out = d + y * w;

        int x = 0;
        for (; x + 15 < w - 1; x += 16) {
            __m128i a0 = _mm_loadu_si128((const __m128i*)(r0 + x));
            __m128i b0 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
            __m128i a1 = _mm_loadu_si128((const __m128i*)(r1 + x));
            __m128i b1 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));

            __m256i a0_16 = _mm256_cvtepu8_epi16(a0);
            __m256i b0_16 = _mm256_cvtepu8_epi16(b0);
            __m256i a1_16 = _mm256_cvtepu8_epi16(a1);
            __m256i b1_16 = _mm256_cvtepu8_epi16(b1);

            __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(a0_16, b0_16),
                                             _mm256_add_epi16(a1_16, b1_16));

            __m256i avg16 = _mm256_srli_epi16(sum16, 2);

            __m128i lo = _mm256_castsi256_si128(avg16);
            __m128i hi = _mm256_extracti128_si256(avg16, 1);

            __m128i out16 = _mm_packus_epi16(lo, hi);

            _mm_storeu_si128((__m128i*)(out + x), out16);
        }

        for (; x < w - 1; ++x) {
            int sum = r0[x] + r0[x + 1] + r1[x] + r1[x + 1];
            out[x] = (uint8_t)(sum / 4);
        }
    }
}


static int max_abs_diff(const uint8_t* a, const uint8_t* b, int n) {
    int md = 0;
    for (int i = 0; i < n; ++i) {
        int d = std::abs((int)a[i] - (int)b[i]);
        md = std::max(md, d);
    }
    return md;
}


template <class F>
static double bench_ms(F&& f, int iters) {
    using clock = std::chrono::steady_clock;

    for (int i = 0; i < 5; ++i) {
        f(); // warm up
    }
    auto t0 = clock::now();

    for (int i = 0; i < iters; ++i) {
        f();
    }
    auto t1 = clock::now();

    std::chrono::duration<double, std::milli> ms = t1 - t0;
    return ms.count() / iters;
}


static void sink_checksum(const uint8_t* data, int n) {
    uint64_t s = 0;
    for (int i = 0; i < n; ++i) {
        s += data[i];
    }
    std::cout << "checksum: " << s << std::endl;
}


int main() {
    std::cout << "AVX2 supported: " << (__builtin_cpu_supports("avx2") ? "yes" : "no") << std::endl;

    std::ifstream f("img.raw", std::ios::binary);
    f.read(reinterpret_cast<char*>(src), N);

    std::vector<uint8_t> out_scalar(N), out_avx2(N);

    blur2x2_scalar(src, out_scalar.data(), W, H);
    blur2x2_avx2(src, out_avx2.data(), W, H);

    int md = max_abs_diff(out_scalar.data(), out_avx2.data(), N);
    std::cout << "max_abs_diff(scalar, avx2) = " << md << std::endl;
    if (md != 0) {
        return 2;
    }

    constexpr int iters = 300;

    double t_scalar = bench_ms([&] { blur2x2_scalar(src, out_scalar.data(), W, H); }, iters);
    double t_avx2   = bench_ms([&] { blur2x2_avx2(src, out_avx2.data(), W, H); }, iters);

    std::cout << "avg scalar: " << t_scalar << " ms" << std::endl;
    std::cout << "avg avx2  : " << t_avx2   << " ms" << std::endl;
    std::cout << "speedup   : " << (t_scalar / t_avx2) << "x" << std::endl;

    sink_checksum(out_avx2.data(), N);
    return 0;
}
