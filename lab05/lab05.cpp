#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

#include <immintrin.h>

static constexpr size_t N = 999968;


void fill_input(std::vector<int8_t>& a, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<int8_t>(dist(rng));
}


int64_t sum_scalar(const int8_t* a, size_t n) {
    int64_t s = 0;
    for (size_t i = 0; i < n; ++i) s += (int64_t)a[i];
    return s;
}


int64_t sum_avx2(const int8_t* a, size_t n) {
    __m256i acc32 = _mm256_setzero_si256(); // 8x int32

    for (size_t i = 0; i < n; i += 32) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));

        __m128i lo128 = _mm256_castsi256_si128(v);
        __m128i hi128 = _mm256_extracti128_si256(v, 1);

        __m256i lo16 = _mm256_cvtepi8_epi16(lo128); // 16x int16
        __m256i hi16 = _mm256_cvtepi8_epi16(hi128); // 16x int16

        __m128i lo16a = _mm256_castsi256_si128(lo16);
        __m128i lo16b = _mm256_extracti128_si256(lo16, 1);
        __m128i hi16a = _mm256_castsi256_si128(hi16);
        __m128i hi16b = _mm256_extracti128_si256(hi16, 1);

        acc32 = _mm256_add_epi32(acc32, _mm256_cvtepi16_epi32(lo16a));
        acc32 = _mm256_add_epi32(acc32, _mm256_cvtepi16_epi32(lo16b));
        acc32 = _mm256_add_epi32(acc32, _mm256_cvtepi16_epi32(hi16a));
        acc32 = _mm256_add_epi32(acc32, _mm256_cvtepi16_epi32(hi16b));
    }

    alignas(32) int32_t tmp[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc32);

    int64_t s = 0;
    for (int k = 0; k < 8; ++k) {
        s += tmp[k];
    }
    return s;
}


template<int UNROLL>
int64_t sum_avx2_unroll(const int8_t* a, size_t n) {
    constexpr size_t VEC = 32;
    constexpr size_t STEP = UNROLL * VEC;

    __m256i acc32 = _mm256_setzero_si256();

    for (size_t i = 0; i < n; i += STEP) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int8_t* p = a + i + u * VEC;
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));

            __m128i lo128 = _mm256_castsi256_si128(v);
            __m128i hi128 = _mm256_extracti128_si256(v, 1);

            __m256i lo16 = _mm256_cvtepi8_epi16(lo128);
            __m256i hi16 = _mm256_cvtepi8_epi16(hi128);

            __m128i lo16a = _mm256_castsi256_si128(lo16);
            __m128i lo16b = _mm256_extracti128_si256(lo16, 1);
            __m128i hi16a = _mm256_castsi256_si128(hi16);
            __m128i hi16b = _mm256_extracti128_si256(hi16, 1);

            acc32 = _mm256_add_epi32(acc32, _mm256_cvtepi16_epi32(lo16a));
            acc32 = _mm256_add_epi32(acc32, _mm256_cvtepi16_epi32(lo16b));
            acc32 = _mm256_add_epi32(acc32, _mm256_cvtepi16_epi32(hi16a));
            acc32 = _mm256_add_epi32(acc32, _mm256_cvtepi16_epi32(hi16b));
        }
    }

    alignas(32) int32_t tmp[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc32);

    int64_t s = 0;
    for (int k = 0; k < 8; ++k) {
        s += tmp[k];
    }
    return s;
}


template<class F>
double bench_ms(F&& fn, int iters = 50) {
    using clock = std::chrono::steady_clock;
    volatile int64_t sink = 0;

    sink ^= fn(); // warm up

    auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) sink ^= fn();
    auto t1 = clock::now();

    (void) sink;
    std::chrono::duration<double, std::milli> dt = t1 - t0;
    return dt.count() / iters;
}


int main() {
    std::vector<int8_t> a(N);
    fill_input(a, 42);

    int64_t s0 = sum_scalar(a.data(), N);
    int64_t s1 = sum_avx2(a.data(), N);
    int64_t s2 = sum_avx2_unroll<2>(a.data(), N);
    int64_t s4 = sum_avx2_unroll<4>(a.data(), N);
    int64_t s8 = sum_avx2_unroll<8>(a.data(), N);

    std::cout << "AVX2          == scalar: " << (s1 == s0 ? "YES" : "NO") << std::endl;
    std::cout << "AVX2 unroll2  == scalar: " << (s2 == s0 ? "YES" : "NO") << std::endl;
    std::cout << "AVX2 unroll4  == scalar: " << (s4 == s0 ? "YES" : "NO") << std::endl;
    std::cout << "AVX2 unroll8  == scalar: " << (s8 == s0 ? "YES" : "NO") << std::endl;

    auto t_scalar = bench_ms([&]{ return sum_scalar(a.data(), N); });
    auto t_avx2   = bench_ms([&]{ return sum_avx2(a.data(), N); });
    auto t_u2     = bench_ms([&]{ return sum_avx2_unroll<2>(a.data(), N); });
    auto t_u4     = bench_ms([&]{ return sum_avx2_unroll<4>(a.data(), N); });
    auto t_u8     = bench_ms([&]{ return sum_avx2_unroll<8>(a.data(), N); });

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Time scalar   : " << t_scalar << " ms" << std::endl;
    std::cout << "Time AVX2     : " << t_avx2   << " ms" << std::endl;
    std::cout << "Time AVX2 u=2 : " << t_u2     << " ms" << std::endl;
    std::cout << "Time AVX2 u=4 : " << t_u4     << " ms" << std::endl;
    std::cout << "Time AVX2 u=8 : " << t_u8     << " ms" << std::endl;
}
