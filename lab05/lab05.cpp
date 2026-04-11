#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstring>

#include <mmintrin.h> // __m64, _mm_*

static constexpr size_t N = 1'000'000;


void fill_input(std::vector<int8_t>& a, std::vector<int8_t>& b, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<int8_t>(dist(rng));
        b[i] = static_cast<int8_t>(dist(rng));
    }
}


void mul_scalar(const int8_t* a, const int8_t* b, int16_t* c, size_t n) {
    for (size_t i = 0; i < n; ++i)
        c[i] = static_cast<int16_t>(a[i]) * static_cast<int16_t>(b[i]);
}


static inline void mmx_mul8_store16(const int8_t* a, const int8_t* b, int16_t* c) {
    __m64 va = *reinterpret_cast<const __m64*>(a);
    __m64 vb = *reinterpret_cast<const __m64*>(b);

    const __m64 zero = _mm_setzero_si64();

    __m64 sa = _mm_cmpgt_pi8(zero, va);
    __m64 sb = _mm_cmpgt_pi8(zero, vb);

    __m64 a_lo = _mm_unpacklo_pi8(va, sa);
    __m64 a_hi = _mm_unpackhi_pi8(va, sa);
    __m64 b_lo = _mm_unpacklo_pi8(vb, sb);
    __m64 b_hi = _mm_unpackhi_pi8(vb, sb);

    __m64 p_lo = _mm_mullo_pi16(a_lo, b_lo);
    __m64 p_hi = _mm_mullo_pi16(a_hi, b_hi);

    *reinterpret_cast<__m64*>(c + 0) = p_lo;
    *reinterpret_cast<__m64*>(c + 4) = p_hi;
}


void mul_mmx(const int8_t* a, const int8_t* b, int16_t* c, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        mmx_mul8_store16(a + i, b + i, c + i);
    }

    for (; i < n; ++i) {
        c[i] = static_cast<int16_t>(a[i]) * static_cast<int16_t>(b[i]);
    }

    _mm_empty();
}


template<int UNROLL>
void mul_mmx_unroll(const int8_t* a, const int8_t* b, int16_t* c, size_t n) {
    constexpr size_t VEC = 8;
    constexpr size_t STEP = UNROLL * VEC;

    size_t i = 0;
    for (; i + STEP <= n; i += STEP) {
        #pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            mmx_mul8_store16(a + i + k * VEC, b + i + k * VEC, c + i + k * VEC);
        }
    }

    for (; i < n; ++i)
        c[i] = static_cast<int16_t>(a[i]) * static_cast<int16_t>(b[i]);

    _mm_empty();
}


bool equal_arrays(const std::vector<int16_t>& x, const std::vector<int16_t>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(int16_t)) == 0;
}


template<class F>
double bench_ms(F&& fn, int iters = 30) {
    using clock = std::chrono::steady_clock;
    fn();

    auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    auto t1 = clock::now();

    std::chrono::duration<double, std::milli> dt = t1 - t0;
    return dt.count() / iters;
}


int main() {
    std::vector<int8_t>  a(N), b(N);
    std::vector<int16_t> c0(N), c1(N), c2(N), c4(N), c8(N);

    fill_input(a, b, 42); // seed = 42

    mul_scalar(a.data(), b.data(), c0.data(), N);
    mul_mmx(a.data(), b.data(), c1.data(), N);
    mul_mmx_unroll<2>(a.data(), b.data(), c2.data(), N);
    mul_mmx_unroll<4>(a.data(), b.data(), c4.data(), N);
    mul_mmx_unroll<8>(a.data(), b.data(), c8.data(), N);

    std::cout << "MMX         = scalar: " << (equal_arrays(c1, c0) ? "YES" : "NO") << std::endl;
    std::cout << "MMX unroll2 = scalar: " << (equal_arrays(c2, c0) ? "YES" : "NO") << std::endl;
    std::cout << "MMX unroll4 = scalar: " << (equal_arrays(c4, c0) ? "YES" : "NO") << std::endl;
    std::cout << "MMX unroll8 = scalar: " << (equal_arrays(c8, c0) ? "YES" : "NO") << std::endl;

    auto t_scalar = bench_ms([&]{ mul_scalar(a.data(), b.data(), c0.data(), N); });
    auto t_mmx    = bench_ms([&]{ mul_mmx(a.data(), b.data(), c1.data(), N); });
    auto t_u2     = bench_ms([&]{ mul_mmx_unroll<2>(a.data(), b.data(), c2.data(), N); });
    auto t_u4     = bench_ms([&]{ mul_mmx_unroll<4>(a.data(), b.data(), c4.data(), N); });
    auto t_u8     = bench_ms([&]{ mul_mmx_unroll<8>(a.data(), b.data(), c8.data(), N); });

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Time scalar   : " << t_scalar << " ms" << std::endl;
    std::cout << "Time MMX      : " << t_mmx    << " ms" << std::endl;
    std::cout << "Time MMX u=2  : " << t_u2     << " ms" << std::endl;
    std::cout << "Time MMX u=4  : " << t_u4     << " ms" << std::endl;
    std::cout << "Time MMX u=8  : " << t_u8     << " ms" << std::endl;
}
