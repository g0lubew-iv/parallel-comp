#include <sys/resource.h>
#include <bits/stdc++.h>
#include <x86intrin.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

using namespace std;

struct Stats {
    double minv, avg;
};

static volatile double g_sink = 0.0;


static inline uint64_t rdtsc_start() {
    unsigned int eax, ebx, ecx, edx; eax = 0;

    __asm__ __volatile__(
        "cpuid \n"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(eax)
        : "memory"
    );

    return __rdtsc(); // intrinsic for RDTSC
}


static inline uint64_t rdtsc_end() {
    unsigned int aux; uint64_t t = __rdtscp(&aux);
    unsigned int eax, ebx, ecx, edx; eax = 0;

    __asm__ __volatile__(
        "cpuid \n"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(eax)
        : "memory"
    );

    return t;
}


static inline uint64_t nsec_now_monotonic_raw() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}


static bool pin_to_cpu0() {
    cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(0, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    return rc == 0;
}


static void try_raise_priority() {
    setpriority(PRIO_PROCESS, 0, -20);
}


static bool is_power_of_two(size_t n) {
    return n && ((n & (n - 1)) == 0);
}


static void fft(vector<complex<double>>& a) {
    const size_t n = a.size();

    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;

        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;

        if (i < j) {
            swap(a[i], a[j]);
        }
    }

    // FFT stages
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * M_PI / double(len);
        complex<double> wlen(cos(ang), sin(ang));

        for (size_t i = 0; i < n; i += len) {
            complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; j++) {
                complex<double> u = a[i + j];
                complex<double> v = a[i + j + len / 2] * w;

                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}


static double checksum(const vector<complex<double>>& a) {
    double s = 0.0;
    for (auto &z : a) {
        s += z.real() * 0.000001 + z.imag() * 0.000002;
    }

    return s;
}


static Stats calc_min_avg(const vector<double>& v) {
    double mn = numeric_limits<double>::infinity();
    double sum = 0.0;

    for (double x : v) {
        mn = min(mn, x); sum += x;
    }

    return {mn, sum / double(v.size())};
}


int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const int K = 10000;

    size_t N = 2048; // FFT size; 2,048 points by default

    bool pinned = pin_to_cpu0(); try_raise_priority();

    vector<complex<double>> input(N);
    uint64_t seed = 123456789;

    auto rng = [&]() {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;

        return seed * 2685821657736338717ULL;
    };

    for (size_t i = 0; i < N; i++) {
        double re = (double)(rng() % 1000000) / 1000000.0;
        double im = (double)(rng() % 1000000) / 1000000.0;
        input[i]  = {re, im};
    }

    // Warm-up run
    auto a = input;
    fft(a);
    g_sink += checksum(a);

    vector<double> t_ns;      t_ns.reserve(K);
    vector<double> t_cycles;  t_cycles.reserve(K);

    std::ofstream out("res.csv");
    out << "run,n,time_ns,tsc_cycles\n";

    for (int i = 0; i < K; i++) {
        // Measure by clock_gettime (ns)
        auto     a1 = input;
        uint64_t t0 = nsec_now_monotonic_raw();
        fft(a1);
        uint64_t t1 = nsec_now_monotonic_raw();

        g_sink += checksum(a1);
        t_ns.push_back(double(t1 - t0));

        // Measure by TSC (cycles)
        auto     a2 = input;
        uint64_t c0 = rdtsc_start();
        fft(a2);
        uint64_t c1 = rdtsc_end();

        g_sink += checksum(a2);
        t_cycles.push_back(double(c1 - c0));

        uint64_t dt_ns = t1 - t0;
        uint64_t dc    = c1 - c0;

        // Write in file
        out << (i + 1) << "," << N << "," << dt_ns << "," << dc << "\n";
    }

    return 0;
}
