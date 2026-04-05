#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static inline void compiler_barrier() { asm volatile("" ::: "memory"); }

using clk = std::chrono::steady_clock;


enum class Pattern {
    Sequential = 0,
    RandomPointerChase = 1,
    RandomWithIndexArray = 2
};


static std::string pattern_name(Pattern p) {
    switch (p) {
        case Pattern::Sequential: return "sequential";
        case Pattern::RandomPointerChase: return "random_pointer_chase";
        case Pattern::RandomWithIndexArray: return "random_with_index_array";
    }
    return "unknown";
}


static volatile double g_sink = 0.0;


static std::vector<uint32_t> make_permutation(uint32_t n, uint64_t seed) {
    std::vector<uint32_t> p(n);
    std::iota(p.begin(), p.end(), 0u);
    std::mt19937_64 rng(seed);
    std::shuffle(p.begin(), p.end(), rng);
    return p;
}


static std::vector<uint32_t> make_cycle_next(const std::vector<uint32_t>& perm) {
    uint32_t n = (uint32_t)perm.size();
    std::vector<uint32_t> next(n);
    for (uint32_t i = 0; i + 1 < n; i++) next[perm[i]] = perm[i + 1];
    next[perm[n - 1]] = perm[0];
    return next;
}


static void touch(const std::vector<double>& a) {
    double s = 0;
    for (size_t i = 0; i < a.size(); i += 64 / sizeof(double)) s += a[i];
    g_sink = s;
}

struct Result {
    double ns_per_iter = 0.0;
    double ns_total = 0.0;
    double sum = 0.0;
};

static Result run_sum(
    const std::vector<double>& a,
    const std::vector<uint32_t>& idx,
    const std::vector<uint32_t>& next,
    Pattern pattern,
    size_t repeats
) {
    const size_t n = a.size();

    compiler_barrier();

    auto t0 = clk::now();
    double total = 0.0;

    if (pattern == Pattern::Sequential) {
        for (size_t r = 0; r < repeats; r++) {
            double s = 0.0;
            for (size_t i = 0; i < n; i++) s += a[i];
            total += s;
        }
    } else if (pattern == Pattern::RandomWithIndexArray) {

        for (size_t r = 0; r < repeats; r++) {
            double s = 0.0;
            for (size_t i = 0; i < n; i++) s += a[idx[i]];
            total += s;
        }
    } else { // RandomPointerChase

        for (size_t r = 0; r < repeats; r++) {
            double s = 0.0;
            uint32_t cur = 0;
            for (size_t i = 0; i < n; i++) {
                cur = next[cur];
                s += a[cur];
            }
            total += s;
        }
    }

    auto t1 = clk::now();
    compiler_barrier();

    std::chrono::duration<double> dt = t1 - t0;
    double ns_total = dt.count() * 1e9;
    double iters_total = double(n) * double(repeats);
    double ns_per_iter = ns_total / iters_total;

    g_sink = total; // не дать выкинуть
    return {ns_per_iter, ns_total, total};
}


static size_t choose_repeats(size_t n, double target_ms = 30.0) {

    double est_ns = double(n) * 2.0;
    double target_ns = target_ms * 1e6;
    size_t r = (size_t)std::max(1.0, target_ns / est_ns);

    if (r > 2000) {
        r = 2000;
    }
    return r;
}


static void run_range(size_t max_bytes, size_t step_bytes, Pattern pattern, uint64_t seed) {
    std::cout << "pattern,bytes,n_elems,repeats,ns_per_iter\n";

    for (size_t bytes = step_bytes; bytes <= max_bytes; bytes += step_bytes) {
        size_t n = bytes / sizeof(double);
        if (n < 2) continue;

        std::vector<double> a(n);

        for (size_t i = 0; i < n; i++) a[i] = double(i % 1024) * 0.25 + 1.0;

        std::vector<uint32_t> idx;
        std::vector<uint32_t> next;

        if (pattern == Pattern::RandomWithIndexArray || pattern == Pattern::RandomPointerChase) {
            auto perm = make_permutation((uint32_t)n, seed ^ (uint64_t)bytes);
            if (pattern == Pattern::RandomWithIndexArray) {
                idx = std::move(perm); // просто перестановка
            } else {
                next = make_cycle_next(perm);
            }
        }

        touch(a);
        if (!idx.empty()) {
            volatile uint32_t t = 0;
            for (size_t i = 0; i < idx.size(); i += 16) t ^= idx[i];
            (void)t;
        }
        if (!next.empty()) {
            volatile uint32_t t = 0;
            for (size_t i = 0; i < next.size(); i += 16) t ^= next[i];
            (void)t;
        }

        size_t warm_repeats = 2;
        (void)run_sum(a, idx, next, pattern, warm_repeats);

        size_t repeats = choose_repeats(n);
        auto res = run_sum(a, idx, next, pattern, repeats);

        std::cout
            << pattern_name(pattern) << ","
            << bytes << ","
            << n << ","
            << repeats << ","
            << res.ns_per_iter
            << "\n";
    }
}

int main(int argc, char** argv) {
    // cache_latency <pattern> <max_mb> <step_kb>

    std::string p = argv[1];
    double max_mb = std::stod(argv[2]);
    double step_kb = std::stod(argv[3]);

    Pattern pattern;
    if (p == "seq") pattern = Pattern::Sequential;
    else if (p == "rnd") pattern = Pattern::RandomPointerChase;
    else if (p == "idx") pattern = Pattern::RandomWithIndexArray;
    else {
        std::cerr << "Unknown pattern. Use seq|rnd|idx\n";
        return 1;
    }

    size_t max_bytes = (size_t)(max_mb * 1024.0 * 1024.0);
    size_t step_bytes = (size_t)(step_kb * 1024.0);

    run_range(max_bytes, step_bytes, pattern, 123456789ULL);
    return 0;
}
