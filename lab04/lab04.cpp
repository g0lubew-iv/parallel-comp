#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>

enum Pattern { SEQ, RND, IDX };

static const char* patName(Pattern p) {
    if (p == SEQ) return "sequential";
    if (p == RND) return "random_pointer_chase";
    return "random_with_index_array";
}

static Pattern parsePattern(const char* s) {
    if (strcmp(s, "seq") == 0) return SEQ;
    if (strcmp(s, "rnd") == 0) return RND;
    if (strcmp(s, "idx") == 0) return IDX;

    printf("Unknown pattern: %s (use seq|rnd|idx)\n", s);
    exit(1);
}

static const int MAX_MB = 150;
static const size_t MAX_BYTES = (size_t)MAX_MB * 1024ull * 1024ull;
static const size_t MAX_N = MAX_BYTES / sizeof(double);

alignas (64) static double   A[MAX_N];
alignas (64) static uint32_t P[MAX_N];

static volatile double sink = 0.0;

static inline uint64_t xorshift64(uint64_t& x) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 2685821657736338717ULL;
}

static void makePermutation(uint32_t* p, uint32_t n, uint64_t seed) {
    for (uint32_t i = 0; i < n; i++) p[i] = i;

    uint64_t s = seed ? seed : 1234567ULL;
    for (uint32_t i = n - 1; i > 0; i--) {
        uint32_t j = (uint32_t)(xorshift64(s) % (uint64_t)(i + 1));
        uint32_t tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
    }
}

static void permToNext(uint32_t* p, uint32_t n) {
    const uint32_t MARK = 0x80000000u;

    uint32_t first = p[0];
    for (uint32_t i = 0; i + 1 < n; i++) {
        uint32_t from = p[i];
        uint32_t to   = p[i + 1];
        p[from] = (to | MARK);
    }
    uint32_t last = p[n - 1];
    p[last] = (first | MARK);

    for (uint32_t i = 0; i < n; i++) p[i] &= ~MARK;
}

static void touchData(const double* a, size_t n) {
    double s = 0.0;
    size_t step = 64 / sizeof(double);
    for (size_t i = 0; i < n; i += step) s += a[i];
    sink = s;
}

static size_t chooseRepeats(size_t n) {
    double target_ms = 200.0;
    double est_ns = (double)n * 5.0; // грубо
    size_t r = (size_t)((target_ms * 1e6) / est_ns);
    if (r < 1) r = 1;
    if (r > 2000) r = 2000;
    return r;
}

static double measureNsPerIter(Pattern pat, size_t n, size_t repeats) {
    using clk = std::chrono::steady_clock;

    auto t0 = clk::now();
    double total = 0.0;

    if (pat == SEQ) {
        for (size_t r = 0; r < repeats; r++) {
            double s = 0.0;
            for (size_t i = 0; i < n; i++) s += A[i];
            total += s;
        }
    } else if (pat == IDX) {
        for (size_t r = 0; r < repeats; r++) {
            double s = 0.0;
            for (size_t i = 0; i < n; i++) s += A[P[i]];
            total += s;
        }
    } else { // RND pointer chase
        for (size_t r = 0; r < repeats; r++) {
            double s = 0.0;
            uint32_t cur = 0;
            for (size_t i = 0; i < n; i++) {
                cur = P[cur];
                s += A[cur];
            }
            total += s;
        }
    }

    auto t1 = clk::now();
    sink = total;

    std::chrono::duration<double> dt = t1 - t0;
    double ns_total = dt.count() * 1e9;
    return ns_total / ((double)n * (double)repeats);
}

int main(int argc, char** argv) {
    Pattern pat = parsePattern(argv[1]);
    double max_mb = atof(argv[2]);
    double step_kb = atof(argv[3]);

    size_t max_bytes = (size_t)(max_mb * 1024.0 * 1024.0);
    size_t step_bytes = (size_t)(step_kb * 1024.0);

    printf("pattern,bytes,n_elems,repeats,ns_per_iter\n");

    for (size_t bytes = step_bytes; bytes <= max_bytes; bytes += step_bytes) {
        size_t n = bytes / sizeof(double);
        if (n < 2) {
            continue;
        }

        for (size_t i = 0; i < n; i++) {
            A[i] = (double)(i % 1024) * 0.25 + 1.0;
        }

        if (pat == IDX || pat == RND) {
            makePermutation(P, (uint32_t)n, 123456789ULL ^ (uint64_t)bytes);
            if (pat == RND) permToNext(P, (uint32_t)n);
        }

        touchData(A, n);
        (void) measureNsPerIter(pat, n, 2);

        size_t repeats = chooseRepeats(n);
        double ns = measureNsPerIter(pat, n, repeats);

        printf("%s,%zu,%zu,%zu,%.6f\n", patName(pat), bytes, n, repeats, ns);
    }

    return 0;
}
