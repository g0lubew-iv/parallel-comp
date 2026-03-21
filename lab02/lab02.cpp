#include <algorithm>
#include <iostream>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>

using f32 = float;


static inline double now_sec() {
    using clock = std::chrono::steady_clock;

    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}


static inline double preal_gflops(std::size_t N, double t_sec) {
    const double V = 2.0 * std::pow(double(N), 3);

    return (V / t_sec) / 1e9;
}


static void fill_random(std::vector<f32>& M, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);

    for (auto &x : M) x = dist(rng);
}


static void zero(std::vector<f32>& C) { std::fill(C.begin(), C.end(), 0.0f); }


static bool compare_C(const std::vector<f32>& C1, const std::vector<f32>& C2, f32 rtol = 1e-4f, f32 atol = 1e-4f) {

    if (C1.size() != C2.size()) return false;

    for (std::size_t i = 0; i < C1.size(); ++i) {
        f32 a = C1[i], b = C2[i];
        f32 diff = std::fabs(a - b);
        f32 tol  = atol + rtol * std::fabs(b);
        if (diff > tol)
            return false;
    }

    return true;
}


static void mul_classic_ijk(const std::vector<f32>& A, const std::vector<f32>& B, std::vector<f32>& C, std::size_t N) {
    zero(C);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            f32 s = 0.0f;
            for (std::size_t k = 0; k < N; ++k)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    }
}


static void transpose(const std::vector<f32>& B, std::vector<f32>& BT, std::size_t N) {
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            BT[j * N + i] = B[i * N + j];
}


static void mul_transposeB(const std::vector<f32>& A, const std::vector<f32>& B, std::vector<f32>& C, std::size_t N, double* t_transpose_sec) {
    std::vector<f32> BT(N*N);

    double t0 = now_sec();
    transpose(B, BT, N);
    double t1 = now_sec();
    if (t_transpose_sec)
        *t_transpose_sec = (t1 - t0);

    zero(C);
    for (std::size_t i = 0; i < N; ++i) {
        const f32* arow = &A[i * N];
        for (std::size_t j = 0; j < N; ++j) {
            const f32* brow = &BT[j * N];
            f32 s = 0.0f;
            for (std::size_t k = 0; k < N; ++k)
                s += arow[k] * brow[k];
            C[i * N + j] = s;
        }
    }
}


static void mul_buffer_colB_unrollM(const std::vector<f32>& A, const std::vector<f32>& B, std::vector<f32>& C, std::size_t N, std::size_t M) {
    zero(C);
    if (M == 0)
        M = 1;

    if (M > N)
        M = N;

    std::vector<f32> tmp(N);

    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < N; ++k)
            tmp[k] = B[k * N + j];

        for (std::size_t i = 0; i < N; ++i) {
            const f32* arow = &A[i * N];
            f32 s = 0.0f;

            std::size_t k = 0;
            for (; k + M <= N; k += M) {
                for (std::size_t u = 0; u < M; ++u)
                    s += arow[k + u] * tmp[k + u];
            }
            for (; k < N; ++k)
                s += arow[k] * tmp[k];

            C[i * N + j] = s;
        }
    }
}


static void mul_blocked_unrollM(const std::vector<f32>& A, const std::vector<f32>& B, std::vector<f32>& C, std::size_t N, std::size_t S, std::size_t M) {
    zero(C);
    if (S == 0)
        S = 1;

    if (M == 0)
        M = 1;

    if (S > N)
        S = N;

    for (std::size_t ii = 0; ii < N; ii += S) {
        std::size_t i_max = std::min(ii + S, N);

        for (std::size_t jj = 0; jj < N; jj += S) {
            std::size_t j_max = std::min(jj + S, N);

            for (std::size_t kk = 0; kk < N; kk += S) {
                std::size_t k_max = std::min(kk + S, N);

                for (std::size_t i = ii; i < i_max; ++i) {
                    const f32* arow = &A[i * N];

                    for (std::size_t j = jj; j < j_max; ++j) {
                        f32 s = C[i * N + j];

                        std::size_t k = kk;

                        for (; k + M <= k_max; k += M) {
                            for (std::size_t u = 0; u < M; ++u)
                                s += arow[k + u] * B[(k + u) * N + j];
                        }

                        for (; k < k_max; ++k)
                            s += arow[k] * B[k * N + j];

                        C[i * N + j] = s;
                    }
                }
            }
        }
    }
}


template<class Fn>
static double bench_best(Fn fn, int repeats) {
    double best = 1e100;

    for (int r = 0; r < repeats; ++r) {
        double t0 = now_sec();
        fn();
        double t1 = now_sec();
        best = std::min(best, t1 - t0);
    }

    return best;
}


static void write_csv_header(std::ofstream& out) {
    out << "mode,N,S,M,algo,t_sec,preal_gflops,extra\n";
}


static void write_csv_row(std::ofstream& out, const std::string& mode, std::size_t N, std::size_t S, std::size_t M, const std::string& algo, double t_sec, double preal, const std::string& extra = "") {
    out << mode << "," << N << "," << S << "," << M << "," << algo << "," << std::setprecision(10) << t_sec << "," << std::setprecision(6) << preal << "," << extra << "\n";
}


static bool arg_eq(const char* a, const char* b) { return std::string(a) == std::string(b); }


int main(int argc, char** argv) {
    std::string mode = "single";
    std::size_t N = 512, S = 64, M = 4;
    std::string out_path = "results.csv";
    int repeats = 1;

    for (int i = 1; i < argc; ++i) {
        if (arg_eq(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (arg_eq(argv[i], "--N") && i + 1 < argc) N = std::stoull(argv[++i]);
        else if (arg_eq(argv[i], "--S") && i + 1 < argc) S = std::stoull(argv[++i]);
        else if (arg_eq(argv[i], "--M") && i + 1 < argc) M = std::stoull(argv[++i]);
        else if (arg_eq(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (arg_eq(argv[i], "--repeats") && i + 1 < argc) repeats = std::stoi(argv[++i]);
    }

    std::ofstream out(out_path, std::ios::out);
    write_csv_header(out);

    auto run_single = [&](std::size_t N0, std::size_t S0, std::size_t M0, const std::string& mode_name) {
        std::vector<f32> A(N0*N0), B(N0*N0);
        std::vector<f32> C1(N0*N0), C2(N0*N0), C3(N0*N0), C4(N0*N0);

        fill_random(A, 1);
        fill_random(B, 2);

        // classic
        double t1 = bench_best([&]{ mul_classic_ijk(A, B, C1, N0); }, repeats);
        write_csv_row(out, mode_name, N0, S0, M0, "classic", t1, preal_gflops(N0, t1));

        // transpose B
        double t_tr = 0.0;
        double t2_total = bench_best([&]{ mul_transposeB(A, B, C2, N0, &t_tr); }, repeats);
        double t2_mul_only = t2_total - t_tr;

        write_csv_row(out, mode_name, N0, S0, M0, "transpose_total", t2_total, preal_gflops(N0, t2_total));
        write_csv_row(out, mode_name, N0, S0, M0, "transpose_only",  t_tr,     0.0, "P_real not applicable");
        write_csv_row(out, mode_name, N0, S0, M0, "transpose_mul_only", t2_mul_only,
                      preal_gflops(N0, t2_mul_only));

        if (!compare_C(C1, C2)) {
            write_csv_row(out, mode_name, N0, S0, M0, "check", 0.0, 0.0, "Mismatch: classic vs transpose");
            return;
        }

        // buffered column + unroll M
        double t3 = bench_best([&]{ mul_buffer_colB_unrollM(A, B, C3, N0, M0); }, repeats);
        write_csv_row(out, mode_name, N0, S0, M0, "buffered", t3, preal_gflops(N0, t3));
        if (!compare_C(C1, C3)) {
            write_csv_row(out, mode_name, N0, S0, M0, "check", 0.0, 0.0, "Mismatch: classic vs buffered");
            return;
        }

        // blocked + unroll M, block size S
        double t4 = bench_best([&]{ mul_blocked_unrollM(A, B, C4, N0, S0, M0); }, repeats);
        write_csv_row(out, mode_name, N0, S0, M0, "blocked", t4, preal_gflops(N0, t4));
        if (!compare_C(C1, C4)) {
            write_csv_row(out, mode_name, N0, S0, M0, "check", 0.0, 0.0, "Mismatch: classic vs blocked");
            return;
        }
    };

    if (mode == "single") {
        run_single(N, S, M, "single");
    } else if (mode == "sweepS") {
        for (std::size_t Ss = 1; Ss <= N; Ss <<= 1) {
            run_single(N, Ss, M, "sweepS");
        }
    } else if (mode == "sweepM") {
        for (std::size_t Mm = 1; Mm <= 16; Mm <<= 1) {
            run_single(N, S, Mm, "sweepM");
        }
    }

    return 0;
}
