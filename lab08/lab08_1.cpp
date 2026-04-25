#include <iostream>
#include <random>
#include <cmath>

using namespace std;

extern "C" void vec_add_cuda(const float *a, const float *b, float *c, int n);

static void vec_add_cpu(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1024;

    float a[N], b[N], c_gpu[N], c_cpu[N];

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int i = 0; i < N; i++) {
        a[i] = dist(rng);
        b[i] = dist(rng);
        c_gpu[i] = 0.0f;
        c_cpu[i] = 0.0f;
    }

    // GPU
    vec_add_cuda(a, b, c_gpu, N);

    // CPU
    vec_add_cpu(a, b, c_cpu, N);

    // compare
    float maxAbsErr = 0.0f;
    int worstIdx = 0;

    for (int i = 0; i < N; i++) {
        float err = std::fabs(c_gpu[i] - c_cpu[i]);
        if (err > maxAbsErr) {
            maxAbsErr = err;
            worstIdx = i;
        }
    }

    cout << "max error |GPU-CPU| = " << maxAbsErr << " at i=" << worstIdx << endl;
    cout << "first 20 GPU results:" << endl;
    for (int i = 0; i < 20; i++) {
        cout << c_gpu[i] << " ";
    }
    cout << endl;

    if (maxAbsErr < 1e-6f)
        cout << "GPU calculations are correct!" << endl;
    else
        cout << "Mismatch detected!" << endl;

    return 0;
}
