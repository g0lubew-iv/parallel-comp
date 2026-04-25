#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void VecAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

static void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

extern "C" void vec_add_cuda(const float *a, const float *b, float *c, int n) {
    int sizeBytes = n * (int) sizeof(float);

    float *a_gpu = nullptr, *b_gpu = nullptr, *c_gpu = nullptr;

    checkCuda(cudaMalloc((void**)&a_gpu, sizeBytes));
    checkCuda(cudaMalloc((void**)&b_gpu, sizeBytes));
    checkCuda(cudaMalloc((void**)&c_gpu, sizeBytes));

    checkCuda(cudaMemcpy(a_gpu, a, sizeBytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(b_gpu, b, sizeBytes, cudaMemcpyHostToDevice));

    dim3 threads(512, 1, 1);
    dim3 blocks((n + threads.x - 1) / threads.x, 1, 1);

    VecAddKernel<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(c, c_gpu, sizeBytes, cudaMemcpyDeviceToHost));

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}
