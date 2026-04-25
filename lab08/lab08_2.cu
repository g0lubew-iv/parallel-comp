#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

static void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

static int getAttr(int dev, cudaDeviceAttr attr) {
    int v = 0;
    checkCuda(cudaDeviceGetAttribute(&v, attr, dev));
    return v;
}

extern "C" void print_cuda_device_info() {
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count));
    std::printf("CUDA devices found: %d\n\n", count);
    if (count == 0) {
        return;
    }

    for (int dev = 0; dev < count; dev++) {
        cudaDeviceProp p{};
        checkCuda(cudaGetDeviceProperties(&p, dev));

        std::printf("Device %d\n", dev);
        std::printf("Name: %s\n", p.name);

        std::printf("Total global memory: %.2f MiB\n",
                    (double)p.totalGlobalMem / (1024.0 * 1024.0));
        std::printf("Constant memory: %.2f KiB\n",
                    (double)p.totalConstMem / 1024.0);
        std::printf("Shared memory per block: %.2f KiB\n",
                    (double)p.sharedMemPerBlock / 1024.0);

        std::printf("Registers per block: %d\n", p.regsPerBlock);
        std::printf("Warp size: %d\n", p.warpSize);
        std::printf("Max threads per block: %d\n", p.maxThreadsPerBlock);

        std::printf("Compute capability: %d.%d\n", p.major, p.minor);
        std::printf("Multiprocessor count (SMs): %d\n", p.multiProcessorCount);

        int coreClockKHz = getAttr(dev, cudaDevAttrClockRate);
        int memClockKHz  = getAttr(dev, cudaDevAttrMemoryClockRate);
        std::printf("Core clock rate: %.2f MHz\n", coreClockKHz / 1000.0);
        std::printf("Memory clock rate: %.2f MHz\n", memClockKHz / 1000.0);

        std::printf("L2 cache size: %.2f KiB\n", (double)p.l2CacheSize / 1024.0);

        std::printf("Memory bus width: %d bits\n", p.memoryBusWidth);

        std::printf("Max threads dim (block): [%d, %d, %d]\n",
                    p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        std::printf("Max grid size: [%d, %d, %d]\n",
                    p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    }
}
