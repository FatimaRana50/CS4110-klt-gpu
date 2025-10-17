#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Found %d CUDA device(s)\n", deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            printf("cudaGetDeviceProperties(%d) failed: %s\n", i, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d: %s, totalGlobalMem=%llu, compute=%d.%d\n",
               i, prop.name, (unsigned long long)prop.totalGlobalMem, prop.major, prop.minor);
    }

    if (deviceCount > 0) {
        int dev = 0;
        err = cudaSetDevice(dev);
        if (err != cudaSuccess) {
            printf("cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));
            return 2;
        }
        // Try a small allocation to test runtime initialization
        void* p = nullptr;
        err = cudaMalloc(&p, 1024);
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return 3;
        }
        printf("cudaMalloc succeeded, ptr=%p\n", p);
        err = cudaFree(p);
        if (err != cudaSuccess) {
            printf("cudaFree failed: %s\n", cudaGetErrorString(err));
            return 4;
        }
        printf("cudaFree succeeded\n");
    }
    return 0;
}
