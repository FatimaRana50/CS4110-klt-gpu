#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    int count=0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("CUDA devices: %d\n", count);
    return 0;
}

