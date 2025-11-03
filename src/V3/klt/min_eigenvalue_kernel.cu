#include <cuda_runtime.h>
#include <math.h>
#include "min_eigenvalue_kernel.cuh"

__global__ void computeMinEigenvaluesKernel(
    const float* gradx,
    const float* grady,
    int* pointlist,
    int ncols,
    int nrows,
    int window_hw,
    int window_hh,
    int borderx,
    int bordery,
    int nSkippedPixels)
{
    // Calculate 2D thread indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate actual pixel coordinates with skipping
    int x = borderx + idx * (nSkippedPixels + 1);
    int y = bordery + idy * (nSkippedPixels + 1);
    
    // Check if this thread processes a valid pixel
    if (x >= ncols - borderx || y >= nrows - bordery) {
        return;
    }
    
    // Compute structure tensor over the window
    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    
    for (int yy = y - window_hh; yy <= y + window_hh; yy++) {
        for (int xx = x - window_hw; xx <= x + window_hw; xx++) {
            // Ensure we don't read out of bounds (though borders should prevent this)
            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows) {
                int pos = yy * ncols + xx;
                float gx = gradx[pos];
                float gy = grady[pos];
                
                gxx += gx * gx;
                gxy += gx * gy;
                gyy += gy * gy;
            }
        }
    }
    
    // Compute minimum eigenvalue: (gxx + gyy - sqrt((gxx - gyy)^2 + 4*gxy^2)) / 2
    float diff = gxx - gyy;
    float discriminant = diff * diff + 4.0f * gxy * gxy;
    
    // Handle potential numerical issues
    if (discriminant < 0.0f) discriminant = 0.0f;
    
    float min_eigenvalue = 0.5f * (gxx + gyy - sqrtf(discriminant));
    
    // Ensure eigenvalue is non-negative
    if (min_eigenvalue < 0.0f) min_eigenvalue = 0.0f;
    
    // Convert to integer for compatibility
    int eigen_int = (int)min_eigenvalue;
    
    // Calculate output position - we need to map 2D grid to 1D array
    int grid_width = (ncols - 2 * borderx + nSkippedPixels) / (nSkippedPixels + 1);
    int output_pos = (idy * grid_width + idx) * 3;
    
    // Store results
    pointlist[output_pos] = x;
    pointlist[output_pos + 1] = y;
    pointlist[output_pos + 2] = eigen_int;
}

// Wrapper function with C linkage
#ifdef __cplusplus
extern "C" {
#endif

void computeMinEigenvaluesGPU(
    const float* gradx_data,
    const float* grady_data,
    int* pointlist,
    int ncols,
    int nrows,
    int window_hw,
    int window_hh,
    int borderx,
    int bordery,
    int nSkippedPixels)
{
    float *d_gradx = NULL, *d_grady = NULL;
    int *d_pointlist = NULL;
    
    // Calculate grid dimensions
    int grid_width = (ncols - 2 * borderx + nSkippedPixels) / (nSkippedPixels + 1);
    int grid_height = (nrows - 2 * bordery + nSkippedPixels) / (nSkippedPixels + 1);
    
    if (grid_width <= 0 || grid_height <= 0) {
        return; // No valid pixels to process
    }
    
    size_t image_size = ncols * nrows * sizeof(float);
    size_t pointlist_size = grid_width * grid_height * 3 * sizeof(int);
    
    // Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc(&d_gradx, image_size);
    if (err != cudaSuccess) { /* handle error */ }
    
    err = cudaMalloc(&d_grady, image_size);
    if (err != cudaSuccess) { /* handle error */ }
    
    err = cudaMalloc(&d_pointlist, pointlist_size);
    if (err != cudaSuccess) { /* handle error */ }
    
    // Copy input data to GPU
    cudaMemcpy(d_gradx, gradx_data, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grady, grady_data, image_size, cudaMemcpyHostToDevice);
    
    // Initialize pointlist to zeros
    cudaMemset(d_pointlist, 0, pointlist_size);
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((grid_width + blockSize.x - 1) / blockSize.x,
                  (grid_height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    computeMinEigenvaluesKernel<<<gridSize, blockSize>>>(
        d_gradx, d_grady, d_pointlist, ncols, nrows,
        window_hw, window_hh, borderx, bordery, nSkippedPixels);
    
    // Check for kernel launch errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle kernel error
    }
    
    // Copy results back to CPU
    cudaMemcpy(pointlist, d_pointlist, pointlist_size, cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_gradx);
    cudaFree(d_grady);
    cudaFree(d_pointlist);
}


#ifdef __cplusplus
}
#endif