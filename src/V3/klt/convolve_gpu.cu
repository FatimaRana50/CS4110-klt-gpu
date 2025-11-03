/*********************************************************************
 * convolve_gpu.cu (Shared-memory + Coalesced, stream-aware)
 * - Keeps EXACT function names / external behavior
 * - Uses shared-memory tiling for ConvH_ZeroPad / ConvV_ZeroPad
 * - Preserves zero-padding semantics and CPU-parity math
 *********************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

#include "convolve_gpu.cuh"

// -------------------------------------------------------------
// Debug macros (unchanged behavior)
// -------------------------------------------------------------
#ifndef CUDA_SAFE
#define CUDA_SAFE(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

#ifndef CUDA_SAFE_SYNC
#define CUDA_SAFE_SYNC() do { \
    cudaError_t _e = cudaDeviceSynchronize(); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA sync error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// -------------------------------------------------------------
// CUDA initialization guard (ensures a context exists early)
// -------------------------------------------------------------
static inline void ensureCudaReady() {
    static bool initialized = false;
    if (initialized) return;

    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        fprintf(stderr, "❌ cudaGetDeviceCount failed: %s\n", cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
    if (count <= 0) {
        fprintf(stderr, "❌ No CUDA devices found. Check drivers/CUDA install.\n");
        exit(EXIT_FAILURE);
    }
    CUDA_SAFE(cudaSetDevice(0));
    CUDA_SAFE(cudaFree(0)); // force context creation
    initialized = true;
}

// -------------------------------------------------------------
// Constant memory for kernel taps (already optimal)
// -------------------------------------------------------------
__constant__ float d_kernel_const[MAX_KERNEL_WIDTH_GPU];

// -------------------------------------------------------------
// Stream management (reused across all launches)
// -------------------------------------------------------------
static cudaStream_t g_stream = nullptr;
static inline void ensureStream() {
    ensureCudaReady();
    if (!g_stream) CUDA_SAFE(cudaStreamCreate(&g_stream));
}
static inline void destroyStream() {
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = nullptr; }
}

// =============================================================
// KERNELS: Shared-memory tiled + coalesced access
// =============================================================

// Horizontal convolution, zero-pad near left/right borders
// Shared tile size: (blockDim.x + 2*radius) * blockDim.y
__global__ void ConvH_ZeroPad(
    const float* __restrict__ in,
    float* __restrict__ out,
    int w, int h, int kwidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // output x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // output y
    if (y >= h) return;

    const int radius = kwidth >> 1;
    const int tileW  = blockDim.x + (radius << 1);
    const int tileH  = blockDim.y;

    extern __shared__ float smem[]; // size = tileW * tileH
    float* tile = smem;

    // Global x where this tile starts (including left halo)
    int gxs = blockIdx.x * blockDim.x - radius;
    int gys = y; // same row

    // Load this block row into shared, with halos. Coalesced by threads in x.
    // Each thread loads multiple elements with stride blockDim.x to cover tileW.
    for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
        int gx = gxs + tx;
        float v = 0.0f;
        if (gx >= 0 && gx < w) {
            v = in[gys * w + gx];
        }
        tile[threadIdx.y * tileW + tx] = v;
    }
    __syncthreads();

    if (x >= w) return;

    // Zero-padding semantics near left/right borders
    if (x < radius || x >= (w - radius)) {
        out[y * w + x] = 0.0f;
        return;
    }

    // Convolution from shared memory
    float sum = 0.0f;
    // In shared, the window for output x starts at offset = threadIdx.x
    int base = threadIdx.y * tileW + threadIdx.x;
    #pragma unroll
    for (int k = 0; k < kwidth; ++k) {
        sum += tile[base + k] * d_kernel_const[kwidth - 1 - k];
    }
    out[y * w + x] = sum;
}

// Vertical convolution, zero-pad near top/bottom borders
// Shared tile size: blockDim.x * (blockDim.y + 2*radius)
__global__ void ConvV_ZeroPad(
    const float* __restrict__ in,
    float* __restrict__ out,
    int w, int h, int kwidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // output x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // output y
    if (x >= w) return;

    const int radius = kwidth >> 1;
    const int tileW  = blockDim.x;
    const int tileH  = blockDim.y + (radius << 1);

    extern __shared__ float smem[]; // size = tileW * tileH
    float* tile = smem;

    // Global y where this tile starts (including top halo)
    int gys = blockIdx.y * blockDim.y - radius;
    int gxs = x;

    // Load this block column into shared, with halos. Coalesced by threads in x.
    // Each thread loads multiple rows with stride blockDim.y to cover tileH.
    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y) {
        int gy = gys + ty;
        float v = 0.0f;
        if (gy >= 0 && gy < h) {
            v = in[gy * w + gxs];
        }
        tile[ty * tileW + threadIdx.x] = v;
    }
    __syncthreads();

    if (y >= h) return;

    // Zero-padding semantics near top/bottom borders
    if (y < radius || y >= (h - radius)) {
        out[y * w + x] = 0.0f;
        return;
    }

    // Convolution from shared memory
    float sum = 0.0f;
    // In shared, the window for output y starts at offset = threadIdx.y
    int base = (threadIdx.y) * tileW + threadIdx.x;
    #pragma unroll
    for (int k = 0; k < kwidth; ++k) {
        sum += tile[base + k * tileW] * d_kernel_const[kwidth - 1 - k];
    }
    out[y * w + x] = sum;
}

// -------------------------------------------------------------
// CPU-parity kernel generation (unchanged behavior)
// -------------------------------------------------------------
static void makeKernelsCPUParity(float sigma, float* gauss, int* gauss_w,
                                 float* deriv, int* deriv_w)
{
    if (sigma < 0.5f) sigma = 0.5f;

    const int MAXW = MAX_KERNEL_WIDTH_GPU;
    const float factor = 0.01f;

    float gfull[MAXW];
    float dfull[MAXW];

    int hw = MAXW / 2;
    float max_gauss = 1.0f;
    float max_gderiv = sigma * expf(-0.5f);

    for (int i = -hw; i <= hw; ++i) {
        float g = expf(-(i * i) / (2.0f * sigma * sigma));
        gfull[i + hw] = g;
        dfull[i + hw] = -i * g;
    }

    int gw = MAXW, dw = MAXW;
    for (int i = -hw; fabsf(gfull[i + hw] / max_gauss) < factor; ++i) gw -= 2;
    for (int i = -hw; fabsf(dfull[i + hw] / max_gderiv) < factor; ++i) dw -= 2;

    if (gw > MAXW) gw = MAXW;
    if (dw > MAXW) dw = MAXW;
    if ((gw & 1) == 0) --gw;
    if ((dw & 1) == 0) --dw;

    int gshift = (MAXW - gw) / 2;
    int dshift = (MAXW - dw) / 2;
    for (int i = 0; i < gw; ++i) gauss[i] = gfull[i + gshift];
    for (int i = 0; i < dw; ++i) deriv[i] = dfull[i + dshift];

    float den = 0.0f;
    for (int i = 0; i < gw; ++i) den += gauss[i];
    if (den != 0.0f) {
        for (int i = 0; i < gw; ++i) gauss[i] /= den;
    }

    int dhw = dw / 2;
    den = 0.0f;
    for (int i = -dhw; i <= dhw; ++i) den -= i * deriv[i + dhw];
    if (den != 0.0f) {
        for (int i = -dhw; i <= dhw; ++i) deriv[i + dhw] /= den;
    }

    *gauss_w = gw;
    *deriv_w = dw;
}

// -------------------------------------------------------------
// GPU memory utilities (kept as-is for now; others depend on them)
// -------------------------------------------------------------
float* allocateGPUFloatImage(int ncols, int nrows) {
    ensureCudaReady();
    float* d = nullptr;
    CUDA_SAFE(cudaMalloc(&d, sizeof(float) * (size_t)ncols * (size_t)nrows));
    return d;
}
void freeGPUFloatImage(float* d_img) { if (d_img) cudaFree(d_img); }
void copyImageToGPU(const float* h, float* d, int ncols, int nrows) {
    ensureCudaReady();
    CUDA_SAFE(cudaMemcpy(d, h, sizeof(float) * (size_t)ncols * (size_t)nrows, cudaMemcpyHostToDevice));
}
void copyImageFromGPU(float* h, const float* d, int ncols, int nrows) {
    ensureCudaReady();
    CUDA_SAFE(cudaMemcpy(h, d, sizeof(float) * (size_t)ncols * (size_t)nrows, cudaMemcpyDeviceToHost));
}

// -------------------------------------------------------------
// Upload kernel taps to constant memory (unchanged API)
// -------------------------------------------------------------
void uploadKernelToGPU(const float* kernel_data, int kernel_width, int) {
    ensureCudaReady();
    if (kernel_width <= 0 || kernel_width > MAX_KERNEL_WIDTH_GPU) {
        fprintf(stderr, "❌ Invalid kernel width %d\n", kernel_width);
        exit(EXIT_FAILURE);
    }
    CUDA_SAFE(cudaMemcpyToSymbol(d_kernel_const, kernel_data,
                                 kernel_width * sizeof(float), 0,
                                 cudaMemcpyHostToDevice));
    // no sync needed; next kernel launch is on same device/stream
}

// -------------------------------------------------------------
// Convolution wrappers — launch with stream + dynamic shared memory
// -------------------------------------------------------------
static inline void chooseLaunch(dim3& block, dim3& grid, int w, int h) {
    // Good coalescing & occupancy for 320×240..HD:
    block = dim3(32, 8, 1);
    grid  = dim3((w + block.x - 1) / block.x,
                 (h + block.y - 1) / block.y, 1);
}

void convolveImageHorizGPU(const float* d_in, int ncols, int nrows,
                           const float* kernel_data, int kw, float* d_out)
{
    ensureStream();
    uploadKernelToGPU(kernel_data, kw, 0);

    dim3 block, grid;
    chooseLaunch(block, grid, ncols, nrows);

    const int radius = kw >> 1;
    const size_t smemBytes = (size_t)(block.y) * (block.x + (radius << 1)) * sizeof(float);

    ConvH_ZeroPad<<<grid, block, smemBytes, g_stream>>>(d_in, d_out, ncols, nrows, kw);
    CUDA_SAFE(cudaGetLastError());
    CUDA_SAFE(cudaStreamSynchronize(g_stream)); // keep host semantics identical
}

void convolveImageVertGPU(const float* d_in, int ncols, int nrows,
                          const float* kernel_data, int kw, float* d_out)
{
    ensureStream();
    uploadKernelToGPU(kernel_data, kw, 0);

    dim3 block, grid;
    chooseLaunch(block, grid, ncols, nrows);

    const int radius = kw >> 1;
    const size_t smemBytes = (size_t)(block.x) * (block.y + (radius << 1)) * sizeof(float);

    ConvV_ZeroPad<<<grid, block, smemBytes, g_stream>>>(d_in, d_out, ncols, nrows, kw);
    CUDA_SAFE(cudaGetLastError());
    CUDA_SAFE(cudaStreamSynchronize(g_stream)); // keep host semantics identical
}

// -------------------------------------------------------------
// High-level GPU pipelines (unchanged math / API)
// -------------------------------------------------------------
void computeSmoothedImageGPU(const float* d_img, int w, int h, float sigma, float* d_smooth)
{
    float g[MAX_KERNEL_WIDTH_GPU], dtmp[MAX_KERNEL_WIDTH_GPU];
    int gw = 0, dw = 0;
    makeKernelsCPUParity(sigma, g, &gw, dtmp, &dw);

    float* d_tmp = allocateGPUFloatImage(w, h);
    convolveImageHorizGPU(d_img, w, h, g, gw, d_tmp);
    convolveImageVertGPU (d_tmp, w, h, g, gw, d_smooth);
    freeGPUFloatImage(d_tmp);
}

void computeGradientsGPU(const float* d_img, int w, int h, float sigma,
                         float* d_gx, float* d_gy)
{
    float g[MAX_KERNEL_WIDTH_GPU], d[MAX_KERNEL_WIDTH_GPU];
    int gw = 0, dw = 0;
    makeKernelsCPUParity(sigma, g, &gw, d, &dw);

    float* d_tmp = allocateGPUFloatImage(w, h);
    // gradx = H(deriv) then V(gauss)
    convolveImageHorizGPU(d_img, w, h, d,  dw, d_tmp);
    convolveImageVertGPU (d_tmp, w, h, g,  gw, d_gx);
    // grady = H(gauss) then V(deriv)
    convolveImageHorizGPU(d_img, w, h, g,  gw, d_tmp);
    convolveImageVertGPU (d_tmp, w, h, d,  dw, d_gy);
    freeGPUFloatImage(d_tmp);
}

// -------------------------------------------------------------
// Legacy C API entry points (unchanged signatures/flow)
// -------------------------------------------------------------
extern "C" {

void _KLTGetKernelWidths(float sigma, int* gw, int* dw) {
    float g[MAX_KERNEL_WIDTH_GPU], d[MAX_KERNEL_WIDTH_GPU];
    makeKernelsCPUParity(sigma, g, gw, d, dw);
}

void _KLTToFloatImage(KLT_PixelType* img, int ncols, int nrows, _KLT_FloatImage f) {
    if (!img || !f) return;
    f->ncols = ncols;
    f->nrows = nrows;
    for (int i = 0; i < ncols * nrows; ++i) f->data[i] = (float)img[i];
}

void _KLTComputeSmoothedImage(_KLT_FloatImage img, float sigma, _KLT_FloatImage smooth) {
    if (!img || !smooth) return;
    const int w = img->ncols, h = img->nrows;
    float* d_in  = allocateGPUFloatImage(w, h);
    float* d_out = allocateGPUFloatImage(w, h);
    copyImageToGPU(img->data, d_in, w, h);
    computeSmoothedImageGPU(d_in, w, h, sigma, d_out);
    copyImageFromGPU(smooth->data, d_out, w, h);
    freeGPUFloatImage(d_in);
    freeGPUFloatImage(d_out);
}

void _KLTComputeGradients(_KLT_FloatImage img, float sigma,
                          _KLT_FloatImage gradx, _KLT_FloatImage grady) {
    if (!img || !gradx || !grady) return;
    const int w = img->ncols, h = img->nrows;
    float* d_in  = allocateGPUFloatImage(w, h);
    float* d_gx  = allocateGPUFloatImage(w, h);
    float* d_gy  = allocateGPUFloatImage(w, h);
    copyImageToGPU(img->data, d_in, w, h);
    computeGradientsGPU(d_in, w, h, sigma, d_gx, d_gy);
    copyImageFromGPU(gradx->data, d_gx, w, h);
    copyImageFromGPU(grady->data, d_gy, w, h);
    freeGPUFloatImage(d_in);
    freeGPUFloatImage(d_gx);
    freeGPUFloatImage(d_gy);
}

} // extern "C"