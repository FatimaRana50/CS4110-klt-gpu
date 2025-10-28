/*********************************************************************
 * convolve_gpu.cu  (CPU-parity version)
 * GPU-accelerated convolution and gradient computation for KLT tracker
 * Matches CPU math: kernel width selection, normalization, zero padding.
 *********************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

#include "convolve_gpu.cuh"  // brings MAX_KERNEL_WIDTH_GPU and helpers

// -------------------------------------------------------------
// Constant memory for current kernel (one at a time)
// -------------------------------------------------------------
__constant__ float d_kernel_const[MAX_KERNEL_WIDTH_GPU];

// -------------------------------------------------------------
// CUDA stream (optional, simple single stream is fine for parity)
// -------------------------------------------------------------
static cudaStream_t g_stream = nullptr;

static inline void ensureStream() {
    if (!g_stream) CUDA_CHECK(cudaStreamCreate(&g_stream));
}

static inline void destroyStream() {
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = nullptr; }
}

// -------------------------------------------------------------
// Device kernels — zero-padding semantics to match CPU
// -------------------------------------------------------------
__global__ void ConvH_ZeroPad(
    const float* __restrict__ in,
    float* __restrict__ out,
    int w, int h,
    int kwidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int radius = kwidth / 2;

    // CPU semantics: write 0 near borders within radius
    if (x < radius || x >= (w - radius)) {
        out[y * w + x] = 0.0f;
        return;
    }

    float sum = 0.0f;
    const float* row = in + y * w + (x - radius);
    // CPU code reverses kernel order (equivalent to using reversed taps)
    for (int k = 0; k < kwidth; ++k) {
        sum += row[k] * d_kernel_const[kwidth - 1 - k];
    }
    out[y * w + x] = sum;
}

__global__ void ConvV_ZeroPad(
    const float* __restrict__ in,
    float* __restrict__ out,
    int w, int h,
    int kwidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int radius = kwidth / 2;

    // CPU semantics: write 0 near borders within radius
    if (y < radius || y >= (h - radius)) {
        out[y * w + x] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int startY = y - radius;
    for (int k = 0; k < kwidth; ++k) {
        int yy = startY + k;      // guaranteed in-range because of the check above
        sum += in[yy * w + x] * d_kernel_const[kwidth - 1 - k];
    }
    out[y * w + x] = sum;
}

// -------------------------------------------------------------
// CPU-parity kernel generation (matches original KLT behavior)
// factor = 0.01 cutoff, shift to center, normalize sum and deriv.
// -------------------------------------------------------------
static void makeKernelsCPUParity(
    float sigma,
    float* gauss, int* gauss_w,
    float* deriv, int* deriv_w)
{
    // guard sigma
    if (sigma < 0.5f) sigma = 0.5f;

    const int MAXW = MAX_KERNEL_WIDTH_GPU;
    const float factor = 0.01f;

    float gfull[MAXW];
    float dfull[MAXW];

    int hw = MAXW / 2;
    float max_gauss = 1.0f;
    float max_gderiv = sigma * expf(-0.5f);

    // full-length (untrimmed) taps
    for (int i = -hw; i <= hw; ++i) {
        float g = expf(-(i * i) / (2.0f * sigma * sigma));
        gfull[i + hw] = g;
        dfull[i + hw] = -i * g;
    }

    // determine effective widths by trimming 1% tails
    int gw = MAXW;
    for (int i = -hw; fabsf(gfull[i + hw] / max_gauss) < factor; ++i) gw -= 2;

    int dw = MAXW;
    for (int i = -hw; fabsf(dfull[i + hw] / max_gderiv) < factor; ++i) dw -= 2;

    // cap to MAXW and ensure odd
    if (gw > MAXW) gw = MAXW;
    if (dw > MAXW) dw = MAXW;
    if ((gw & 1) == 0) --gw;
    if ((dw & 1) == 0) --dw;

    // shift to center (remove leading zeros)
    int gshift = (MAXW - gw) / 2;
    int dshift = (MAXW - dw) / 2;
    for (int i = 0; i < gw; ++i) gauss[i] = gfull[i + gshift];
    for (int i = 0; i < dw; ++i) deriv[i] = dfull[i + dshift];

    // normalize: gauss sums to 1
    float den = 0.0f;
    for (int i = 0; i < gw; ++i) den += gauss[i];
    if (den != 0.0f) {
        for (int i = 0; i < gw; ++i) gauss[i] /= den;
    }

    // normalize derivative: sum(-i * deriv[i]) = 1 with i centered
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
// Low-level helpers required by convolve_gpu.cuh (GPU memory utils)
// -------------------------------------------------------------
float* allocateGPUFloatImage(int ncols, int nrows) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, sizeof(float) * ncols * nrows));
    return d;
}

void freeGPUFloatImage(float* d_img) {
    if (d_img) cudaFree(d_img);
}

void copyImageToGPU(const float* h_img, float* d_img, int ncols, int nrows) {
    CUDA_CHECK(cudaMemcpy(d_img, h_img, sizeof(float) * ncols * nrows, cudaMemcpyHostToDevice));
}

void copyImageFromGPU(float* h_img, const float* d_img, int ncols, int nrows) {
    CUDA_CHECK(cudaMemcpy(h_img, d_img, sizeof(float) * ncols * nrows, cudaMemcpyDeviceToHost));
}

// -------------------------------------------------------------
// Upload kernel taps into constant memory
// kernel_type is unused here; we upload whichever taps we need next.
// -------------------------------------------------------------
void uploadKernelToGPU(const float* kernel_data, int kernel_width, int /*kernel_type*/) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_const, kernel_data,
                                  kernel_width * sizeof(float), 0,
                                  cudaMemcpyHostToDevice));
}

// -------------------------------------------------------------
// Host wrappers that launch the zero-pad kernels
// -------------------------------------------------------------
void convolveImageHorizGPU(
    const float* d_imgin,
    int ncols, int nrows,
    const float* kernel_data, int kernel_width,
    float* d_imgout)
{
    ensureStream();
    uploadKernelToGPU(kernel_data, kernel_width, 0);

    dim3 block(32, 16);
    dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
    ConvH_ZeroPad<<<grid, block, 0, g_stream>>>(d_imgin, d_imgout, ncols, nrows, kernel_width);
    CUDA_CHECK(cudaGetLastError());
}

void convolveImageVertGPU(
    const float* d_imgin,
    int ncols, int nrows,
    const float* kernel_data, int kernel_width,
    float* d_imgout)
{
    ensureStream();
    uploadKernelToGPU(kernel_data, kernel_width, 0);

    dim3 block(32, 16);
    dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
    ConvV_ZeroPad<<<grid, block, 0, g_stream>>>(d_imgin, d_imgout, ncols, nrows, kernel_width);
    CUDA_CHECK(cudaGetLastError());
}

// -------------------------------------------------------------
// High-level GPU pipelines (separable, CPU-parity)
// -------------------------------------------------------------
void computeSmoothedImageGPU(
    const float* d_img,
    int ncols, int nrows,
    float sigma,
    float* d_smooth)
{
    ensureStream();

    float g[MAX_KERNEL_WIDTH_GPU];
    float dtmp[MAX_KERNEL_WIDTH_GPU]; int gw = 0, dw = 0;
    makeKernelsCPUParity(sigma, g, &gw, dtmp, &dw);

    float* d_tmp = allocateGPUFloatImage(ncols, nrows);

    // H with gauss
    convolveImageHorizGPU(d_img, ncols, nrows, g, gw, d_tmp);
    // V with gauss
    convolveImageVertGPU (d_tmp, ncols, nrows, g, gw, d_smooth);

    freeGPUFloatImage(d_tmp);
}

void computeGradientsGPU(
    const float* d_img,
    int ncols, int nrows,
    float sigma,
    float* d_gradx,
    float* d_grady)
{
    ensureStream();

    float g[MAX_KERNEL_WIDTH_GPU];
    float derv[MAX_KERNEL_WIDTH_GPU];
    int gw = 0, dw = 0;
    makeKernelsCPUParity(sigma, g, &gw, derv, &dw);

    float* d_tmp = allocateGPUFloatImage(ncols, nrows);

    // gradx = H(deriv) then V(gauss)
    convolveImageHorizGPU(d_img, ncols, nrows, derv, dw, d_tmp);
    convolveImageVertGPU (d_tmp, ncols, nrows, g,    gw, d_gradx);

    // grady = H(gauss) then V(deriv)
    convolveImageHorizGPU(d_img, ncols, nrows, g,    gw, d_tmp);
    convolveImageVertGPU (d_tmp, ncols, nrows, derv, dw, d_grady);

    freeGPUFloatImage(d_tmp);
}

// -------------------------------------------------------------
// Convenience: convert+upload (optional utility)
// -------------------------------------------------------------
float* convertAndUploadToGPU(KLT_PixelType* img, int ncols, int nrows)
{
    // CPU-side convert to float then upload
    const int N = ncols * nrows;
    float* h = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h[i] = (float)img[i];
    float* d = allocateGPUFloatImage(ncols, nrows);
    copyImageToGPU(h, d, ncols, nrows);
    free(h);
    return d;
}

void computeGradientsPipelineGPU(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady)
{
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

// -------------------------------------------------------------
// Legacy C API entry points (link targets from C code)
// -------------------------------------------------------------
extern "C" {

// Report kernel widths using the same rule as CPU (1% cutoff, capped)
void _KLTGetKernelWidths(float sigma, int *gauss_width, int *gaussderiv_width)
{
    float g[MAX_KERNEL_WIDTH_GPU], d[MAX_KERNEL_WIDTH_GPU];
    int gw = 0, dw = 0;
    makeKernelsCPUParity(sigma, g, &gw, d, &dw);
    if (gauss_width) *gauss_width = gw;
    if (gaussderiv_width) *gaussderiv_width = dw;
}

// Convert 8-bit pixels to float image (CPU-side, parity)
void _KLTToFloatImage(KLT_PixelType *img, int ncols, int nrows, _KLT_FloatImage floatimg)
{
    if (!img || !floatimg) return;
    floatimg->ncols = ncols;
    floatimg->nrows = nrows;
    float* dst = floatimg->data;
    const int N = ncols * nrows;
    for (int i = 0; i < N; ++i) dst[i] = (float)img[i];
}

// Smoothed image via separable Gaussian on GPU
void _KLTComputeSmoothedImage(_KLT_FloatImage img, float sigma, _KLT_FloatImage smooth)
{
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

// Gradients via separable derivative/Gaussian on GPU
void _KLTComputeGradients(_KLT_FloatImage img, float sigma,
                          _KLT_FloatImage gradx, _KLT_FloatImage grady)
{
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
