/*********************************************************************
 * convolve.c
 *********************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define MAX_KERNEL_WIDTH 71

typedef struct {
  int width;
  float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

// Debug / error helpers
static inline void CHECK_CUDA(const char *where) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static void dump_stats(const float *arr, int ncols, int nrows, const char *name, int maxprint) {
    int N = ncols * nrows;
    int nan = 0, infs = 0;
    for (int i = 0; i < N; ++i) {
        if (isnan(arr[i])) nan++;
        if (isinf(arr[i])) infs++;
    }
    fprintf(stderr, "[DEBUG] %s: total=%d nan=%d inf=%d\n", name, N, nan, infs);
    int printed = 0;
    for (int r = 0; r < nrows && printed < maxprint; ++r) {
        for (int c = 0; c < ncols && printed < maxprint; ++c, ++printed) {
            fprintf(stderr, "%s[%d,%d]=%g\n", name, r, c, arr[r * ncols + c]);
        }
    }
}

/* Simple host-side separable convolution (fallback) */
static void host_convolve_separable(const _KLT_FloatImage imgin,
                                   const ConvolutionKernel horiz_kernel,
                                   const ConvolutionKernel vert_kernel,
                                   _KLT_FloatImage imgout)
{
  int ncols = imgin->ncols;
  int nrows = imgin->nrows;
  int hw_h = horiz_kernel.width / 2;
  int hw_v = vert_kernel.width / 2;
  size_t N = ncols * nrows;
  float *tmp = (float*) malloc(N * sizeof(float));
  if (!tmp) {
    KLTError("host_convolve_separable: malloc failed");
    return;
  }

  /* horizontal pass */
  for (int r = 0; r < nrows; ++r) {
    for (int c = 0; c < ncols; ++c) {
      float sum = 0.0f;
      for (int k = -hw_h; k <= hw_h; ++k) {
        int cc = c + k;
        if (cc < 0) cc = 0;
        if (cc >= ncols) cc = ncols - 1;
        sum += imgin->data[r * ncols + cc] * horiz_kernel.data[k + hw_h];
      }
      tmp[r * ncols + c] = sum;
    }
  }

  /* vertical pass */
  for (int r = 0; r < nrows; ++r) {
    for (int c = 0; c < ncols; ++c) {
      float sum = 0.0f;
      for (int k = -hw_v; k <= hw_v; ++k) {
        int rr = r + k;
        if (rr < 0) rr = 0;
        if (rr >= nrows) rr = nrows - 1;
        sum += tmp[rr * ncols + c] * vert_kernel.data[k + hw_v];
      }
      imgout->data[r * ncols + c] = sum;
    }
  }

  free(tmp);
}

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;


/*********************************************************************
 * _KLTToFloatImage
 *********************************************************************/
void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols * nrows;
  float *ptrout = floatimg->data;

  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend) *ptrout++ = (float)*img++;
}


/*********************************************************************
 * _computeKernels
 *********************************************************************/
static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  const int hw = MAX_KERNEL_WIDTH / 2;
  float max_gauss = 1.0f, max_gaussderiv = (float)(sigma * exp(-0.5f));

  for (i = -hw; i <= hw; i++) {
    gauss->data[i + hw] = (float)exp(-i * i / (2 * sigma * sigma));
    gaussderiv->data[i + hw] = -i * gauss->data[i + hw];
  }

  gauss->width = MAX_KERNEL_WIDTH;
  for (i = -hw; fabs(gauss->data[i + hw] / max_gauss) < factor; i++, gauss->width -= 2);
  gaussderiv->width = MAX_KERNEL_WIDTH;
  for (i = -hw; fabs(gaussderiv->data[i + hw] / max_gaussderiv) < factor; i++, gaussderiv->width -= 2);

  if (gauss->width == MAX_KERNEL_WIDTH || gaussderiv->width == MAX_KERNEL_WIDTH)
    KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d too small for sigma=%f", MAX_KERNEL_WIDTH, sigma);

  for (i = 0; i < gauss->width; i++)
    gauss->data[i] = gauss->data[i + (MAX_KERNEL_WIDTH - gauss->width) / 2];
  for (i = 0; i < gaussderiv->width; i++)
    gaussderiv->data[i] = gaussderiv->data[i + (MAX_KERNEL_WIDTH - gaussderiv->width) / 2];

  const int hw2 = gaussderiv->width / 2;
  float den = 0.0;
  for (i = 0; i < gauss->width; i++) den += gauss->data[i];
  for (i = 0; i < gauss->width; i++) gauss->data[i] /= den;

  den = 0.0;
  for (i = -hw2; i <= hw2; i++) den -= i * gaussderiv->data[i + hw2];
  for (i = -hw2; i <= hw2; i++) gaussderiv->data[i + hw2] /= den;

  sigma_last = sigma;
}


/*********************************************************************
 * _KLTGetKernelWidths
 *********************************************************************/
void _KLTGetKernelWidths(float sigma, int *gauss_width, int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}


/*********************************************************************
 * _convolveImageHoriz
 *********************************************************************/
__global__ void zeroLeftCols(float* out, int radius, int ncols, int row)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < radius)
    out[row * ncols + i] = 0.0f;
}

__global__ void convolveRow(const float* in, const float* kernel,
                            float* out, int ncols, int row, int kWidth)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int radius = kWidth / 2;
  if (i < radius || i >= ncols - radius) return;

  float sum = 0.0f;
  for (int k = 0; k < kWidth; k++)
    sum += in[row * ncols + i + k - radius] * kernel[k];
  out[row * ncols + i] = sum;
}

__global__ void zeroRightCols(float* out, int start, int ncols, int row)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + start;
  if (i < ncols)
    out[row * ncols + i] = 0.0f;
}

/* Prototype for horizontal 2D kernel (defined later) */
__global__ void _convolveImageHoriz2D(
    const float* imgin_data,
    int ncols, int nrows,
    const float* kernel_data, int kernel_width,
    float* imgout_data);

static void _convolveImageHoriz(
  _KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout)
{
  assert(kernel.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  int ncols = imgin->ncols;
  int nrows = imgin->nrows;
  int radius = kernel.width / 2;

  float *d_in, *d_out, *d_kernel;
  size_t imgSize = ncols * nrows * sizeof(float);
  size_t kSize = kernel.width * sizeof(float);

  cudaMalloc(&d_in, imgSize);
  cudaMalloc(&d_out, imgSize);
  cudaMalloc(&d_kernel, kSize);
  /* ensure output buffer starts zeroed to avoid leftover values */
  cudaMemset(d_out, 0, imgSize);
  CHECK_CUDA("cudaMemset d_out");

  cudaMemcpy(d_in, imgin->data, imgSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel.data, kSize, cudaMemcpyHostToDevice);
  CHECK_CUDA("cudaMemcpy inputs to device");

  /* Launch a single 2D kernel to compute the horizontal convolution for all
     pixels. This avoids per-row kernel launch complexity and should be more
     reliable across devices. */
  /* choose between per-row kernels (legacy) and the single 2D kernel.  */
  char *use_per_row = getenv("KLT_USE_PER_ROW");
  float ms = 0.0f;
  if (use_per_row && use_per_row[0] != '\0') {
    /* Per-row approach: launch three 1D kernels per row */
    const int threads = 256;
    const int gridx = (ncols + threads - 1) / threads;
    cudaEvent_t tstart_row, tstop_row;
    cudaEventCreate(&tstart_row);
    cudaEventCreate(&tstop_row);
    cudaEventRecord(tstart_row, 0);

    for (int r = 0; r < nrows; ++r) {
      /* zero left border */
      zeroLeftCols<<<gridx, threads>>>(d_out, radius, ncols, r);
      /* convolve interior pixels for this row */
      convolveRow<<<gridx, threads>>>(d_in, d_kernel, d_out, ncols, r, kernel.width);
      /* zero right border: start index = ncols - radius */
      zeroRightCols<<<gridx, threads>>>(d_out, ncols - radius, ncols, r);
    }

    cudaEventRecord(tstop_row, 0);
    cudaEventSynchronize(tstop_row);
    cudaEventElapsedTime(&ms, tstart_row, tstop_row);
    CHECK_CUDA("_convolveImageHoriz per-row kernels");
    cudaEventDestroy(tstart_row);
    cudaEventDestroy(tstop_row);

  } else {
    dim3 block2(16, 16);
    dim3 grid2((ncols + block2.x - 1) / block2.x,
               (nrows + block2.y - 1) / block2.y);
    /* timing for horizontal convolution */
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    cudaEventRecord(tstart, 0);
    _convolveImageHoriz2D<<<grid2, block2>>>(d_in, ncols, nrows, d_kernel, kernel.width, d_out);
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&ms, tstart, tstop);
    CHECK_CUDA("_convolveImageHoriz kernels");
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);
  }

  /* append timing to profiling file (best-effort) */
  {
    FILE *ft = fopen("profiling/gpu_timing.txt", "a");
    if (ft) {
      fprintf(ft, "convolve_horiz %g\n", (double)ms);
      fclose(ft);
    }
  }

  /* Optional device-side checks: copy a small patch from device to host to
     confirm whether d_out contains zeros (kernel issue) or the host copy is
     failing. Enable by setting environment variable KLT_DEVICE_CHECKS=1. */
  {
    char *devchk = getenv("KLT_DEVICE_CHECKS");
    if (devchk && devchk[0] != '\0') {
      int prow = nrows < 8 ? nrows : 8;
      int pcol = ncols < 16 ? ncols : 16;
      float *host_patch = (float*)malloc(sizeof(float) * prow * pcol);
      if (host_patch) {
        for (int r = 0; r < prow; ++r) {
          size_t offset = (size_t)r * (size_t)ncols;
          cudaMemcpy(host_patch + r * pcol, d_out + offset, sizeof(float) * pcol, cudaMemcpyDeviceToHost);
        }
        /* Print a compact summary */
        int printed = 0;
        double sum = 0.0;
        int zeros = 0;
        for (int r = 0; r < prow; ++r) {
          for (int c = 0; c < pcol; ++c) {
            float v = host_patch[r * pcol + c];
            sum += fabs((double)v);
            if (v == 0.0f) zeros++;
            if (printed < 64) {
              fprintf(stderr, "[DEVCHK] d_out[%d,%d]=%g\n", r, c, v);
              printed++;
            }
          }
        }
        fprintf(stderr, "[DEVCHK] patch sum_abs=%g zeros=%d/%d\n", sum, zeros, prow * pcol);
        free(host_patch);
      } else {
        fprintf(stderr, "[DEVCHK] malloc failed for host_patch\n");
      }
    }
  }
  cudaMemcpy(imgout->data, d_out, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_kernel);
}


/*********************************************************************
 * Fully GPU offloaded _convolveImageVert
 *********************************************************************/
__global__ void _convolveImageVert(
    const float* imgin_data,
    int ncols, int nrows,
    const float* kernel_data, int kernel_width,
    float* imgout_data)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= ncols || row >= nrows) return;

  int radius = kernel_width / 2;
  int idx = row * ncols + col;

  if (row < radius || row >= nrows - radius) {
    imgout_data[idx] = 0.0f;
    return;
  }

  float sum = 0.0f;
  for (int k = -radius; k <= radius; k++) {
    int r = row + k;
    if (r >= 0 && r < nrows)
      sum += imgin_data[r * ncols + col] * kernel_data[k + radius];
  }

  imgout_data[idx] = sum;
}

/* Prototype for horizontal 2D kernel (defined later) */
__global__ void _convolveImageHoriz2D(
  const float* imgin_data,
  int ncols, int nrows,
  const float* kernel_data, int kernel_width,
  float* imgout_data);
/* 2D horizontal convolution kernel: each thread computes one output pixel
   using the 1D horizontal kernel. Borders are handled by clamping. */
__global__ void _convolveImageHoriz2D(
    const float* imgin_data,
    int ncols, int nrows,
    const float* kernel_data, int kernel_width,
    float* imgout_data)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= ncols || row >= nrows) return;

  int radius = kernel_width / 2;
  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    int cc = col + k;
    if (cc < 0) cc = 0;
    if (cc >= ncols) cc = ncols - 1;
    sum += imgin_data[row * ncols + cc] * kernel_data[k + radius];
  }
  imgout_data[row * ncols + col] = sum;
}


/*********************************************************************
 * _convolveSeparate
 *********************************************************************/
static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  _KLT_FloatImage tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);

#ifdef DEBUG
  /* Debug: check result of horizontal convolution */
  dump_stats(tmpimg->data, tmpimg->ncols, tmpimg->nrows, "tmpimg_after_horiz", 16);
#endif

  // Allocate and run vertical kernel
  float *d_in, *d_out, *d_kernel;
  size_t imgSize = tmpimg->ncols * tmpimg->nrows * sizeof(float);
  size_t kSize = vert_kernel.width * sizeof(float);

  cudaMalloc(&d_in, imgSize);
  cudaMalloc(&d_out, imgSize);
  cudaMalloc(&d_kernel, kSize);

  cudaMemcpy(d_in, tmpimg->data, imgSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, vert_kernel.data, kSize, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((tmpimg->ncols + block.x - 1) / block.x,
            (tmpimg->nrows + block.y - 1) / block.y);

  _convolveImageVert<<<grid, block>>>(d_in, tmpimg->ncols, tmpimg->nrows,
                                      d_kernel, vert_kernel.width, d_out);

  /* timing for vertical convolution */
  {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);
    _convolveImageVert<<<grid, block>>>(d_in, tmpimg->ncols, tmpimg->nrows,
                                        d_kernel, vert_kernel.width, d_out);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    float msv = 0.0f;
    cudaEventElapsedTime(&msv, t0, t1);
    CHECK_CUDA("_convolveImageVert");
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    FILE *fv = fopen("profiling/gpu_timing.txt", "a");
    if (fv) { fprintf(fv, "convolve_vert %g\n", (double)msv); fclose(fv); }
  }
  cudaMemcpy(imgout->data, d_out, imgSize, cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_kernel);

  _KLTFreeFloatImage(tmpimg);
}


/*********************************************************************
 * GPU _KLTComputeGradientsKernel + host launcher
 *********************************************************************/
__global__
void _KLTComputeGradientsKernel(
    const float* img, int ncols, int nrows,
    const float* gauss, int gauss_width,
    const float* gaussderiv, int gaussderiv_width,
    float* gradx, float* grady)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= ncols || y >= nrows) return;

  int idx = y * ncols + x;
  int radius = gauss_width / 2;

  float gx = 0.0f, gy = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    int yy = min(max(y + k, 0), nrows - 1);
    int xx = min(max(x + k, 0), ncols - 1);
    gx += img[y * ncols + xx] * gaussderiv[k + radius];
    gy += img[yy * ncols + x] * gaussderiv[k + radius];
  }

  gradx[idx] = gx;
  grady[idx] = gy;
}

void _KLTComputeGradients(
  _KLT_FloatImage img, float sigma,
  _KLT_FloatImage gradx, _KLT_FloatImage grady)
{
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  int ncols = img->ncols, nrows = img->nrows;

  /* Debug: print kernel info to ensure kernels were computed correctly */
  fprintf(stderr, "[DEBUG] _KLTComputeGradients: sigma=%g gauss_w=%d gaussderiv_w=%d\n",
          sigma, gauss_kernel.width, gaussderiv_kernel.width);
  for (int ki = 0; ki < gauss_kernel.width && ki < 16; ++ki)
    fprintf(stderr, "[DEBUG] gauss[%d]=%g\n", ki, gauss_kernel.data[ki]);
  for (int ki = 0; ki < gaussderiv_kernel.width && ki < 16; ++ki)
    fprintf(stderr, "[DEBUG] gaussderiv[%d]=%g\n", ki, gaussderiv_kernel.data[ki]);

  /* Some convolution kernels need to be reversed for horizontal vs vertical
    application (convolution vs correlation). Create a reversed derivative
    kernel for the horizontal pass and use it to compute gradx. */
  ConvolutionKernel rev_gaussderiv = gaussderiv_kernel;
  for (int i = 0; i < gaussderiv_kernel.width; ++i)
   rev_gaussderiv.data[i] = gaussderiv_kernel.data[gaussderiv_kernel.width - 1 - i];

  /* time the two separable convolutions used to build gradients */
  {
    cudaEvent_t gstart, gstop;
    cudaEventCreate(&gstart);
    cudaEventCreate(&gstop);
    cudaEventRecord(gstart, 0);
    _convolveSeparate(img, rev_gaussderiv, gauss_kernel, gradx);
    _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
    cudaEventRecord(gstop, 0);
    cudaEventSynchronize(gstop);
    float gms = 0.0f;
    cudaEventElapsedTime(&gms, gstart, gstop);
    cudaEventDestroy(gstart);
    cudaEventDestroy(gstop);
    FILE *fg = fopen("profiling/gpu_timing.txt", "a");
    if (fg) { fprintf(fg, "compute_gradients %g\n", (double)gms); fclose(fg); }
  }

  /* compute compact debug summaries (sum of absolute values) */
  {
    double sum_abs_x = 0.0, sum_abs_y = 0.0;
    int npix = ncols * nrows;
    for (int i = 0; i < npix; ++i) {
      sum_abs_x += fabs((double)gradx->data[i]);
      sum_abs_y += fabs((double)grady->data[i]);
    }
#ifdef DEBUG
    fprintf(stderr, "[DEBUG] gradx summary: sum_abs=%.6g\n", sum_abs_x);
    fprintf(stderr, "[DEBUG] grady summary: sum_abs=%.6g\n", sum_abs_y);
#endif
  }

  /* Quick checksum to ensure GPU produced non-trivial gradients. If not,
     fall back to a reliable host-side separable convolution to preserve
     correctness. This avoids pulling in extra dependencies. */
  /* opt-in fallback: check environment variable KLT_HOST_FALLBACK */
  char *env = getenv("KLT_HOST_FALLBACK");
  if (env && env[0] != '\0') {
    double abs_sum = 0.0;
    int npix = ncols * nrows;
    for (int i = 0; i < npix; ++i) abs_sum += fabs((double)gradx->data[i]) + fabs((double)grady->data[i]);
    if (abs_sum < 1e-6) {
      fprintf(stderr, "[WARN] GPU gradients look empty (abs_sum=%g). Falling back to host convolution.\n", abs_sum);
      /* compute gradx = derivative x, smooth y */
      host_convolve_separable(img, rev_gaussderiv, gauss_kernel, gradx);
      /* compute grady = smooth x, derivative y */
      host_convolve_separable(img, gauss_kernel, gaussderiv_kernel, grady);
#ifdef DEBUG
      /* compact fallback summaries: compute actual sums after fallback */
      {
        double fb_sum_x = 0.0, fb_sum_y = 0.0;
        for (int i = 0; i < npix; ++i) {
          fb_sum_x += fabs((double)gradx->data[i]);
          fb_sum_y += fabs((double)grady->data[i]);
        }
        fprintf(stderr, "[DEBUG] gradx.fallback sum_abs=%.6g\n", fb_sum_x);
        fprintf(stderr, "[DEBUG] grady.fallback sum_abs=%.6g\n", fb_sum_y);
      }
#endif
    }
  }
}


/*********************************************************************
 * _KLTComputeSmoothedImage
 *********************************************************************/
void _KLTComputeSmoothedImage(
  _KLT_FloatImage img, float sigma,
  _KLT_FloatImage smooth)
{
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}

#ifdef __cplusplus
}
#endif
