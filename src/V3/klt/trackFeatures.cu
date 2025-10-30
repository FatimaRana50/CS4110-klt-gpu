/*********************************************************************
 * trackFeatures.c  (GPU copies cached per image/level; no per-feature H2D)
 *
 * - Keeps all function names and tracking logic unchanged
 * - Eliminates repeated cudaMalloc/cudaMemcpy per feature window
 *   by caching device copies of pyramid and gradient images
 * - Reuses small device buffers for window outputs
 * - Uses a single CUDA stream (g_stream) to avoid global device syncs
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>      /* fabs() */
#include <stdlib.h>    /* malloc(), free() */
#include <string.h>    /* memset() */
#include <stdio.h>     /* fflush() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve_gpu.cuh"  /* for computing pyramid and CUDA types */
#include "klt.h"
#include "klt_util.h"  /* _KLT_FloatImage */
#include "pyramid.h"
#include "trackFeatures.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* --- small helper macros --- */
#define KLT_CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    return; \
  } \
} while(0)

#define KLT_CUDA_CHECK_RETURN(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    return -1; \
  } \
} while(0)

#ifdef USE_CUDA
/* ======================= GPU helpers (device + kernels) ==================== */

__device__ __forceinline__ float dev_bilinear(const float *img, int ncols, int nrows, float x, float y) {
  int xt = (int)floorf(x);
  int yt = (int)floorf(y);
  if (xt < 0) xt = 0;
  if (yt < 0) yt = 0;
  if (xt >= ncols-1) xt = ncols-2;
  if (yt >= nrows-1) yt = nrows-2;
  float ax = x - xt;
  float ay = y - yt;
  const float *p = img + yt * ncols + xt;
  float v00 = p[0];
  float v10 = p[1];
  float v01 = p[ncols];
  float v11 = p[ncols + 1];
  return (1-ax)*(1-ay)*v00 + ax*(1-ay)*v10 + (1-ax)*ay*v01 + ax*ay*v11;
}

/* Windowed intensity difference: reads from device-resident images */
__global__ void computeIntensityDifferenceOptimized(
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    int ncols, int nrows,
    float x1, float y1,
    float x2, float y2,
    int width, int height,
    float* __restrict__ imgdiff)
{
  extern __shared__ float shared_mem[];
  float* shared_diff = shared_mem;

  const int hw = width  / 2;
  const int hh = height / 2;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  if (tx >= width || ty >= height) return;

  float rel_x = tx - hw;
  float rel_y = ty - hh;

  float g1 = dev_bilinear(img1, ncols, nrows, x1 + rel_x, y1 + rel_y);
  float g2 = dev_bilinear(img2, ncols, nrows, x2 + rel_x, y2 + rel_y);
  float diff = g1 - g2;

  int idx = ty * width + tx;
  shared_diff[idx] = diff;
  __syncthreads();

  imgdiff[idx] = shared_diff[idx];
}

/* Gradient sum into window buffers, reading from device-resident grad images */
__global__ void gradientSumKernelShared(
    const float* __restrict__ gradx1, const float* __restrict__ grady1,
    const float* __restrict__ gradx2, const float* __restrict__ grady2,
    float* __restrict__ gradx_out, float* __restrict__ grady_out,
    int width, int height,
    float x1, float y1, float x2, float y2,
    int ncols, int nrows)
{
  extern __shared__ float sdata[];
  float* s_gradx = sdata;                    /* width*height */
  float* s_grady = sdata + width*height;     /* width*height */

  const int hw = width  / 2;
  const int hh = height / 2;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  if (tx >= width || ty >= height) return;

  float rel_x = tx - hw;
  float rel_y = ty - hh;

  float gx1 = dev_bilinear(gradx1, ncols, nrows, x1 + rel_x, y1 + rel_y);
  float gx2 = dev_bilinear(gradx2, ncols, nrows, x2 + rel_x, y2 + rel_y);
  float gy1 = dev_bilinear(grady1, ncols, nrows, x1 + rel_x, y1 + rel_y);
  float gy2 = dev_bilinear(grady2, ncols, nrows, x2 + rel_x, y2 + rel_y);

  int idx = ty * width + tx;
  s_gradx[idx] = gx1 + gx2;
  s_grady[idx] = gy1 + gy2;

  __syncthreads();

  gradx_out[idx] = s_gradx[idx];
  grady_out[idx] = s_grady[idx];
}

/* ======================= Per-image device cache ============================ */

/* We cache device copies keyed by the host pointer address.
   For each unique host image pointer (e.g., pyramid1->img[r]->data),
   we allocate once and reuse across *all* features in this frame. */

typedef struct {
  const float* h_ptr;   /* host pointer identity */
  float* d_ptr;         /* device buffer */
  int ncols, nrows;
  size_t bytes;
} DevImageCache;

static DevImageCache g_img1_cache = {0};
static DevImageCache g_img2_cache = {0};
static DevImageCache g_gx1_cache  = {0};
static DevImageCache g_gy1_cache  = {0};
static DevImageCache g_gx2_cache  = {0};
static DevImageCache g_gy2_cache  = {0};

/* Small window buffers (reused) */
static float* g_d_imgdiff   = NULL;
static float* g_d_gradx_win = NULL;
static float* g_d_grady_win = NULL;
static int    g_win_w = 0, g_win_h = 0;

/* Single stream for all kernels here */
static cudaStream_t g_stream = NULL;

/* Forward declarations */
static void tf_init_stream(void);
static void tf_free_stream(void);
static void tf_clear_all_caches(void);

static void ensure_window_buffers(int width, int height)
{
  size_t need = (size_t)width * (size_t)height * sizeof(float);
  if (g_win_w == width && g_win_h == height && g_d_imgdiff && g_d_gradx_win && g_d_grady_win)
    return;

  /* Realloc */
  if (g_d_imgdiff)   cudaFree(g_d_imgdiff);
  if (g_d_gradx_win) cudaFree(g_d_gradx_win);
  if (g_d_grady_win) cudaFree(g_d_grady_win);

  KLT_CUDA_CHECK(cudaMalloc((void**)&g_d_imgdiff, need));
  KLT_CUDA_CHECK(cudaMalloc((void**)&g_d_gradx_win, need));
  KLT_CUDA_CHECK(cudaMalloc((void**)&g_d_grady_win, need));
  g_win_w = width;
  g_win_h = height;
}

static inline void ensure_dev_image_cached(DevImageCache* c, const float* h_ptr, int ncols, int nrows)
{
  if (c->h_ptr == h_ptr && c->d_ptr && c->ncols == ncols && c->nrows == nrows) return;

  /* Different image (or first time): free old and upload */
  if (c->d_ptr) cudaFree(c->d_ptr);
  c->h_ptr = NULL;
  c->d_ptr = NULL;
  c->ncols = ncols;
  c->nrows = nrows;
  c->bytes = (size_t)ncols * (size_t)nrows * sizeof(float);

  KLT_CUDA_CHECK(cudaMalloc((void**)&c->d_ptr, c->bytes));
  KLT_CUDA_CHECK(cudaMemcpyAsync(c->d_ptr, h_ptr, c->bytes, cudaMemcpyHostToDevice, g_stream));
  KLT_CUDA_CHECK(cudaStreamSynchronize(g_stream)); /* keep semantics deterministic */
  c->h_ptr = h_ptr;
}

static void tf_init_stream(void)
{
  if (!g_stream) {
    cudaError_t e = cudaStreamCreate(&g_stream);
    if (e != cudaSuccess) {
      fprintf(stderr, "CUDA stream create failed: %s\n", cudaGetErrorString(e));
      g_stream = NULL; /* we’ll fallback to default stream if needed */
    }
  }
}

static void tf_free_stream(void)
{
  if (g_stream) { cudaStreamDestroy(g_stream); g_stream = NULL; }
}

static void tf_clear_cache(DevImageCache* c)
{
  if (c->d_ptr) { cudaFree(c->d_ptr); c->d_ptr = NULL; }
  c->h_ptr = NULL; c->ncols = c->nrows = 0; c->bytes = 0;
}

static void tf_clear_all_caches(void)
{
  tf_clear_cache(&g_img1_cache);
  tf_clear_cache(&g_img2_cache);
  tf_clear_cache(&g_gx1_cache);
  tf_clear_cache(&g_gy1_cache);
  tf_clear_cache(&g_gx2_cache);
  tf_clear_cache(&g_gy2_cache);

  if (g_d_imgdiff)   { cudaFree(g_d_imgdiff);   g_d_imgdiff   = NULL; }
  if (g_d_gradx_win) { cudaFree(g_d_gradx_win); g_d_gradx_win = NULL; }
  if (g_d_grady_win) { cudaFree(g_d_grady_win); g_d_grady_win = NULL; }
  g_win_w = g_win_h = 0;
}

#endif /* USE_CUDA */

/* ======================= Original CPU helpers (unchanged) ================== */

typedef float *_FloatWindow;

static float _interpolate(float x, float y, _KLT_FloatImage img)
{
  int xt = (int) x;
  int yt = (int) y;
  float ax = x - xt;
  float ay = y - yt;
  float *ptr = img->data + (img->ncols*yt) + xt;

#ifndef _DNDEBUG
  if (xt<0 || yt<0 || xt>=img->ncols-1 || yt>=img->nrows-1) {
    fprintf(stderr, "(xt,yt)=(%d,%d)  imgsize=(%d,%d)\n"
            "(x,y)=(%f,%f)  (ax,ay)=(%f,%f)\n",
            xt, yt, img->ncols, img->nrows, x, y, ax, ay);
    fflush(stderr);
  }
#endif

  assert (xt >= 0 && yt >= 0 && xt <= img->ncols - 2 && yt <= img->nrows - 2);

  return ( (1-ax) * (1-ay) * *ptr +
           ax   * (1-ay) * *(ptr+1) +
           (1-ax) *   ay   * *(ptr+(img->ncols)) +
           ax   *   ay   * *(ptr+(img->ncols)+1) );
}

static void _computeIntensityDifference(
  _KLT_FloatImage img1, _KLT_FloatImage img2,
  float x1, float y1, float x2, float y2,
  int width, int height, _FloatWindow imgdiff)
{
  register int hw = width/2, hh = height/2;
  float g1, g2;
  register int i, j;

  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1);
      g2 = _interpolate(x2+i, y2+j, img2);
      *imgdiff++ = g1 - g2;
    }
}

static void _computeGradientSum(
  _KLT_FloatImage gradx1, _KLT_FloatImage grady1,
  _KLT_FloatImage gradx2, _KLT_FloatImage grady2,
  float x1, float y1, float x2, float y2,
  int width, int height, _FloatWindow gradx, _FloatWindow grady)
{
  register int hw = width/2, hh = height/2;
  float g1, g2;
  register int i, j;

  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, gradx1);
      g2 = _interpolate(x2+i, y2+j, gradx2);
      *gradx++ = g1 + g2;
      g1 = _interpolate(x1+i, y1+j, grady1);
      g2 = _interpolate(x2+i, y2+j, grady2);
      *grady++ = g1 + g2;
    }
}

static void _computeIntensityDifferenceLightingInsensitive(
  _KLT_FloatImage img1, _KLT_FloatImage img2,
  float x1, float y1, float x2, float y2,
  int width, int height, _FloatWindow imgdiff)
{
  register int hw = width/2, hh = height/2;
  float g1, g2, sum1_squared = 0, sum2_squared = 0;
  register int i, j;

  float sum1 = 0, sum2 = 0;
  float mean1, mean2, alpha, belta;

  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1);
      g2 = _interpolate(x2+i, y2+j, img2);
      sum1 += g1;    sum2 += g2;
      sum1_squared += g1*g1;
      sum2_squared += g2*g2;
    }
  mean1=sum1_squared/(width*height);
  mean2=sum2_squared/(width*height);
  alpha = (float) sqrt(mean1/mean2);
  mean1=sum1/(width*height);
  mean2=sum2/(width*height);
  belta = mean1-alpha*mean2;

  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1);
      g2 = _interpolate(x2+i, y2+j, img2);
      *imgdiff++ = g1- g2*alpha-belta;
    }
}

static void _computeGradientSumLightingInsensitive(
  _KLT_FloatImage gradx1, _KLT_FloatImage grady1,
  _KLT_FloatImage gradx2, _KLT_FloatImage grady2,
  _KLT_FloatImage img1, _KLT_FloatImage img2,
  float x1, float y1, float x2, float y2,
  int width, int height, _FloatWindow gradx, _FloatWindow grady)
{
  register int hw = width/2, hh = height/2;
  float g1, g2, sum1_squared = 0, sum2_squared = 0;
  register int i, j;

  float mean1, mean2, alpha;
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1);
      g2 = _interpolate(x2+i, y2+j, img2);
      sum1_squared += g1;    sum2_squared += g2;
    }
  mean1 = sum1_squared/(width*height);
  mean2 = sum2_squared/(width*height);
  alpha = (float) sqrt(mean1/mean2);

  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, gradx1);
      g2 = _interpolate(x2+i, y2+j, gradx2);
      *gradx++ = g1 + g2*alpha;
      g1 = _interpolate(x1+i, y1+j, grady1);
      g2 = _interpolate(x2+i, y2+j, grady2);
      *grady++ = g1+ g2*alpha;
    }
}

static void _compute2by2GradientMatrix(
  _FloatWindow gradx, _FloatWindow grady,
  int width, int height, float *gxx, float *gxy, float *gyy)
{
  register float gx, gy;
  register int i;

  *gxx = 0.0f;  *gxy = 0.0f;  *gyy = 0.0f;
  for (i = 0 ; i < width * height ; i++)  {
    gx = *gradx++;
    gy = *grady++;
    *gxx += gx*gx;
    *gxy += gx*gy;
    *gyy += gy*gy;
  }
}

static void _compute2by1ErrorVector(
  _FloatWindow imgdiff, _FloatWindow gradx, _FloatWindow grady,
  int width, int height, float step_factor, float *ex, float *ey)
{
  register float diff;
  register int i;

  *ex = 0;  *ey = 0;
  for (i = 0 ; i < width * height ; i++)  {
    diff = *imgdiff++;
    *ex += diff * (*gradx++);
    *ey += diff * (*grady++);
  }
  *ex *= step_factor;
  *ey *= step_factor;
}

static int _solveEquation(
  float gxx, float gxy, float gyy, float ex, float ey, float small, float *dx, float *dy)
{
  float det = gxx*gyy - gxy*gxy;
  if (det < small)  return KLT_SMALL_DET;
  *dx = (gyy*ex - gxy*ey)/det;
  *dy = (gxx*ey - gxy*ex)/det;
  return KLT_TRACKED;
}

static _FloatWindow _allocateFloatWindow(int width, int height)
{
  _FloatWindow fw = (_FloatWindow) malloc(width*height*sizeof(float));
  if (fw == NULL)  KLTError("(_allocateFloatWindow) Out of memory.");
  return fw;
}

static float _sumAbsFloatWindow(_FloatWindow fw, int width, int height)
{
  float sum = 0.0f;
  int w;
  for ( ; height > 0 ; height--)
    for (w=0 ; w < width ; w++)
      sum += (float) fabs(*fw++);
  return sum;
}

/* ======================== GPU wrappers (same names) ======================== */

#ifdef USE_CUDA
/* This function name stays the same, but now reuses cached device copies.
   It only copies the tiny window result back to host. */
void computeIntensityDifferenceGPUWrapper(
    _KLT_FloatImage img1, _KLT_FloatImage img2,
    float x1, float y1, float x2, float y2,
    int width, int height, float* imgdiff)
{
  if (width <= 0 || height <= 0) return;
  if (width > 1024 || height > 1024) return; /* sanity */

  tf_init_stream();
  ensure_window_buffers(width, height);

  /* Stage/refresh device copies for these host images (pyramid level pointers) */
  ensure_dev_image_cached(&g_img1_cache, img1->data, img1->ncols, img1->nrows);
  ensure_dev_image_cached(&g_img2_cache, img2->data, img2->ncols, img2->nrows);

  dim3 block(width, height, 1);
  dim3 grid(1, 1, 1);
  size_t shmem = (size_t)width * (size_t)height * sizeof(float);

  computeIntensityDifferenceOptimized<<<grid, block, shmem, g_stream>>>(
      g_img1_cache.d_ptr, g_img2_cache.d_ptr,
      img1->ncols, img1->nrows,
      x1, y1, x2, y2,
      width, height,
      g_d_imgdiff);

  cudaError_t ke = cudaGetLastError();
  if (ke != cudaSuccess) {
    fprintf(stderr, "computeIntensityDifferenceOptimized launch failed: %s\n", cudaGetErrorString(ke));
    cudaDeviceSynchronize();
    return;
  }

  KLT_CUDA_CHECK(cudaMemcpyAsync(imgdiff, g_d_imgdiff,
                                 (size_t)width * (size_t)height * sizeof(float),
                                 cudaMemcpyDeviceToHost, g_stream));
  KLT_CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

/* Same function name, now reuses cached device gradient images and small buffers */
void computeGradientSumGPUShared(
    float* gradx1, float* grady1,
    float* gradx2, float* grady2,
    int ncols, int nrows,
    float x1, float y1, float x2, float y2,
    int width, int height, float* gradx, float* grady)
{
  if (!gradx1 || !grady1 || !gradx2 || !grady2 || !gradx || !grady) return;
  if (width <= 0 || height <= 0) return;

  tf_init_stream();
  ensure_window_buffers(width, height);

  /* Cache gradient images (host pointers here are from pyramids) */
  ensure_dev_image_cached(&g_gx1_cache, gradx1, ncols, nrows);
  ensure_dev_image_cached(&g_gy1_cache, grady1, ncols, nrows);
  ensure_dev_image_cached(&g_gx2_cache, gradx2, ncols, nrows);
  ensure_dev_image_cached(&g_gy2_cache, grady2, ncols, nrows);

  dim3 block(width, height, 1);
  dim3 grid(1, 1, 1);
  size_t shmem = 2ULL * (size_t)width * (size_t)height * sizeof(float); /* gradx+grady */

  gradientSumKernelShared<<<grid, block, shmem, g_stream>>>(
      g_gx1_cache.d_ptr, g_gy1_cache.d_ptr,
      g_gx2_cache.d_ptr, g_gy2_cache.d_ptr,
      g_d_gradx_win, g_d_grady_win,
      width, height,
      x1, y1, x2, y2,
      ncols, nrows);

  cudaError_t ke = cudaGetLastError();
  if (ke != cudaSuccess) {
    fprintf(stderr, "gradientSumKernelShared launch failed: %s\n", cudaGetErrorString(ke));
    cudaDeviceSynchronize();
    return;
  }

  size_t bytes = (size_t)width * (size_t)height * sizeof(float);
  KLT_CUDA_CHECK(cudaMemcpyAsync(gradx, g_d_gradx_win, bytes, cudaMemcpyDeviceToHost, g_stream));
  KLT_CUDA_CHECK(cudaMemcpyAsync(grady, g_d_grady_win, bytes, cudaMemcpyDeviceToHost, g_stream));
  KLT_CUDA_CHECK(cudaStreamSynchronize(g_stream));
}
#else
/* CPU fallbacks remain (names unchanged) */
void computeIntensityDifferenceGPUWrapper(
    _KLT_FloatImage img1, _KLT_FloatImage img2,
    float x1, float y1, float x2, float y2,
    int width, int height, float* imgdiff)
{
  _computeIntensityDifference(img1, img2, x1, y1, x2, y2, width, height, imgdiff);
}

void computeGradientSumGPUShared(
    float* gradx1, float* grady1,
    float* gradx2, float* grady2,
    int ncols, int nrows,
    float x1, float y1, float x2, float y2,
    int width, int height, float* gradx, float* grady)
{
  _KLT_FloatImage gx1 = _KLTCreateFloatImage(ncols, nrows);
  _KLT_FloatImage gy1 = _KLTCreateFloatImage(ncols, nrows);
  _KLT_FloatImage gx2 = _KLTCreateFloatImage(ncols, nrows);
  _KLT_FloatImage gy2 = _KLTCreateFloatImage(ncols, nrows);
  memcpy(gx1->data, gradx1, ncols*nrows*sizeof(float));
  memcpy(gy1->data, grady1, ncols*nrows*sizeof(float));
  memcpy(gx2->data, gradx2, ncols*nrows*sizeof(float));
  memcpy(gy2->data, grady2, ncols*nrows*sizeof(float));
  _computeGradientSum(gx1, gy1, gx2, gy2, x1, y1, x2, y2, width, height, gradx, grady);
  _KLTFreeFloatImage(gx1); _KLTFreeFloatImage(gy1);
  _KLTFreeFloatImage(gx2); _KLTFreeFloatImage(gy2);
}
#endif

/* ========================== Tracker core (unchanged math) ================== */

static int _trackFeature(
  float x1, float y1, float *x2, float *y2,
  _KLT_FloatImage img1,
  _KLT_FloatImage gradx1, _KLT_FloatImage grady1,
  _KLT_FloatImage img2,
  _KLT_FloatImage gradx2, _KLT_FloatImage grady2,
  int width, int height, float step_factor,
  int max_iterations, float small, float th,
  float max_residue, int lighting_insensitive)
{
  _FloatWindow imgdiff, gradx, grady;
  float gxx, gxy, gyy, ex, ey, dx, dy;
  int iteration = 0;
  int status;
  int hw = width/2;
  int hh = height/2;
  int nc = img1->ncols;
  int nr = img1->nrows;
  float one_plus_eps = 1.001f;

  imgdiff = _allocateFloatWindow(width, height);
  gradx   = _allocateFloatWindow(width, height);
  grady   = _allocateFloatWindow(width, height);

  do  {
    if (  x1-hw < 0.0f || nc-( x1+hw) < one_plus_eps ||
         *x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps ||
          y1-hh < 0.0f || nr-( y1+hh) < one_plus_eps ||
         *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps) {
      status = KLT_OOB;
      break;
    }

    if (lighting_insensitive) {
      _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2,
                                                     width, height, imgdiff);
      _computeGradientSumLightingInsensitive(gradx1, grady1, gradx2, grady2,
                                             img1, img2, x1, y1, *x2, *y2,
                                             width, height, gradx, grady);
    } else {
      /* GPU-accelerated (now cached device images; no per-feature H2D) */
      computeIntensityDifferenceGPUWrapper(
          img1, img2, x1, y1, *x2, *y2, width, height, imgdiff);

      computeGradientSumGPUShared(
          gradx1->data, grady1->data,
          gradx2->data, grady2->data,
          img1->ncols, img1->nrows,
          x1, y1, *x2, *y2,
          width, height, gradx, grady);
    }

    _compute2by2GradientMatrix(gradx, grady, width, height, &gxx, &gxy, &gyy);
    _compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor, &ex, &ey);

    status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
    if (status == KLT_SMALL_DET)  break;

    *x2 += dx;
    *y2 += dy;
    iteration++;

  }  while ((fabs(dx)>=th || fabs(dy)>=th) && iteration < max_iterations);

  if (*x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps ||
      *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps)
    status = KLT_OOB;

  if (status == KLT_TRACKED)  {
    if (lighting_insensitive)
      _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2,
                                                     width, height, imgdiff);
    else
      computeIntensityDifferenceGPUWrapper(
          img1, img2, x1, y1, *x2, *y2, width, height, imgdiff);

    if (_sumAbsFloatWindow(imgdiff, width, height)/(width*height) > max_residue)
      status = KLT_LARGE_RESIDUE;
  }

  free(imgdiff);
  free(gradx);
  free(grady);

  if (status == KLT_SMALL_DET)  return KLT_SMALL_DET;
  else if (status == KLT_OOB)   return KLT_OOB;
  else if (status == KLT_LARGE_RESIDUE)  return KLT_LARGE_RESIDUE;
  else if (iteration >= max_iterations)  return KLT_MAX_ITERATIONS;
  else  return KLT_TRACKED;
}

static KLT_BOOL _outOfBounds(float x, float y, int ncols, int nrows, int borderx, int bordery)
{
  return (x < borderx || x > ncols-1-borderx ||
          y < bordery || y > nrows-1-bordery );
}

/* ============================ Public entry ================================= */

#ifdef USE_CUDA
extern void _KLTComputeGradients(_KLT_FloatImage img, float sigma, _KLT_FloatImage gradx, _KLT_FloatImage grady);
extern void _KLTComputeSmoothedImage(_KLT_FloatImage img, float sigma, _KLT_FloatImage smooth);
extern void _KLTToFloatImage(KLT_PixelType *img, int ncols, int nrows, _KLT_FloatImage floatimg);
extern void _KLTGetKernelWidths(float sigma, int *gauss_width, int *gaussderiv_width);
#endif

extern int KLT_verbose;

void KLTTrackFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img1,
  KLT_PixelType *img2,
  int ncols,
  int nrows,
  KLT_FeatureList featurelist)
{
  _KLT_FloatImage tmpimg, floatimg1, floatimg2;
  _KLT_Pyramid pyramid1, pyramid1_gradx, pyramid1_grady,
    pyramid2, pyramid2_gradx, pyramid2_grady;
  float subsampling = (float) tc->subsampling;
  float xloc, yloc, xlocout, ylocout;
  int val;
  int indx, r;
  KLT_BOOL floatimg1_created = FALSE;
  int i;

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Tracking %d features in a %d by %d image...  ",
      KLTCountRemainingFeatures(featurelist), ncols, nrows);
    fflush(stderr);
  }

  if (tc->window_width % 2 != 1) {
    tc->window_width = tc->window_width+1;
    KLTWarning("Tracking context's window width must be odd.  "
      "Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height % 2 != 1) {
    tc->window_height = tc->window_height+1;
    KLTWarning("Tracking context's window height must be odd.  "
      "Changing to %d.\n", tc->window_height);
  }
  if (tc->window_width < 3) {
    tc->window_width = 3;
    KLTWarning("Tracking context's window width must be at least three.  \n"
      "Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height < 3) {
    tc->window_height = 3;
    KLTWarning("Tracking context's window height must be at least three.  \n"
      "Changing to %d.\n", tc->window_height);
  }

  tmpimg = _KLTCreateFloatImage(ncols, nrows);

  if (tc->sequentialMode && tc->pyramid_last != NULL)  {
    pyramid1 = (_KLT_Pyramid) tc->pyramid_last;
    pyramid1_gradx = (_KLT_Pyramid) tc->pyramid_last_gradx;
    pyramid1_grady = (_KLT_Pyramid) tc->pyramid_last_grady;
    if (pyramid1->ncols[0] != ncols || pyramid1->nrows[0] != nrows)
      KLTError("(KLTTrackFeatures) Size of incoming image (%d by %d) "
      "is different from size of previous image (%d by %d)\n",
      ncols, nrows, pyramid1->ncols[0], pyramid1->nrows[0]);
    assert(pyramid1_gradx != NULL);
    assert(pyramid1_grady != NULL);
  } else  {
    floatimg1_created = TRUE;
    floatimg1 = _KLTCreateFloatImage(ncols, nrows);
    _KLTToFloatImage(img1, ncols, nrows, tmpimg);
    _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg1);
    pyramid1 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
    _KLTComputePyramid(floatimg1, pyramid1, tc->pyramid_sigma_fact);
    pyramid1_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
    pyramid1_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
    for (i = 0 ; i < tc->nPyramidLevels ; i++)
      _KLTComputeGradients(pyramid1->img[i], tc->grad_sigma,
                           pyramid1_gradx->img[i], pyramid1_grady->img[i]);
  }

  floatimg2 = _KLTCreateFloatImage(ncols, nrows);
  _KLTToFloatImage(img2, ncols, nrows, tmpimg);
  _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg2);
  pyramid2 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
  _KLTComputePyramid(floatimg2, pyramid2, tc->pyramid_sigma_fact);
  pyramid2_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
  pyramid2_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
  for (i = 0 ; i < tc->nPyramidLevels ; i++)
    _KLTComputeGradients(pyramid2->img[i], tc->grad_sigma,
                         pyramid2_gradx->img[i], pyramid2_grady->img[i]);

  if (tc->writeInternalImages)  {
    char fname[80];
    for (i = 0 ; i < tc->nPyramidLevels ; i++)  {
      sprintf(fname, "kltimg_tf_i%d.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid1->img[i], fname);
      sprintf(fname, "kltimg_tf_i%d_gx.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid1_gradx->img[i], fname);
      sprintf(fname, "kltimg_tf_i%d_gy.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid1_grady->img[i], fname);
      sprintf(fname, "kltimg_tf_j%d.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid2->img[i], fname);
      sprintf(fname, "kltimg_tf_j%d_gx.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid2_gradx->img[i], fname);
      sprintf(fname, "kltimg_tf_j%d_gy.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid2_grady->img[i], fname);
    }
  }

  /* Per-feature tracking (unchanged loop/logic) */
  for (indx = 0 ; indx < featurelist->nFeatures ; indx++)  {
    if (featurelist->feature[indx]->val >= 0)  {
      float xloc, yloc, xlocout, ylocout;
      xloc = featurelist->feature[indx]->x;
      yloc = featurelist->feature[indx]->y;

      for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
        xloc /= subsampling;  yloc /= subsampling;
      }
      xlocout = xloc;  ylocout = yloc;

      for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
        xloc    *= subsampling;  yloc    *= subsampling;
        xlocout *= subsampling;  ylocout *= subsampling;

        val = _trackFeature(xloc, yloc,
                            &xlocout, &ylocout,
                            pyramid1->img[r],
                            pyramid1_gradx->img[r], pyramid1_grady->img[r],
                            pyramid2->img[r],
                            pyramid2_gradx->img[r], pyramid2_grady->img[r],
                            tc->window_width, tc->window_height,
                            tc->step_factor,
                            tc->max_iterations,
                            tc->min_determinant,
                            tc->min_displacement,
                            tc->max_residue,
                            tc->lighting_insensitive);

        if (val==KLT_SMALL_DET || val==KLT_OOB)
          break;
      }

      if (val == KLT_OOB) {
        featurelist->feature[indx]->x   = -1.0f;
        featurelist->feature[indx]->y   = -1.0f;
        featurelist->feature[indx]->val = KLT_OOB;
      } else if (_outOfBounds(xlocout, ylocout, ncols, nrows, tc->borderx, tc->bordery))  {
        featurelist->feature[indx]->x   = -1.0f;
        featurelist->feature[indx]->y   = -1.0f;
        featurelist->feature[indx]->val = KLT_OOB;
      } else if (val == KLT_SMALL_DET)  {
        featurelist->feature[indx]->x   = -1.0f;
        featurelist->feature[indx]->y   = -1.0f;
        featurelist->feature[indx]->val = KLT_SMALL_DET;
      } else if (val == KLT_LARGE_RESIDUE)  {
        featurelist->feature[indx]->x   = -1.0f;
        featurelist->feature[indx]->y   = -1.0f;
        featurelist->feature[indx]->val = KLT_LARGE_RESIDUE;
      } else if (val == KLT_MAX_ITERATIONS)  {
        featurelist->feature[indx]->x   = -1.0f;
        featurelist->feature[indx]->y   = -1.0f;
        featurelist->feature[indx]->val = KLT_MAX_ITERATIONS;
      } else  {
        featurelist->feature[indx]->x = xlocout;
        featurelist->feature[indx]->y = ylocout;
        featurelist->feature[indx]->val = KLT_TRACKED;
      }
    }
  }

  if (tc->sequentialMode)  {
    tc->pyramid_last       = pyramid2;
    tc->pyramid_last_gradx = pyramid2_gradx;
    tc->pyramid_last_grady = pyramid2_grady;
  } else  {
    _KLTFreePyramid(pyramid2);
    _KLTFreePyramid(pyramid2_gradx);
    _KLTFreePyramid(pyramid2_grady);
  }

  _KLTFreeFloatImage(tmpimg);
  if (floatimg1_created)  _KLTFreeFloatImage(floatimg1);
  _KLTFreeFloatImage(floatimg2);
  _KLTFreePyramid(pyramid1);
  _KLTFreePyramid(pyramid1_gradx);
  _KLTFreePyramid(pyramid1_grady);

#ifdef USE_CUDA
  /* Important: free per-frame device caches (images + window buffers) */
  tf_clear_all_caches();
  tf_free_stream();
#endif

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features successfully tracked.\n",
      KLTCountRemainingFeatures(featurelist));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_tf*.pgm'.\n");
    fflush(stderr);
  }
}
