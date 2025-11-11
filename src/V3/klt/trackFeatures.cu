/*********************************************************************
 * trackFeatures_optimized.cu  (ENHANCED - File 2 Optimizations Applied + TIMING)
 *
 * Key optimizations added from File 2:
 * - Batched feature processing with FeatureData structure
 * - Texture memory for hardware-accelerated interpolation
 * - Constant memory for tracking parameters
 * - Pointer swapping for sequential mode pyramid reuse
 * - Persistent feature buffers
 * - All features processed in parallel on GPU
 * 
 * NEW: GPU execution timing from first to last GPU operation
 * 
 * Algorithm and computations UNCHANGED from File 1
 *********************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "base.h"
#include "error.h"
#include "convolve_gpu.cuh"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"
#include "trackFeatures.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* Configuration */
#define MAX_PYRAMID_LEVELS 8
#define MAX_WINDOW_SIZE 64
#define NUM_STREAMS 4

#define KLT_CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    return; \
  } \
} while(0)

#ifdef USE_CUDA

/* ============== OPTIMIZATION 1: Constant Memory (from File 2) ============== */
// Store tracking parameters in constant memory - read by ALL threads but never change
__constant__ int c_window_width;
__constant__ int c_window_height;
__constant__ float c_step_factor;
__constant__ int c_max_iterations;
__constant__ float c_min_determinant;
__constant__ float c_min_displacement;
__constant__ float c_max_residue;

/* ============== OPTIMIZATION 2: Batched Feature Data (from File 2) ========= */
// Structure for batched feature transfers - all features in ONE transfer
typedef struct {
    float x1, y1;      // Reference position in img1
    float x2, y2;      // Tracked position in img2
    int status;        // Tracking status
} FeatureData;

/* ======================= GPU Kernels (UNCHANGED LOGIC) ===================== */

__device__ __forceinline__ float dev_bilinear(const float *img, int ncols, int nrows, 
                                               float x, float y) {
  int xt = (int)floorf(x);
  int yt = (int)floorf(y);
  xt = max(0, min(xt, ncols-2));
  yt = max(0, min(yt, nrows-2));
  
  float ax = x - xt;
  float ay = y - yt;
  const float *p = img + yt * ncols + xt;
  
  float v00 = p[0];
  float v10 = p[1];
  float v01 = p[ncols];
  float v11 = p[ncols + 1];
  
  return (1-ax)*(1-ay)*v00 + ax*(1-ay)*v10 + (1-ax)*ay*v01 + ax*ay*v11;
}

/* ============== OPTIMIZATION 3: Texture Memory (from File 2) =============== */
// Texture-based interpolation for hardware acceleration and better cache
__device__ __forceinline__ float dev_bilinear_tex(
    cudaTextureObject_t tex,
    int ncols, int nrows,
    float x, float y)
{
    int xt = (int)floorf(x);
    int yt = (int)floorf(y);
    
    if (xt < 0 || yt < 0 || xt >= ncols-1 || yt >= nrows-1)
        return 0.0f;
    
    float ax = x - xt;
    float ay = y - yt;
    
    // Compute 1D indices for 2D row-major layout
    int idx00 = yt * ncols + xt;
    int idx01 = idx00 + 1;
    int idx10 = idx00 + ncols;
    int idx11 = idx10 + 1;
    
    // Use texture cache for reads (benefits from spatial locality)
    float v00 = tex1Dfetch<float>(tex, idx00);
    float v01 = tex1Dfetch<float>(tex, idx01);
    float v10 = tex1Dfetch<float>(tex, idx10);
    float v11 = tex1Dfetch<float>(tex, idx11);
    
    // Manual bilinear interpolation (same math as dev_bilinear)
    return (1-ax)*(1-ay)*v00 + ax*(1-ay)*v01 + (1-ax)*ay*v10 + ax*ay*v11;
}

__global__ void computeIntensityDifferenceOptimized(
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    int ncols, int nrows,
    float x1, float y1,
    float x2, float y2,
    int width, int height,
    float* __restrict__ imgdiff)
{
  const int hw = width  / 2;
  const int hh = height / 2;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  if (tx >= width || ty >= height) return;
  
  float rel_x = tx - hw;
  float rel_y = ty - hh;
  
  float g1 = dev_bilinear(img1, ncols, nrows, x1 + rel_x, y1 + rel_y);
  float g2 = dev_bilinear(img2, ncols, nrows, x2 + rel_x, y2 + rel_y);
  
  imgdiff[ty * width + tx] = g1 - g2;
}

__global__ void gradientSumKernelShared(
    const float* __restrict__ gradx1, const float* __restrict__ grady1,
    const float* __restrict__ gradx2, const float* __restrict__ grady2,
    float* __restrict__ gradx_out, float* __restrict__ grady_out,
    int width, int height,
    float x1, float y1, float x2, float y2,
    int ncols, int nrows)
{
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
  gradx_out[idx] = gx1 + gx2;
  grady_out[idx] = gy1 + gy2;
}

/* ========= OPTIMIZATION 4: Batched Tracking Kernel (from File 2) =========== */
// Process ALL features in parallel on GPU - exact same computation as _trackFeature
__global__ void trackFeaturesBatchedKernel(
    cudaTextureObject_t tex_img1, cudaTextureObject_t tex_img2,
    cudaTextureObject_t tex_gradx1, cudaTextureObject_t tex_grady1,
    cudaTextureObject_t tex_gradx2, cudaTextureObject_t tex_grady2,
    FeatureData *features,
    int num_features,
    int ncols, int nrows)
{
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feat_idx >= num_features) return;
    
    // Read from batched structure
    FeatureData feat = features[feat_idx];
    
    // Skip features that have already failed
    if (feat.status < 0) {
        return;
    }
    
    // Each thread tracks one feature - EXACT SAME LOGIC as File 1's _trackFeature
    float x1 = feat.x1;
    float y1 = feat.y1;
    float x2 = feat.x2;
    float y2 = feat.y2;
    
    // Use constant memory parameters (faster than passed parameters)
    int hw = c_window_width / 2;
    int hh = c_window_height / 2;
    float one_plus_eps = 1.001f;
    int iteration = 0;
    int status = KLT_TRACKED;
    float dx, dy;
    
    // Newton-Raphson iteration loop ON GPU (exact same logic as File 1)
    do {
        // Boundary check (exact same as File 1)
        if (x1 - hw < 0.0f || ncols - (x1 + hw) < one_plus_eps ||
            x2 - hw < 0.0f || ncols - (x2 + hw) < one_plus_eps ||
            y1 - hh < 0.0f || nrows - (y1 + hh) < one_plus_eps ||
            y2 - hh < 0.0f || nrows - (y2 + hh) < one_plus_eps) {
            status = KLT_OOB;
            break;
        }
        
        // Compute gradient matrix and error vector ON GPU
        // EXACT SAME COMPUTATION as File 1's CPU version
        float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
        float ex = 0.0f, ey = 0.0f;
        
        for (int j = -hh; j <= hh; j++) {
            for (int i = -hw; i <= hw; i++) {
                // Use texture cache for reads (optimization from File 2)
                float g1 = dev_bilinear_tex(tex_img1, ncols, nrows, x1 + i, y1 + j);
                float g2 = dev_bilinear_tex(tex_img2, ncols, nrows, x2 + i, y2 + j);
                float diff = g1 - g2;
                
                // Sum gradients (exact same as File 1)
                float gx = dev_bilinear_tex(tex_gradx1, ncols, nrows, x1 + i, y1 + j) +
                           dev_bilinear_tex(tex_gradx2, ncols, nrows, x2 + i, y2 + j);
                float gy = dev_bilinear_tex(tex_grady1, ncols, nrows, x1 + i, y1 + j) +
                           dev_bilinear_tex(tex_grady2, ncols, nrows, x2 + i, y2 + j);
                
                gxx += gx * gx;
                gxy += gx * gy;
                gyy += gy * gy;
                ex += diff * gx;
                ey += diff * gy;
            }
        }
        
        ex *= c_step_factor;
        ey *= c_step_factor;
        
        // Solve equation (exact same as File 1)
        float det = gxx * gyy - gxy * gxy;
        if (det < c_min_determinant) {
            status = KLT_SMALL_DET;
            break;
        }
        
        dx = (gyy * ex - gxy * ey) / det;
        dy = (gxx * ey - gxy * ex) / det;
        
        x2 += dx;
        y2 += dy;
        iteration++;
        
    } while ((fabsf(dx) >= c_min_displacement || fabsf(dy) >= c_min_displacement) && 
             iteration < c_max_iterations);
    
    // Final boundary check (exact same as File 1)
    if (x2 - hw < 0.0f || ncols - (x2 + hw) < one_plus_eps ||
        y2 - hh < 0.0f || nrows - (y2 + hh) < one_plus_eps) {
        status = KLT_OOB;
    }
    
    // Check residue if tracked (exact same as File 1)
    if (status == KLT_TRACKED) {
        float sum_abs_diff = 0.0f;
        for (int j = -hh; j <= hh; j++) {
            for (int i = -hw; i <= hw; i++) {
                float g1 = dev_bilinear_tex(tex_img1, ncols, nrows, x1 + i, y1 + j);
                float g2 = dev_bilinear_tex(tex_img2, ncols, nrows, x2 + i, y2 + j);
                sum_abs_diff += fabsf(g1 - g2);
            }
        }
        if (sum_abs_diff / (c_window_width * c_window_height) > c_max_residue) {
            status = KLT_LARGE_RESIDUE;
        }
    }
    
    if (iteration >= c_max_iterations && status == KLT_TRACKED) {
        status = KLT_MAX_ITERATIONS;
    }
    
    // Write results back to batched structure
    features[feat_idx].x2 = x2;
    features[feat_idx].y2 = y2;
    features[feat_idx].status = status;
}

/* ======================= Memory Pool Architecture ========================== */

typedef struct {
  float* d_img;
  float* d_gradx;
  float* d_grady;
  int ncols, nrows;
  size_t capacity;
  int valid;
  cudaEvent_t ready_event;
} DevImageCacheLevel;

typedef struct {
  float* h_img_pinned;
  float* h_gradx_pinned;
  float* h_grady_pinned;
  size_t capacity;
} PinnedStagingBuffer;

static struct {
  DevImageCacheLevel pyramid1_levels[MAX_PYRAMID_LEVELS];
  DevImageCacheLevel pyramid2_levels[MAX_PYRAMID_LEVELS];
  
  PinnedStagingBuffer staging[NUM_STREAMS];
  
  /* Window buffers (device) */
  float* d_imgdiff;
  float* d_gradx_win;
  float* d_grady_win;
  size_t win_capacity;
  
  /* Result buffers (pinned host) */
  float* h_imgdiff_pinned;
  float* h_gradx_pinned;
  float* h_grady_pinned;
  
  /* OPTIMIZATION 5: Persistent feature buffers (from File 2) */
  FeatureData* d_features;
  FeatureData* h_features;
  int max_features;
  
  cudaStream_t streams[NUM_STREAMS];
  cudaEvent_t events[NUM_STREAMS];
  
  /* Timing events for performance measurement */
  cudaEvent_t timing_start;
  cudaEvent_t timing_stop;
  
  int initialized;
  int num_levels;
  int pyramids_on_gpu;  // Flag for pointer swapping optimization
} g_pool = {0};

/* ======================= Memory Pool Functions ============================= */

static void tf_init_memory_pool(int num_pyramid_levels)
{
  if (g_pool.initialized) return;
  
  g_pool.num_levels = num_pyramid_levels;
  
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreateWithFlags(&g_pool.streams[i], cudaStreamNonBlocking);
    cudaEventCreate(&g_pool.events[i]);
  }
  
  /* Create timing events */
  cudaEventCreate(&g_pool.timing_start);
  cudaEventCreate(&g_pool.timing_stop);
  
  size_t max_win_bytes = (size_t)MAX_WINDOW_SIZE * (size_t)MAX_WINDOW_SIZE * sizeof(float);
  
  cudaMalloc((void**)&g_pool.d_imgdiff, max_win_bytes);
  cudaMalloc((void**)&g_pool.d_gradx_win, max_win_bytes);
  cudaMalloc((void**)&g_pool.d_grady_win, max_win_bytes);
  g_pool.win_capacity = max_win_bytes;
  
  cudaMallocHost((void**)&g_pool.h_imgdiff_pinned, max_win_bytes);
  cudaMallocHost((void**)&g_pool.h_gradx_pinned, max_win_bytes);
  cudaMallocHost((void**)&g_pool.h_grady_pinned, max_win_bytes);
  
  for (int i = 0; i < MAX_PYRAMID_LEVELS; i++) {
    cudaEventCreate(&g_pool.pyramid1_levels[i].ready_event);
    cudaEventCreate(&g_pool.pyramid2_levels[i].ready_event);
    g_pool.pyramid1_levels[i].valid = 0;
    g_pool.pyramid2_levels[i].valid = 0;
  }
  
  g_pool.initialized = 1;
  fprintf(stderr, "[TF_POOL] Initialized: %d streams, max_window=%dx%d\n", 
          NUM_STREAMS, MAX_WINDOW_SIZE, MAX_WINDOW_SIZE);
}

/* ====== OPTIMIZATION 6: Pointer Swapping for Sequential Mode (File 2) ====== */
static inline void ensure_pyramid_level_cached(
    DevImageCacheLevel* cache,
    const float* h_img, const float* h_gradx, const float* h_grady,
    int ncols, int nrows,
    int stream_idx)
{
  size_t img_bytes = (size_t)ncols * (size_t)nrows * sizeof(float);
  cudaStream_t stream = g_pool.streams[stream_idx % NUM_STREAMS];
  
  if (cache->capacity < img_bytes || cache->d_img == NULL) {
    if (cache->d_img) {
      cudaFree(cache->d_img);
      cudaFree(cache->d_gradx);
      cudaFree(cache->d_grady);
    }
    
    cudaMalloc((void**)&cache->d_img, img_bytes);
    cudaMalloc((void**)&cache->d_gradx, img_bytes);
    cudaMalloc((void**)&cache->d_grady, img_bytes);
    cache->capacity = img_bytes;
    cache->valid = 0;
  }
  
  cache->ncols = ncols;
  cache->nrows = nrows;
  
  PinnedStagingBuffer* staging = &g_pool.staging[stream_idx % NUM_STREAMS];
  
  if (staging->capacity < img_bytes) {
    if (staging->h_img_pinned) {
      cudaFreeHost(staging->h_img_pinned);
      cudaFreeHost(staging->h_gradx_pinned);
      cudaFreeHost(staging->h_grady_pinned);
    }
    cudaMallocHost((void**)&staging->h_img_pinned, img_bytes);
    cudaMallocHost((void**)&staging->h_gradx_pinned, img_bytes);
    cudaMallocHost((void**)&staging->h_grady_pinned, img_bytes);
    staging->capacity = img_bytes;
  }
  
  memcpy(staging->h_img_pinned, h_img, img_bytes);
  memcpy(staging->h_gradx_pinned, h_gradx, img_bytes);
  memcpy(staging->h_grady_pinned, h_grady, img_bytes);
  
  cudaMemcpyAsync(cache->d_img, staging->h_img_pinned, img_bytes, 
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(cache->d_gradx, staging->h_gradx_pinned, img_bytes, 
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(cache->d_grady, staging->h_grady_pinned, img_bytes, 
                  cudaMemcpyHostToDevice, stream);
  
  cudaEventRecord(cache->ready_event, stream);
  cache->valid = 1;
}

// Upload pyramids with pointer swapping optimization from File 2
static void upload_pyramids_to_gpu(
    _KLT_Pyramid pyramid1, _KLT_Pyramid pyramid1_gradx, _KLT_Pyramid pyramid1_grady,
    _KLT_Pyramid pyramid2, _KLT_Pyramid pyramid2_gradx, _KLT_Pyramid pyramid2_grady,
    int nlevels, int sequential_mode_reuse)
{
    // OPTIMIZATION: In sequential mode, pyramid2 from last frame = pyramid1 of this frame
    int upload_pyramid1 = 1;
    
    if (sequential_mode_reuse && g_pool.pyramids_on_gpu) {
        // Swap pyramid pointers: last frame's pyramid2 becomes this frame's pyramid1
        for (int i = 0; i < nlevels; i++) {
            DevImageCacheLevel temp = g_pool.pyramid1_levels[i];
            g_pool.pyramid1_levels[i] = g_pool.pyramid2_levels[i];
            g_pool.pyramid2_levels[i] = temp;
        }
        upload_pyramid1 = 0;  // Pyramid1 already on GPU!
    }
    
    // Upload pyramids with streaming
    for (int i = 0; i < nlevels; i++) {
        int stream_id = i % NUM_STREAMS;
        
        // Upload pyramid1 only if needed
        if (upload_pyramid1) {
            ensure_pyramid_level_cached(
                &g_pool.pyramid1_levels[i],
                pyramid1->img[i]->data,
                pyramid1_gradx->img[i]->data,
                pyramid1_grady->img[i]->data,
                pyramid1->img[i]->ncols,
                pyramid1->img[i]->nrows,
                stream_id);
        }
        
        // Always upload pyramid2 (current frame)
        ensure_pyramid_level_cached(
            &g_pool.pyramid2_levels[i],
            pyramid2->img[i]->data,
            pyramid2_gradx->img[i]->data,
            pyramid2_grady->img[i]->data,
            pyramid2->img[i]->ncols,
            pyramid2->img[i]->nrows,
            (stream_id + 1) % NUM_STREAMS);
    }
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(g_pool.streams[i]);
    }
    
    g_pool.pyramids_on_gpu = 1;
}

static void tf_invalidate_caches(void)
{
  for (int i = 0; i < MAX_PYRAMID_LEVELS; i++) {
    g_pool.pyramid1_levels[i].valid = 0;
    g_pool.pyramid2_levels[i].valid = 0;
  }
  g_pool.pyramids_on_gpu = 0;
}

void KLTFreeGPUResources(void)
{
  if (!g_pool.initialized) return;
  
  for (int i = 0; i < MAX_PYRAMID_LEVELS; i++) {
    if (g_pool.pyramid1_levels[i].d_img) {
      cudaFree(g_pool.pyramid1_levels[i].d_img);
      cudaFree(g_pool.pyramid1_levels[i].d_gradx);
      cudaFree(g_pool.pyramid1_levels[i].d_grady);
      cudaEventDestroy(g_pool.pyramid1_levels[i].ready_event);
    }
    if (g_pool.pyramid2_levels[i].d_img) {
      cudaFree(g_pool.pyramid2_levels[i].d_img);
      cudaFree(g_pool.pyramid2_levels[i].d_gradx);
      cudaFree(g_pool.pyramid2_levels[i].d_grady);
      cudaEventDestroy(g_pool.pyramid2_levels[i].ready_event);
    }
  }
  
  for (int i = 0; i < NUM_STREAMS; i++) {
    if (g_pool.staging[i].h_img_pinned) {
      cudaFreeHost(g_pool.staging[i].h_img_pinned);
      cudaFreeHost(g_pool.staging[i].h_gradx_pinned);
      cudaFreeHost(g_pool.staging[i].h_grady_pinned);
    }
  }
  
  if (g_pool.d_imgdiff) cudaFree(g_pool.d_imgdiff);
  if (g_pool.d_gradx_win) cudaFree(g_pool.d_gradx_win);
  if (g_pool.d_grady_win) cudaFree(g_pool.d_grady_win);
  
  if (g_pool.h_imgdiff_pinned) cudaFreeHost(g_pool.h_imgdiff_pinned);
  if (g_pool.h_gradx_pinned) cudaFreeHost(g_pool.h_gradx_pinned);
  if (g_pool.h_grady_pinned) cudaFreeHost(g_pool.h_grady_pinned);
  
  // Free persistent feature buffers
  if (g_pool.d_features) cudaFree(g_pool.d_features);
  if (g_pool.h_features) cudaFreeHost(g_pool.h_features);
  
  // Destroy timing events
  if (g_pool.timing_start) cudaEventDestroy(g_pool.timing_start);
  if (g_pool.timing_stop) cudaEventDestroy(g_pool.timing_stop);
  
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(g_pool.streams[i]);
    cudaEventDestroy(g_pool.events[i]);
  }
  
  memset(&g_pool, 0, sizeof(g_pool));
}

/* ======================= GPU Wrapper Functions ============================= */

static void computeIntensityDifferenceGPU(
    DevImageCacheLevel* level1,
    DevImageCacheLevel* level2,
    float x1, float y1, float x2, float y2,
    int width, int height,
    float* h_imgdiff_out,
    int stream_idx)
{
  if (width > MAX_WINDOW_SIZE || height > MAX_WINDOW_SIZE) return;
  
  cudaStream_t stream = g_pool.streams[stream_idx % NUM_STREAMS];
  
  cudaStreamWaitEvent(stream, level1->ready_event, 0);
  cudaStreamWaitEvent(stream, level2->ready_event, 0);
  
  dim3 block(width, height, 1);
  dim3 grid(1, 1, 1);
  
  computeIntensityDifferenceOptimized<<<grid, block, 0, stream>>>(
      level1->d_img, level2->d_img,
      level1->ncols, level1->nrows,
      x1, y1, x2, y2,
      width, height,
      g_pool.d_imgdiff);
  
  size_t bytes = (size_t)width * (size_t)height * sizeof(float);
  cudaMemcpyAsync(h_imgdiff_out, g_pool.d_imgdiff, bytes, 
                  cudaMemcpyDeviceToHost, stream);
  
  cudaStreamSynchronize(stream);
}

static void computeGradientSumGPU(
    DevImageCacheLevel* level1,
    DevImageCacheLevel* level2,
    float x1, float y1, float x2, float y2,
    int width, int height,
    float* h_gradx_out, float* h_grady_out,
    int stream_idx)
{
  if (width > MAX_WINDOW_SIZE || height > MAX_WINDOW_SIZE) return;
  
  cudaStream_t stream = g_pool.streams[stream_idx % NUM_STREAMS];
  
  cudaStreamWaitEvent(stream, level1->ready_event, 0);
  cudaStreamWaitEvent(stream, level2->ready_event, 0);
  
  dim3 block(width, height, 1);
  dim3 grid(1, 1, 1);
  
  gradientSumKernelShared<<<grid, block, 0, stream>>>(
      level1->d_gradx, level1->d_grady,
      level2->d_gradx, level2->d_grady,
      g_pool.d_gradx_win, g_pool.d_grady_win,
      width, height,
      x1, y1, x2, y2,
      level1->ncols, level1->nrows);
  
  size_t bytes = (size_t)width * (size_t)height * sizeof(float);
  cudaMemcpyAsync(h_gradx_out, g_pool.d_gradx_win, bytes, 
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_grady_out, g_pool.d_grady_win, bytes, 
                  cudaMemcpyDeviceToHost, stream);
  
  cudaStreamSynchronize(stream);
}

/* ========== OPTIMIZATION 7: Texture Object Creation (File 2) =============== */
static cudaTextureObject_t createTextureObject(float* d_data, int width, int height) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_data;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = width * height * sizeof(float);
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    return texObj;
}

#endif /* USE_CUDA */

/* ======================= CPU helpers (UNCHANGED) ============================ */

typedef float *_FloatWindow;

static float _interpolate(float x, float y, _KLT_FloatImage img)
{
  int xt = (int) x;
  int yt = (int) y;
  float ax = x - xt;
  float ay = y - yt;
  float *ptr = img->data + (img->ncols*yt) + xt;
  
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
  float gxx, float gxy, float gyy, float ex, float ey, 
  float small, float *dx, float *dy)
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

/* ========================== Tracker core (CORRECTED) ======================= */

static int _trackFeature(
  float x1, float y1, float *x2, float *y2,
  _KLT_FloatImage img1,
  _KLT_FloatImage gradx1, _KLT_FloatImage grady1,
  _KLT_FloatImage img2,
  _KLT_FloatImage gradx2, _KLT_FloatImage grady2,
  int width, int height, float step_factor,
  int max_iterations, float small, float th,
  float max_residue, int lighting_insensitive,
  int pyramid_level)
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
    
#ifdef USE_CUDA
    if (!lighting_insensitive && g_pool.initialized) {
      /* Use GPU for window extraction */
      computeIntensityDifferenceGPU(
          &g_pool.pyramid1_levels[pyramid_level],
          &g_pool.pyramid2_levels[pyramid_level],
          x1, y1, *x2, *y2,
          width, height,
          imgdiff,
          pyramid_level);
      
      computeGradientSumGPU(
          &g_pool.pyramid1_levels[pyramid_level],
          &g_pool.pyramid2_levels[pyramid_level],
          x1, y1, *x2, *y2,
          width, height,
          gradx, grady,
          pyramid_level);
    } else {
      /* CPU fallback */
      _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, width, height, imgdiff);
      _computeGradientSum(gradx1, grady1, gradx2, grady2, x1, y1, *x2, *y2, 
                         width, height, gradx, grady);
    }
#else
    _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, width, height, imgdiff);
    _computeGradientSum(gradx1, grady1, gradx2, grady2, x1, y1, *x2, *y2, 
                       width, height, gradx, grady);
#endif
    
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
#ifdef USE_CUDA
    if (!lighting_insensitive && g_pool.initialized) {
      computeIntensityDifferenceGPU(
          &g_pool.pyramid1_levels[pyramid_level],
          &g_pool.pyramid2_levels[pyramid_level],
          x1, y1, *x2, *y2,
          width, height,
          imgdiff,
          pyramid_level);
    } else {
      _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, width, height, imgdiff);
    }
#else
    _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, width, height, imgdiff);
#endif
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

static KLT_BOOL _outOfBounds(float x, float y, int ncols, int nrows, 
                             int borderx, int bordery)
{
  return (x < borderx || x > ncols-1-borderx ||
          y < bordery || y > nrows-1-bordery );
}

/* ============================ Public entry (ENHANCED) ===================== */

#ifdef USE_CUDA
extern void _KLTComputeGradients(_KLT_FloatImage img, float sigma, 
                                _KLT_FloatImage gradx, _KLT_FloatImage grady);
extern void _KLTComputeSmoothedImage(_KLT_FloatImage img, float sigma, 
                                     _KLT_FloatImage smooth);
extern void _KLTToFloatImage(KLT_PixelType *img, int ncols, int nrows, 
                             _KLT_FloatImage floatimg);
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
  
#ifdef USE_CUDA
  /* Initialize memory pool once at start */
  tf_init_memory_pool(tc->nPyramidLevels);
  
  /* ============= START GPU TIMING ============= */
  /* Ensure no previous GPU work is running before starting timer */
  cudaDeviceSynchronize();
  
  /* Start timer - will capture ALL GPU operations:
   * 1. Constant memory uploads (cudaMemcpyToSymbol)
   * 2. Pyramid smoothing (_KLTComputeSmoothedImage) 
   * 3. Gradient computation (_KLTComputeGradients)
   * 4. Pyramid uploads to GPU
   * 5. Feature tracking kernels
   */
  cudaEventRecord(g_pool.timing_start, 0);
  
  /* Invalidate caches for new frame pair */
  tf_invalidate_caches();
  
  /* OPTIMIZATION: Upload tracking parameters to constant memory ONCE */
  cudaMemcpyToSymbol(c_window_width, &tc->window_width, sizeof(int));
  cudaMemcpyToSymbol(c_window_height, &tc->window_height, sizeof(int));
  cudaMemcpyToSymbol(c_step_factor, &tc->step_factor, sizeof(float));
  cudaMemcpyToSymbol(c_max_iterations, &tc->max_iterations, sizeof(int));
  cudaMemcpyToSymbol(c_min_determinant, &tc->min_determinant, sizeof(float));
  cudaMemcpyToSymbol(c_min_displacement, &tc->min_displacement, sizeof(float));
  cudaMemcpyToSymbol(c_max_residue, &tc->max_residue, sizeof(float));
#endif
  
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
  
#ifdef USE_CUDA
  /* Ensure all pyramid preprocessing GPU work (from convolve_gpu.cu) is complete
   * before proceeding to tracking. This is critical for accurate timing. */
  cudaDeviceSynchronize();
#endif
  
#ifdef USE_CUDA
  /* OPTIMIZATION: Upload pyramids with pointer swapping for sequential mode */
  int sequential_reuse = (tc->sequentialMode && tc->pyramid_last != NULL);
  upload_pyramids_to_gpu(pyramid1, pyramid1_gradx, pyramid1_grady,
                         pyramid2, pyramid2_gradx, pyramid2_grady,
                         tc->nPyramidLevels, sequential_reuse);
  
  /* OPTIMIZATION: Allocate persistent feature buffers */
  int nFeatures = featurelist->nFeatures;
  if (g_pool.max_features < nFeatures) {
    if (g_pool.d_features) cudaFree(g_pool.d_features);
    if (g_pool.h_features) cudaFreeHost(g_pool.h_features);
    
    cudaMalloc(&g_pool.d_features, nFeatures * sizeof(FeatureData));
    cudaMallocHost(&g_pool.h_features, nFeatures * sizeof(FeatureData));
    g_pool.max_features = nFeatures;
  }
  
  FeatureData *h_features = g_pool.h_features;
  FeatureData *d_features = g_pool.d_features;
#endif
  
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
  
#ifdef USE_CUDA
  /* ========== OPTIMIZATION: Use Batched GPU Tracking (from File 2) ========== */
  if (g_pool.initialized && !tc->lighting_insensitive) {
    // Process all pyramid levels with batched GPU kernel
    for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
      int stream_id = r % NUM_STREAMS;
      cudaStream_t stream = g_pool.streams[stream_id];
      
      // Prepare batched feature data for this pyramid level
      int active_features = 0;
      for (indx = 0 ; indx < nFeatures ; indx++)  {
        h_features[indx].status = featurelist->feature[indx]->val;
        
        if (featurelist->feature[indx]->val >= 0)  {
          if (r == tc->nPyramidLevels - 1) {
            // First level: initialize from feature list
            xloc = featurelist->feature[indx]->x;
            yloc = featurelist->feature[indx]->y;
            // Transform to coarsest resolution
            for (int rr = tc->nPyramidLevels - 1 ; rr >= 0 ; rr--)  {
              xloc /= subsampling;  yloc /= subsampling;
            }
            xloc *= subsampling;  yloc *= subsampling;
            h_features[indx].x1 = xloc;
            h_features[indx].y1 = yloc;
            h_features[indx].x2 = xloc;
            h_features[indx].y2 = yloc;
          } else {
            // Propagate from previous level
            h_features[indx].x1 *= subsampling;
            h_features[indx].y1 *= subsampling;
            h_features[indx].x2 *= subsampling;
            h_features[indx].y2 *= subsampling;
          }
          active_features++;
        }
      }
      
      if (active_features == 0) break;
      
      // Create texture objects for hardware interpolation
      int ncols_level = pyramid1->ncols[r];
      int nrows_level = pyramid1->nrows[r];
      cudaTextureObject_t tex_img1 = createTextureObject(
          g_pool.pyramid1_levels[r].d_img, ncols_level, nrows_level);
      cudaTextureObject_t tex_img2 = createTextureObject(
          g_pool.pyramid2_levels[r].d_img, ncols_level, nrows_level);
      cudaTextureObject_t tex_gradx1 = createTextureObject(
          g_pool.pyramid1_levels[r].d_gradx, ncols_level, nrows_level);
      cudaTextureObject_t tex_grady1 = createTextureObject(
          g_pool.pyramid1_levels[r].d_grady, ncols_level, nrows_level);
      cudaTextureObject_t tex_gradx2 = createTextureObject(
          g_pool.pyramid2_levels[r].d_gradx, ncols_level, nrows_level);
      cudaTextureObject_t tex_grady2 = createTextureObject(
          g_pool.pyramid2_levels[r].d_grady, ncols_level, nrows_level);
      
      // Single batched H2D transfer
      cudaMemcpyAsync(d_features, h_features, nFeatures * sizeof(FeatureData), 
                      cudaMemcpyHostToDevice, stream);
      
      // Launch batched tracking kernel
      int threadsPerBlock = 256;
      int blocksPerGrid = (nFeatures + threadsPerBlock - 1) / threadsPerBlock;
      
      trackFeaturesBatchedKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
          tex_img1, tex_img2,
          tex_gradx1, tex_grady1,
          tex_gradx2, tex_grady2,
          d_features,
          nFeatures,
          ncols_level, nrows_level);
      
      // Single batched D2H transfer
      cudaMemcpyAsync(h_features, d_features, nFeatures * sizeof(FeatureData), 
                      cudaMemcpyDeviceToHost, stream);
      
      // Sync this stream
      cudaStreamSynchronize(stream);
      
      // Destroy texture objects
      cudaDestroyTextureObject(tex_img1);
      cudaDestroyTextureObject(tex_img2);
      cudaDestroyTextureObject(tex_gradx1);
      cudaDestroyTextureObject(tex_grady1);
      cudaDestroyTextureObject(tex_gradx2);
      cudaDestroyTextureObject(tex_grady2);
      
      // Update feature status for this pyramid level
      for (indx = 0 ; indx < nFeatures ; indx++)  {
        if (featurelist->feature[indx]->val >= 0) {
          if (h_features[indx].status == KLT_SMALL_DET || 
              h_features[indx].status == KLT_OOB) {
            featurelist->feature[indx]->val = h_features[indx].status;
          }
        }
      }
    }
    
    // Final update of feature positions from GPU results
    for (indx = 0 ; indx < nFeatures ; indx++)  {
      if (featurelist->feature[indx]->val >= 0) {
        val = h_features[indx].status;
        
        if (val == KLT_TRACKED || val == KLT_MAX_ITERATIONS) {
          xlocout = h_features[indx].x2;
          ylocout = h_features[indx].y2;
        } else {
          xlocout = -1.0;
          ylocout = -1.0;
        }
        
        /* Record feature */
        if (val == KLT_OOB) {
          featurelist->feature[indx]->x   = -1.0;
          featurelist->feature[indx]->y   = -1.0;
          featurelist->feature[indx]->val = KLT_OOB;
        } else if (_outOfBounds(xlocout, ylocout, ncols, nrows, 
                               tc->borderx, tc->bordery))  {
          featurelist->feature[indx]->x   = -1.0;
          featurelist->feature[indx]->y   = -1.0;
          featurelist->feature[indx]->val = KLT_OOB;
        } else if (val == KLT_SMALL_DET)  {
          featurelist->feature[indx]->x   = -1.0;
          featurelist->feature[indx]->y   = -1.0;
          featurelist->feature[indx]->val = KLT_SMALL_DET;
        } else if (val == KLT_LARGE_RESIDUE)  {
          featurelist->feature[indx]->x   = -1.0;
          featurelist->feature[indx]->y   = -1.0;
          featurelist->feature[indx]->val = KLT_LARGE_RESIDUE;
        } else if (val == KLT_MAX_ITERATIONS)  {
          featurelist->feature[indx]->x   = -1.0;
          featurelist->feature[indx]->y   = -1.0;
          featurelist->feature[indx]->val = KLT_MAX_ITERATIONS;
        } else  {
          featurelist->feature[indx]->x = xlocout;
          featurelist->feature[indx]->y = ylocout;
          featurelist->feature[indx]->val = KLT_TRACKED;
        }
      }
    }
  } else {
#endif
    /* ============ ORIGINAL PER-FEATURE TRACKING LOOP (CPU FALLBACK) ========== */
    for (indx = 0 ; indx < featurelist->nFeatures ; indx++)  {
      if (featurelist->feature[indx]->val >= 0)  {
        xloc = featurelist->feature[indx]->x;
        yloc = featurelist->feature[indx]->y;
        
        /* Scale to coarsest pyramid level */
        for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
          xloc /= subsampling;  yloc /= subsampling;
        }
        xlocout = xloc;  ylocout = yloc;
        
        /* Track through pyramid levels (coarse to fine) */
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
                              tc->lighting_insensitive,
                              r);
          
          if (val==KLT_SMALL_DET || val==KLT_OOB)
            break;
        }
        
        /* Update feature status */
        if (val == KLT_OOB) {
          featurelist->feature[indx]->x   = -1.0f;
          featurelist->feature[indx]->y   = -1.0f;
          featurelist->feature[indx]->val = KLT_OOB;
        } else if (_outOfBounds(xlocout, ylocout, ncols, nrows, 
                               tc->borderx, tc->bordery))  {
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
#ifdef USE_CUDA
  }
  
  /* ============= FINAL SYNCHRONIZATION BEFORE STOPPING TIMER ============= */
  /* Ensure ALL GPU work across all streams is complete before stopping timer */
  if (g_pool.initialized) {
    for (int s = 0; s < NUM_STREAMS; s++) {
      cudaStreamSynchronize(g_pool.streams[s]);
    }
  }
  
  /* Also synchronize the convolve_gpu.cu stream and any other device work */
  cudaDeviceSynchronize();
  
  /* ============= STOP GPU TIMING AND PRINT RESULT ============= */
  cudaEventRecord(g_pool.timing_stop, 0);
  cudaEventSynchronize(g_pool.timing_stop);
  
  float gpu_time_ms = 0.0f;
  cudaEventElapsedTime(&gpu_time_ms, g_pool.timing_start, g_pool.timing_stop);
  float gpu_time_sec = gpu_time_ms / 1000.0f;
  
  fprintf(stderr, "\n========================================\n");
  fprintf(stderr, "GPU Execution Time: %.4f seconds\n", gpu_time_sec);
  fprintf(stderr, "========================================\n");
  fprintf(stderr, "NOTE: Time includes pyramid preprocessing\n");
  fprintf(stderr, "      (smoothing + gradients) and tracking\n");
  fprintf(stderr, "========================================\n");
#endif
  
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
  
  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features successfully tracked.\n",
      KLTCountRemainingFeatures(featurelist));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_tf*.pgm'.\n");
    fflush(stderr);
  }
}