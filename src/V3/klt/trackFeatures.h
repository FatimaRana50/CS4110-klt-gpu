#ifndef _KLT_CUDA_H_
#define _KLT_CUDA_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <assert.h>
#include <math.h>		/* fabs() */
#include <stdlib.h>		/* malloc() */
#include <stdio.h>		/* fflush() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve_gpu.cuh"	/* for computing pyramid */
#include "klt.h"
#include "klt_util.h"	/* _KLT_FloatImage */
#include "pyramid.h"
#include "cuda_runtime.h"

/* ============================================================
 * Device image container (opaque pointer on host)
 * ============================================================
 */
typedef struct {
    float *d_ptr;    /* device pointer */
    int ncols;
    int nrows;
} KLTDeviceImage;

/* ============================================================
 * Image Upload / Free
 * ============================================================
 */
int klt_cuda_upload_image(const float *h_img, int ncols, int nrows, KLTDeviceImage *out);
void klt_cuda_free_image(KLTDeviceImage *img);

int klt_cuda_upload_gradients(const float *h_gradx, const float *h_grady, int ncols, int nrows,
                              KLTDeviceImage *out_gradx, KLTDeviceImage *out_grady);

/* ============================================================
 * Batched Computation
 * ============================================================
 */
int klt_cuda_compute_batch(
    const KLTDeviceImage *img1_dev, const KLTDeviceImage *img2_dev,
    const KLTDeviceImage *gradx1_dev, const KLTDeviceImage *grady1_dev,
    const KLTDeviceImage *gradx2_dev, const KLTDeviceImage *grady2_dev,
    int ncols, int nrows,
    const float *feat_x_host, const float *feat_y_host,
    const float *feat_x2_host, const float *feat_y2_host,
    int feature_count,
    int width, int height,
    int lighting_insensitive,
    float *out_gxx, float *out_gxy, float *out_gyy,
    float *out_ex, float *out_ey, float *out_residue
);

/* ============================================================
 * Gradient Sum (Existing GPU Function)
 * ============================================================
 */
void computeGradientSumGPUShared(
    float* gradx1, float* grady1,
    float* gradx2, float* grady2,
    int ncols, int nrows,
    float x1, float y1, float x2, float y2,
    int width, int height, float* gradx, float* grady
);

/* ============================================================
 * Intensity Difference (New GPU Function)
 * ============================================================
 */
void _computeIntensityDifferenceGPU(
    const float* d_img1,
    const float* d_img2,
    int ncols, int nrows,
    float x1, float y1,
    float x2, float y2,
    int width, int height,
    float* d_imgdiff
);

/* ============================================================
 * Device & Kernel Declarations
 * ============================================================
 */
__device__ float dev_bilinear(const float *img, int ncols, int nrows, float x, float y);

__global__ void gradientSumKernelShared(
    const float* gradx1, const float* grady1,
    const float* gradx2, const float* grady2,
    float* gradx, float* grady,
    int width, int height,
    float x1, float y1, float x2, float y2,
    int ncols, int nrows
);

/* ============================================================
 * CUDA Error Checking Macros
 * ============================================================
 */
#define KLT_CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return -1; \
    } \
} while(0)

#define KLT_CUDA_CHECK_VOID(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return; \
    } \
} while(0)

#ifdef __cplusplus
}
#endif

#endif /* _KLT_CUDA_H_ */