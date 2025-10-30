/*********************************************************************
 * convolve_gpu.cuh
 * 
 * CUDA-accelerated convolution operations for KLT Feature Tracker
 * Optimized for maximum throughput using shared memory, constant
 * memory, and memory coalescing strategies.
 *********************************************************************/

#ifndef _CONVOLVE_GPU_CUH_
#define _CONVOLVE_GPU_CUH_

#include "klt.h"
#include "klt_util.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

/* Maximum kernel width for constant memory storage */
#define MAX_KERNEL_WIDTH_GPU 71

/* CUDA error checking macro */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/* Kernel launch configuration structure */
#ifdef __CUDACC__   // Only compile this block when using nvcc
typedef struct {
    dim3 blockDim;
    dim3 gridDim;
    size_t sharedMemSize;
} KernelConfig;
#endif


/*********************************************************************
 * GPU Memory Management
 *********************************************************************/

/* Allocate GPU memory for float image */
float* allocateGPUFloatImage(int ncols, int nrows);

/* Free GPU memory */
void freeGPUFloatImage(float* d_img);

/* Copy image to GPU */
void copyImageToGPU(const float* h_img, float* d_img, int ncols, int nrows);

/* Copy image from GPU */
void copyImageFromGPU(float* h_img, const float* d_img, int ncols, int nrows);

/*********************************************************************
 * GPU Convolution Functions (Host Interface)
 *********************************************************************/

/* Horizontal convolution on GPU */
void convolveImageHorizGPU(
    const float* d_imgin,
    int ncols, int nrows,
    const float* kernel_data,
    int kernel_width,
    float* d_imgout);

/* Vertical convolution on GPU */
void convolveImageVertGPU(
    const float* d_imgin,
    int ncols, int nrows,
    const float* kernel_data,
    int kernel_width,
    float* d_imgout);

/* Combined gradient computation on GPU */
void computeGradientsGPU(
    const float* d_img,
    int ncols, int nrows,
    float sigma,
    float* d_gradx,
    float* d_grady);

/* Smoothed image computation on GPU */
void computeSmoothedImageGPU(
    const float* d_img,
    int ncols, int nrows,
    float sigma,
    float* d_smooth);

/*********************************************************************
 * High-Level GPU Pipeline Functions
 *********************************************************************/

/* Convert image to float and upload to GPU */
float* convertAndUploadToGPU(
    KLT_PixelType* img,
    int ncols, int nrows);

/* Complete gradient computation pipeline on GPU */
void computeGradientsPipelineGPU(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady);

/* Compute kernel on GPU and store in constant memory */
void uploadKernelToGPU(const float* kernel_data, int kernel_width, int kernel_type);

/*********************************************************************
 * LEGACY _KLT* FUNCTION API (GPU-BACKED)
 * These maintain compatibility with existing KLT C modules
 *********************************************************************/

void _KLTToFloatImage(
    KLT_PixelType *img,
    int ncols, int nrows,
    _KLT_FloatImage floatimg);

void _KLTComputeGradients(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady);

void _KLTGetKernelWidths(
    float sigma,
    int *gauss_width,
    int *gaussderiv_width);

void _KLTComputeSmoothedImage(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage smooth);

#ifdef __cplusplus
}
#endif

#endif