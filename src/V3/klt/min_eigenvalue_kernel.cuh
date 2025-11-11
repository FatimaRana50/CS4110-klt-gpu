#ifndef MIN_EIGENVALUE_KERNEL_CUH
#define MIN_EIGENVALUE_KERNEL_CUH

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
    int nSkippedPixels);

#ifdef __cplusplus
}
#endif

#endif