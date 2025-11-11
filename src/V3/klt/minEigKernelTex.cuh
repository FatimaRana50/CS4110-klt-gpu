#pragma once
#include <cuda_runtime.h>

/* Computes minimum eigenvalue at every valid pixel (corner strength)
 * Writes results as (x,y,val) triplets into d_pointlist
 */
__global__ void minEigKernelTex(cudaTextureObject_t tex_gx,
                                cudaTextureObject_t tex_gy,
                                int *d_pointlist,
                                int ncols, int nrows,
                                int win_hw, int win_hh,
                                int borderx, int bordery,
                                int skip);
