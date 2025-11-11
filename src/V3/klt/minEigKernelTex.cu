/*********************************************************************
 * minEigKernelTex.cu
 * Batched minimum-eigenvalue kernel using textures for gradients.
 *********************************************************************/
#include "minEigKernelTex.cuh"
#include <math.h>

__global__ void minEigKernelTex(cudaTextureObject_t tex_gx,
                                cudaTextureObject_t tex_gy,
                                int *d_pointlist,
                                int ncols, int nrows,
                                int win_hw, int win_hh,
                                int borderx, int bordery,
                                int skip)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int x = borderx + gx * (skip + 1);
    int y = bordery + gy * (skip + 1);

    if (x >= ncols - borderx || y >= nrows - bordery) return;

    float gxx = 0.f, gxy = 0.f, gyy = 0.f;

    // Accumulate gradient covariance matrix within window
    for (int dy = -win_hh; dy <= win_hh; ++dy) {
        for (int dx = -win_hw; dx <= win_hw; ++dx) {
            float gxv = tex2D<float>(tex_gx, x + dx + 0.5f, y + dy + 0.5f);
            float gyv = tex2D<float>(tex_gy, x + dx + 0.5f, y + dy + 0.5f);
            gxx += gxv * gxv;
            gxy += gxv * gyv;
            gyy += gyv * gyv;
        }
    }

    // Compute smallest eigenvalue of [gxx gxy; gxy gyy]
    float tr = gxx + gyy;
    float det_term = (gxx - gyy) * (gxx - gyy) + 4.f * gxy * gxy;
    float mineig = 0.5f * (tr - sqrtf(det_term));

    // Write to global memory
    int gridW = (ncols - 2 * borderx + skip) / (skip + 1);
    int idx = (gy * gridW + gx) * 3;
    d_pointlist[idx + 0] = x;
    d_pointlist[idx + 1] = y;
    d_pointlist[idx + 2] = __float2int_rn(mineig);
}
