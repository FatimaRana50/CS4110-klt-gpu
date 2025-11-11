/*********************************************************************
 * lkTrackBatchTex.cu
 * Batched Lucas–Kanade optical flow tracker using textures.
 *********************************************************************/
#include "lkTrackBatchTex.cuh"
#include <math.h>

#define MAX_ITERS 5
#define EPSILON   1e-3f

__global__ void lkTrackBatchTex(cudaTextureObject_t tex_img1,
                                cudaTextureObject_t tex_img2,
                                cudaTextureObject_t tex_gx1,
                                cudaTextureObject_t tex_gy1,
                                const float* d_x1, const float* d_y1,
                                float* d_x2, float* d_y2,
                                int nfeat, int win_w, int win_h,
                                int ncols, int nrows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfeat) return;

    float x = d_x1[i];
    float y = d_y1[i];
    float u = 0.f, v = 0.f;

    for (int it = 0; it < MAX_ITERS; ++it) {
        float Gxx = 0.f, Gxy = 0.f, Gyy = 0.f, Ex = 0.f, Ey = 0.f;

        for (int dy = -win_h; dy <= win_h; ++dy) {
            for (int dx = -win_w; dx <= win_w; ++dx) {
                float xx1 = x + dx + 0.5f;
                float yy1 = y + dy + 0.5f;
                float xx2 = x + u + dx + 0.5f;
                float yy2 = y + v + dy + 0.5f;

                float Ix = tex2D<float>(tex_gx1, xx1, yy1);
                float Iy = tex2D<float>(tex_gy1, xx1, yy1);
                float I1 = tex2D<float>(tex_img1, xx1, yy1);
                float I2 = tex2D<float>(tex_img2, xx2, yy2);
                float It = I2 - I1;

                Gxx += Ix * Ix;
                Gxy += Ix * Iy;
                Gyy += Iy * Iy;
                Ex  += Ix * It;
                Ey  += Iy * It;
            }
        }

        float det = Gxx * Gyy - Gxy * Gxy + 1e-6f;
        float du = (-Gyy * Ex + Gxy * Ey) / det;
        float dv = ( Gxy * Ex - Gxx * Ey) / det;
        u += du;
        v += dv;

        if (fabsf(du) + fabsf(dv) < EPSILON)
            break;
    }

    d_x2[i] = x + u;
    d_y2[i] = y + v;
}
