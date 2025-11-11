/*********************************************************************
 * klt_gpu_context.cu
 * GPU context manager for KLT tracker
 *
 * Handles:
 *   - Persistent device allocations
 *   - Streams & events
 *   - Texture creation/destruction
 *   - Host pinned buffers
 *********************************************************************/

#include "klt_gpu_context.h"
#include <string.h>

/* -------------------------------------------------------------- */
/* Utility: create a 2D texture object for a device float image   */
/* -------------------------------------------------------------- */
static cudaTextureObject_t make_tex_2d(const float* d_ptr, int w, int h)
{
    cudaResourceDesc res = {};
    res.resType = cudaResourceTypePitch2D;
    res.res.pitch2D.devPtr = (void*)d_ptr;
    res.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    res.res.pitch2D.width = w;
    res.res.pitch2D.height = h;
    res.res.pitch2D.pitchInBytes = w * sizeof(float);

    cudaTextureDesc tex = {};
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode     = cudaFilterModeLinear;
    tex.readMode       = cudaReadModeElementType;
    tex.normalizedCoords = 0;

    cudaTextureObject_t t = 0;
    cudaCreateTextureObject(&t, &res, &tex, nullptr);
    return t;
}

/* -------------------------------------------------------------- */
/* Create and initialize GPU context                              */
/* -------------------------------------------------------------- */
int klt_gpu_create(KLTGpuContext* c,
                   int ncols, int nrows,
                   int nlevels, int subsampling,
                   int maxF,
                   int ww, int wh,
                   int bx, int by,
                   int skip)
{
    if (!c) return -1;
    memset(c, 0, sizeof(*c));

    c->ncols = ncols;
    c->nrows = nrows;
    c->nlevels = nlevels;
    c->subsampling = subsampling;
    c->max_features = maxF;
    c->window_w = ww;
    c->window_h = wh;
    c->borderx = bx;
    c->bordery = by;
    c->nSkippedPixels = skip;

    // Streams
    for (int i = 0; i < 4; ++i)
        cudaStreamCreateWithFlags(&c->streams[i], cudaStreamNonBlocking);

    // Events
    cudaEventCreateWithFlags(&c->ev_up_done,      cudaEventDisableTiming);
    cudaEventCreateWithFlags(&c->ev_feat_sel_done,cudaEventDisableTiming);
    cudaEventCreateWithFlags(&c->ev_track_done,   cudaEventDisableTiming);

    // Device allocations
    size_t imgBytes = (size_t)ncols * nrows * sizeof(float);
    cudaMalloc(&c->d_img,   imgBytes);
    cudaMalloc(&c->d_gradx, imgBytes);
    cudaMalloc(&c->d_grady, imgBytes);

    cudaMalloc(&c->d_feat_x,  maxF * sizeof(float));
    cudaMalloc(&c->d_feat_y,  maxF * sizeof(float));
    cudaMalloc(&c->d_feat_x2, maxF * sizeof(float));
    cudaMalloc(&c->d_feat_y2, maxF * sizeof(float));
    cudaMalloc(&c->d_feat_val,maxF * sizeof(int));

    // Host pinned buffers
    cudaHostAlloc((void**)&c->h_img_pinned, imgBytes, cudaHostAllocDefault);
    c->h_pointlist_pinned = nullptr;
    c->h_pointlist_bytes = 0;

    // No textures yet
    c->tex_img = 0;
    c->tex_gx  = 0;
    c->tex_gy  = 0;

    c->d_pointlist = nullptr;
    c->d_workspace = nullptr;
    return 0;
}

/* -------------------------------------------------------------- */
/* Destroy all allocations                                        */
/* -------------------------------------------------------------- */
void klt_gpu_destroy(KLTGpuContext* c)
{
    if (!c) return;

    if (c->tex_img) cudaDestroyTextureObject(c->tex_img);
    if (c->tex_gx)  cudaDestroyTextureObject(c->tex_gx);
    if (c->tex_gy)  cudaDestroyTextureObject(c->tex_gy);

    cudaFree(c->d_img);
    cudaFree(c->d_gradx);
    cudaFree(c->d_grady);
    cudaFree(c->d_feat_x);
    cudaFree(c->d_feat_y);
    cudaFree(c->d_feat_x2);
    cudaFree(c->d_feat_y2);
    cudaFree(c->d_feat_val);
    if (c->d_pointlist) cudaFree(c->d_pointlist);
    if (c->d_workspace) cudaFree(c->d_workspace);

    if (c->h_img_pinned) cudaFreeHost(c->h_img_pinned);
    if (c->h_pointlist_pinned) cudaFreeHost(c->h_pointlist_pinned);

    cudaEventDestroy(c->ev_up_done);
    cudaEventDestroy(c->ev_feat_sel_done);
    cudaEventDestroy(c->ev_track_done);

    for (int i = 0; i < 4; ++i)
        if (c->streams[i]) cudaStreamDestroy(c->streams[i]);

    memset(c, 0, sizeof(*c));
}

/* -------------------------------------------------------------- */
/* Recreate textures for gradients or images                      */
/* -------------------------------------------------------------- */
int klt_gpu_build_textures(KLTGpuContext* c)
{
    if (c->tex_img) cudaDestroyTextureObject(c->tex_img);
    if (c->tex_gx)  cudaDestroyTextureObject(c->tex_gx);
    if (c->tex_gy)  cudaDestroyTextureObject(c->tex_gy);

    c->tex_img = make_tex_2d(c->d_img,   c->ncols, c->nrows);
    c->tex_gx  = make_tex_2d(c->d_gradx, c->ncols, c->nrows);
    c->tex_gy  = make_tex_2d(c->d_grady, c->ncols, c->nrows);
    return 0;
}

/* -------------------------------------------------------------- */
/* Upload frame to device asynchronously                           */
/* -------------------------------------------------------------- */
int klt_gpu_upload_frame(KLTGpuContext* c, const float* h_img)
{
    if (!c || !h_img) return -1;
    size_t bytes = (size_t)c->ncols * c->nrows * sizeof(float);
    memcpy(c->h_img_pinned, h_img, bytes);
    cudaMemcpyAsync(c->d_img, c->h_img_pinned, bytes,
                    cudaMemcpyHostToDevice, c->streams[0]);
    cudaEventRecord(c->ev_up_done, c->streams[0]);
    return 0;
}

/* -------------------------------------------------------------- */
/* Ensure pointlist buffer exists and is large enough             */
/* -------------------------------------------------------------- */
int klt_gpu_ensure_pointlist(KLTGpuContext* c, size_t triplets)
{
    size_t bytes = triplets * sizeof(int);
    if (bytes > c->h_pointlist_bytes) {
        if (c->h_pointlist_pinned) cudaFreeHost(c->h_pointlist_pinned);
        cudaHostAlloc((void**)&c->h_pointlist_pinned, bytes, cudaHostAllocDefault);
        c->h_pointlist_bytes = bytes;
    }
    if (!c->d_pointlist) cudaMalloc(&c->d_pointlist, bytes);
    return 0;
}
