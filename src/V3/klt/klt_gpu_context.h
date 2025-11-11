#ifndef KLT_GPU_CONTEXT_H
#define KLT_GPU_CONTEXT_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int ncols, nrows;
    int nlevels, subsampling;
    int max_features;
    int window_w, window_h;
    int borderx, bordery;
    int nSkippedPixels;

    cudaStream_t streams[4];
    cudaEvent_t ev_up_done, ev_feat_sel_done, ev_track_done;

    float *d_img, *d_gradx, *d_grady;
    float *d_feat_x, *d_feat_y, *d_feat_x2, *d_feat_y2;
    int   *d_feat_val, *d_pointlist;

    float *h_img_pinned;
    int   *h_pointlist_pinned;
    size_t h_pointlist_bytes;

    cudaTextureObject_t tex_img, tex_gx, tex_gy;

    float *d_workspace;
} KLTGpuContext;

int  klt_gpu_create(KLTGpuContext* ctx,
                    int ncols, int nrows,
                    int nlevels, int subsampling,
                    int max_features,
                    int window_w, int window_h,
                    int borderx, int bordery,
                    int nSkippedPixels);

void klt_gpu_destroy(KLTGpuContext* ctx);
int  klt_gpu_build_textures(KLTGpuContext* ctx);
int  klt_gpu_upload_frame(KLTGpuContext* ctx, const float* h_img);
int  klt_gpu_ensure_pointlist(KLTGpuContext* ctx, size_t triplets);

#ifdef __cplusplus
}
#endif
#endif
