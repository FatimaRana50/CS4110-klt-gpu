/*********************************************************************
 * selectGoodFeatures.c - OpenACC-enabled version
 *
 * OpenACC offloads the eigenvalue scoring loop over candidate pixels.
 * All original KLT logic (sorting, min-distance, bookkeeping) is
 * preserved on the CPU.
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define fsqrt(X) sqrt(X)

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"

/* OpenACC header */
#include <openacc.h>

int KLT_verbose = 1;

typedef enum {SELECTING_ALL, REPLACING_SOME} selectionMode;

#define SWAP3(list, i, j)               \
{register int *pi, *pj, tmp;            \
     pi=list+3*(i); pj=list+3*(j);      \
     tmp=*pi; *pi++=*pj; *pj++=tmp;     \
     tmp=*pi; *pi++=*pj; *pj++=tmp;     \
     tmp=*pi; *pi++=*pj; *pj=tmp; }

void _quicksort(int *pointlist, int n)
{
  unsigned int i, j, ln, rn;
  while (n > 1) {
    SWAP3(pointlist, 0, n/2);
    for (i = 0, j = n; ; ) {
      do --j; while (pointlist[3*j+2] < pointlist[2]);
      do ++i; while (i < j && pointlist[3*i+2] > pointlist[2]);
      if (i >= j) break;
      SWAP3(pointlist, i, j);
    }
    SWAP3(pointlist, j, 0);
    ln = j;
    rn = n - ++j;
    if (ln < rn) {
      _quicksort(pointlist, ln);
      pointlist += 3*j;
      n = rn;
    } else {
      _quicksort(pointlist + 3*j, rn);
      n = ln;
    }
  }
}
#undef SWAP3

static void _fillFeaturemap(int x, int y, uchar *featuremap,
                            int mindist, int ncols, int nrows)
{
  int ix, iy;
  for (iy = y - mindist ; iy <= y + mindist ; iy++)
    for (ix = x - mindist ; ix <= x + mindist ; ix++)
      if (ix >= 0 && ix < ncols && iy >= 0 && iy < nrows)
        featuremap[iy*ncols+ix] = 1;
}

static void _enforceMinimumDistance(
  int *pointlist, int npoints, KLT_FeatureList featurelist,
  int ncols, int nrows, int mindist, int min_eigenvalue,
  KLT_BOOL overwriteAllFeatures)
{
  int indx, x, y, val;
  uchar *featuremap;
  int *ptr;

  if (min_eigenvalue < 1)  min_eigenvalue = 1;
  featuremap = (uchar *) malloc(ncols * nrows * sizeof(uchar));
  memset(featuremap, 0, ncols*nrows);
  mindist--;

  if (!overwriteAllFeatures)
    for (indx = 0 ; indx < featurelist->nFeatures ; indx++)
      if (featurelist->feature[indx]->val >= 0)  {
        x = (int) featurelist->feature[indx]->x;
        y = (int) featurelist->feature[indx]->y;
        _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
      }

  ptr = pointlist;
  indx = 0;
  while (1)  {
    if (ptr >= pointlist + 3*npoints)  {
      while (indx < featurelist->nFeatures)  {
        if (overwriteAllFeatures || featurelist->feature[indx]->val < 0) {
          featurelist->feature[indx]->x   = -1;
          featurelist->feature[indx]->y   = -1;
          featurelist->feature[indx]->val = KLT_NOT_FOUND;
          featurelist->feature[indx]->aff_img = NULL;
          featurelist->feature[indx]->aff_img_gradx = NULL;
          featurelist->feature[indx]->aff_img_grady = NULL;
          featurelist->feature[indx]->aff_x = -1.0;
          featurelist->feature[indx]->aff_y = -1.0;
          featurelist->feature[indx]->aff_Axx = 1.0;
          featurelist->feature[indx]->aff_Ayx = 0.0;
          featurelist->feature[indx]->aff_Axy = 0.0;
          featurelist->feature[indx]->aff_Ayy = 1.0;
        }
        indx++;
      }
      break;
    }
    x = *ptr++; y = *ptr++; val = *ptr++;
    assert(x >= 0 && x < ncols && y >= 0 && y < nrows);
    while (!overwriteAllFeatures && indx < featurelist->nFeatures &&
           featurelist->feature[indx]->val >= 0) indx++;
    if (indx >= featurelist->nFeatures)  break;
    if (!featuremap[y*ncols+x] && val >= min_eigenvalue)  {
      featurelist->feature[indx]->x   = (KLT_locType) x;
      featurelist->feature[indx]->y   = (KLT_locType) y;
      featurelist->feature[indx]->val = (int) val;
      featurelist->feature[indx]->aff_img = NULL;
      featurelist->feature[indx]->aff_img_gradx = NULL;
      featurelist->feature[indx]->aff_img_grady = NULL;
      featurelist->feature[indx]->aff_x = -1.0;
      featurelist->feature[indx]->aff_y = -1.0;
      featurelist->feature[indx]->aff_Axx = 1.0;
      featurelist->feature[indx]->aff_Ayx = 0.0;
      featurelist->feature[indx]->aff_Axy = 0.0;
      featurelist->feature[indx]->aff_Ayy = 1.0;
      indx++;
      _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
    }
  }
  free(featuremap);
}

#ifdef KLT_USE_QSORT
static int _comparePoints(const void *a, const void *b)
{
  int v1 = *(((int *) a) + 2);
  int v2 = *(((int *) b) + 2);
  if (v1 > v2)  return(-1);
  else if (v1 < v2)  return(1);
  else return(0);
}
#endif

static void _sortPointList(int *pointlist, int npoints)
{
#ifdef KLT_USE_QSORT
  qsort(pointlist, npoints, 3*sizeof(int), _comparePoints);
#else
  _quicksort(pointlist, npoints);
#endif
}

static float _minEigenvalue(float gxx, float gxy, float gyy)
{
  return (float) ((gxx + gyy - sqrt((gxx - gyy)*(gxx - gyy) + 4*gxy*gxy))/2.0f);
}

/*********************************************************************/

void _KLTSelectGoodFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img,
  int ncols, int nrows,
  KLT_FeatureList featurelist,
  selectionMode mode)
{
  _KLT_FloatImage floatimg, gradx, grady;
  int window_hw, window_hh;
  int *pointlist;
  int npoints = 0;
  KLT_BOOL overwriteAllFeatures = (mode == SELECTING_ALL) ? TRUE : FALSE;
  KLT_BOOL floatimages_created = FALSE;

  /* Sanitize window size */
  if (tc->window_width % 2 != 1) {
    tc->window_width = tc->window_width+1;
    KLTWarning("Tracking context's window width must be odd. Changing to %d.\n",
               tc->window_width);
  }
  if (tc->window_height % 2 != 1) {
    tc->window_height = tc->window_height+1;
    KLTWarning("Tracking context's window height must be odd. Changing to %d.\n",
               tc->window_height);
  }
  if (tc->window_width < 3) {
    tc->window_width = 3;
    KLTWarning("Tracking context's window width must be at least three. Changing to %d.\n",
               tc->window_width);
  }
  if (tc->window_height < 3) {
    tc->window_height = 3;
    KLTWarning("Tracking context's window height must be at least three. Changing to %d.\n",
               tc->window_height);
  }
  window_hw = tc->window_width/2;
  window_hh = tc->window_height/2;

  pointlist = (int *) malloc(ncols * nrows * 3 * sizeof(int));

  /* Reuse previous pyramid if sequential mode, else compute fresh */
  if (mode == REPLACING_SOME && tc->sequentialMode && tc->pyramid_last != NULL)  {
    floatimg = ((_KLT_Pyramid) tc->pyramid_last)->img[0];
    gradx    = ((_KLT_Pyramid) tc->pyramid_last_gradx)->img[0];
    grady    = ((_KLT_Pyramid) tc->pyramid_last_grady)->img[0];
    assert(gradx != NULL);
    assert(grady != NULL);
  } else  {
    floatimages_created = TRUE;
    floatimg = _KLTCreateFloatImage(ncols, nrows);
    gradx    = _KLTCreateFloatImage(ncols, nrows);
    grady    = _KLTCreateFloatImage(ncols, nrows);
    if (tc->smoothBeforeSelecting)  {
      _KLT_FloatImage tmpimg = _KLTCreateFloatImage(ncols, nrows);
      _KLTToFloatImage(img, ncols, nrows, tmpimg);
      _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);
      _KLTFreeFloatImage(tmpimg);
    } else {
      _KLTToFloatImage(img, ncols, nrows, floatimg);
    }
    _KLTComputeGradients(floatimg, tc->grad_sigma, gradx, grady);
  }

  if (tc->writeInternalImages)  {
    _KLTWriteFloatImageToPGM(floatimg, "kltimg_sgfrlf.pgm");
    _KLTWriteFloatImageToPGM(gradx,   "kltimg_sgfrlf_gx.pgm");
    _KLTWriteFloatImageToPGM(grady,   "kltimg_sgfrlf_gy.pgm");
  }

/* ===================== OPENACC COMPUTATION ===================== */
{
    unsigned int limit = 1;
    for (int i = 0 ; i < (int)sizeof(int) ; i++)  limit *= 256;
    limit = limit/2 - 1;

    int borderx = tc->borderx < window_hw ? window_hw : tc->borderx;
    int bordery = tc->bordery < window_hh ? window_hh : tc->bordery;

    const int step = tc->nSkippedPixels + 1;

    int nx = (ncols - 2*borderx) / step;
    int ny = (nrows - 2*bordery) / step;
    int total_points = nx * ny;

    if (total_points <= 0) {
        npoints = 0;
    } else {

        float *gx_img = gradx->data;
        float *gy_img = grady->data;

        /* 
         * Optimized OpenACC kernel:
         * - loops over candidate indices (iy,ix)
         * - eliminates divisions
         * - precomputes window boundaries
         * - precomputes row-base offsets
         * - keeps accumulation order identical
         */

        #pragma acc parallel loop collapse(2)
        for (int iy = 0 ; iy < ny ; iy++) {
            for (int ix = 0 ; ix < nx ; ix++) {

                int x = borderx + ix * step;
                int y = bordery + iy * step;

                /* Precompute window bounds */
                int x0 = x - window_hw;
                int x1 = x + window_hw;
                int y0 = y - window_hh;
                int y1 = y + window_hh;

                float gxx = 0.0f;
                float gxy = 0.0f;
                float gyy = 0.0f;

                /* 
                 * Math-preserved summation.
                 * Using row_base avoids repeated (yy*ncols).
                 */
                for (int yy = y0 ; yy <= y1 ; yy++) {
                    int row_base = yy * ncols;
                    for (int xx = x0 ; xx <= x1 ; xx++) {
                        float gx = gx_img[row_base + xx];
                        float gy = gy_img[row_base + xx];
                        gxx += gx * gx;
                        gxy += gx * gy;
                        gyy += gy * gy;
                    }
                }

                /* Eigenvalue formula EXACTLY preserved */
                float diff = gxx - gyy;
                float sq = diff*diff + 4.0f * gxy * gxy;
                float val = (gxx + gyy - sqrtf(sq)) * 0.5f;

                if (val > (float)limit) val = (float)limit;

                int idx = iy * nx + ix;

                pointlist[3*idx    ] = x;
                pointlist[3*idx + 1] = y;
                pointlist[3*idx + 2] = (int)val;
            }
        }

        npoints = total_points;
    }
}
/* =================== END OPENACC COMPUTATION =================== */


  _sortPointList(pointlist, npoints);

  if (tc->mindist < 0)  {
    KLTWarning("(_KLTSelectGoodFeatures) Tracking context field tc->mindist "
               "is negative (%d); setting to zero", tc->mindist);
    tc->mindist = 0;
  }

  _enforceMinimumDistance(pointlist, npoints, featurelist, ncols, nrows,
                          tc->mindist, tc->min_eigenvalue, overwriteAllFeatures);

  free(pointlist);

  if (floatimages_created)  {
    _KLTFreeFloatImage(floatimg);
    _KLTFreeFloatImage(gradx);
    _KLTFreeFloatImage(grady);
  }
}

void KLTSelectGoodFeatures(
  KLT_TrackingContext tc, KLT_PixelType *img,
  int ncols, int nrows, KLT_FeatureList fl)
{
  if (KLT_verbose >= 1)  {
    fprintf(stderr,
      "(KLT) Selecting the %d best features from a %d by %d image...  ",
      fl->nFeatures, ncols, nrows);
    fflush(stderr);
  }

  _KLTSelectGoodFeatures(tc, img, ncols, nrows, fl, SELECTING_ALL);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features found.\n", KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}

void KLTReplaceLostFeatures(
  KLT_TrackingContext tc, KLT_PixelType *img,
  int ncols, int nrows, KLT_FeatureList fl)
{
  int nLostFeatures = fl->nFeatures - KLTCountRemainingFeatures(fl);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,
      "(KLT) Attempting to replace %d features in a %d by %d image...  ",
      nLostFeatures, ncols, nrows);
    fflush(stderr);
  }

  if (nLostFeatures > 0)
    _KLTSelectGoodFeatures(tc, img, ncols, nrows, fl, REPLACING_SOME);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features replaced.\n",
            nLostFeatures - fl->nFeatures + KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}
