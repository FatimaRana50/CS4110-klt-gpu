/*********************************************************************
 * pyramid.c  (OpenACC-enhanced)
 *
 * - Builds Gaussian pyramid levels as in the original KLT code.
 * - OpenACC parallelizes the per-level subsampling loops.
 * - All math and pyramid layout are unchanged.
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>     /* malloc() */
#include <string.h>     /* memcpy(), memset() */
#include <math.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"   /* for computing pyramid */
#include "pyramid.h"

/* OpenACC */
#include <openacc.h>

/*********************************************************************
 *
 */

_KLT_Pyramid _KLTCreatePyramid(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels)
{
  _KLT_Pyramid pyramid;
  int nbytes = sizeof(_KLT_PyramidRec) +
    nlevels * sizeof(_KLT_FloatImage *) +
    nlevels * sizeof(int) +
    nlevels * sizeof(int);
  int i;

  if (subsampling != 2 && subsampling != 4 &&
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTCreatePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  /* Allocate memory for structure and set parameters */
  pyramid = (_KLT_Pyramid) malloc(nbytes);
  if (pyramid == NULL)
    KLTError("(_KLTCreatePyramid)  Out of memory");

  /* Set parameters */
  pyramid->subsampling = subsampling;
  pyramid->nLevels = nlevels;
  pyramid->img   = (_KLT_FloatImage *) (pyramid + 1);
  pyramid->ncols = (int *) (pyramid->img + nlevels);
  pyramid->nrows = (int *) (pyramid->ncols + nlevels);

  /* Allocate memory for each level of pyramid and assign pointers */
  for (i = 0 ; i < nlevels ; i++)  {
    pyramid->img[i]   = _KLTCreateFloatImage(ncols, nrows);
    pyramid->ncols[i] = ncols;
    pyramid->nrows[i] = nrows;
    ncols /= subsampling;
    nrows /= subsampling;
  }

  return pyramid;
}


/*********************************************************************
 *
 */

void _KLTFreePyramid(
  _KLT_Pyramid pyramid)
{
  int i;

  /* Free images */
  for (i = 0 ; i < pyramid->nLevels ; i++)
    _KLTFreeFloatImage(pyramid->img[i]);

  /* Free structure */
  free(pyramid);
}


/*********************************************************************
 * _KLTComputePyramid
 *
 * Builds a Gaussian pyramid:
 *   Level 0: copy of input img
 *   Level i>0: smoothed + subsampled version of previous level
 *
 * OpenACC:
 *   - The inner subsampling loop for each level is parallelized
 *     with a 2D (y,x) parallel loop.
 *********************************************************************/

void _KLTComputePyramid(
  _KLT_FloatImage img,
  _KLT_Pyramid pyramid,
  float sigma_fact)
{
  _KLT_FloatImage currimg, tmpimg;
  int ncols = img->ncols;
  int nrows = img->nrows;
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;  /* empirically determined */
  int oldncols, oldnrows;
  int i, x, y;

  if (subsampling != 2 && subsampling != 4 &&
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  assert(pyramid->ncols[0] == img->ncols);
  assert(pyramid->nrows[0] == img->nrows);

  /* Copy original image to level 0 of pyramid (CPU copy) */
  memcpy(pyramid->img[0]->data, img->data,
         (size_t)(ncols * nrows) * sizeof(float));

  currimg = img;

  /* Build subsequent levels */
  for (i = 1 ; i < pyramid->nLevels ; i++)  {
    /* Smooth the current image into a temporary buffer (CPU routine) */
    tmpimg = _KLTCreateFloatImage(ncols, nrows);
    _KLTComputeSmoothedImage(currimg, sigma, tmpimg);

    /* Subsample: compute size of new level */
    oldncols = ncols;
    oldnrows = nrows;
    ncols /= subsampling;
    nrows /= subsampling;

    /* OpenACC: parallelize subsampling over (y,x) for this level */
    {
      float *tmpdata  = tmpimg->data;
      float *pyrdata  = pyramid->img[i]->data;
      const int ss    = subsampling;
      const int shalf = subhalf;
      const int onc   = oldncols;
      const int nc    = ncols;
      const int nr    = nrows;

      #pragma acc parallel loop collapse(2) \
                       present(tmpdata, pyrdata) \
                       independent
      for (y = 0 ; y < nr ; y++) {
        for (x = 0 ; x < nc ; x++) {
          int src_y = ss * y + shalf;
          int src_x = ss * x + shalf;
          pyrdata[y*nc + x] =
            tmpdata[src_y * onc + src_x];
        }
      }
    }

    /* Next level uses the newly computed pyramid image */
    currimg = pyramid->img[i];

    _KLTFreeFloatImage(tmpimg);
  }
}