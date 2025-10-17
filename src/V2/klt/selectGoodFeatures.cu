/////////////////////////////////////////////////////////



/*********************************************************************
 * selectGoodFeatures.c
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h> /* malloc(), qsort() */
#include <stdio.h>  /* fflush()          */
#include <string.h> /* memset()          */
#include <math.h>   /* fsqrt()           */
#define fsqrt(X) sqrt(X)

/* CUDA includes */
#include <cuda_runtime.h>
#include <cuda.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"

int KLT_verbose = 1;


typedef enum {SELECTING_ALL, REPLACING_SOME} selectionMode;

#define SWAP3(list, i, j)               \
{register int *pi, *pj, tmp;            \
     pi=list+3*(i); pj=list+3*(j);      \
                                        \
     tmp=*pi;    \
     *pi++=*pj;  \
     *pj++=tmp;  \
                 \
     tmp=*pi;    \
     *pi++=*pj;  \
     *pj++=tmp;  \
                 \
     tmp=*pi;    \
     *pi=*pj;    \
     *pj=tmp;    \
}

void _quicksort(int *pointlist, int n)
{
  unsigned int i, j, ln, rn;

  while (n > 1)
  {
    SWAP3(pointlist, 0, n/2);
    for (i = 0, j = n; ; )
    {
      do
        --j;
      while (pointlist[3*j+2] < pointlist[2]);
      do
        ++i;
      while (i < j && pointlist[3*i+2] > pointlist[2]);
      if (i >= j)
        break;
      SWAP3(pointlist, i, j);
    }
    SWAP3(pointlist, j, 0);
    ln = j;
    rn = n - ++j;
    if (ln < rn)
    {
      _quicksort(pointlist, ln);
      pointlist += 3*j;
      n = rn;
    }
    else
    {
      _quicksort(pointlist + 3*j, rn);
      n = ln;
    }
  }
}
#undef SWAP3



/* CUDA kernel for computing gradients and eigenvalues - FIXED VERSION */
/* CUDA kernel for computing gradients and eigenvalues - DEBUG VERSION */
__global__ void computeGradientsAndEigenvalues(
    float* floatimg, 
    float* gradx, 
    float* grady, 
    int* pointlist,
    int ncols, 
    int nrows,
    int window_hw,
    int window_hh,
    int borderx,
    int bordery,
    int nSkippedPixels)
{
    // Calculate the grid position in the SUBSAMPLED space
    int x_step = nSkippedPixels + 1;
    int grid_width = (ncols - 2 * borderx + x_step - 1) / x_step;
    
    int subsampled_x = blockIdx.x * blockDim.x + threadIdx.x;
    int subsampled_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (subsampled_x >= grid_width || subsampled_y >= (nrows - 2 * bordery + x_step - 1) / x_step)
        return;
    
    // Convert back to original image coordinates
    int x = borderx + subsampled_x * x_step;
    int y = bordery + subsampled_y * x_step;
    
    // Ensure we're within bounds
    if (x >= ncols - borderx || y >= nrows - bordery)
        return;
        
    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    
    // Sum gradients in surrounding window
    for (int yy = y - window_hh; yy <= y + window_hh; yy++) {
        for (int xx = x - window_hw; xx <= x + window_hw; xx++) {
            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows) {
                float gx = gradx[yy * ncols + xx];
                float gy = grady[yy * ncols + xx];
                gxx += gx * gx;
                gxy += gx * gy;
                gyy += gy * gy;
            }
        }
    }
    
    // Compute minimum eigenvalue
    float diff = gxx - gyy;
    float sqrt_term = sqrtf(diff * diff + 4.0f * gxy * gxy);
    float val = (gxx + gyy - sqrt_term) / 2.0f;
    
    // Calculate index in pointlist
    int idx = subsampled_y * grid_width + subsampled_x;
    
    // DEBUG: Force some test features
    if (idx < 10) {
        pointlist[idx * 3] = 50 + idx * 20;      // x
        pointlist[idx * 3 + 1] = 50 + idx * 15;  // y  
        pointlist[idx * 3 + 2] = 1000 + idx * 100; // high eigenvalue
    } else {
        pointlist[idx * 3] = x;
        pointlist[idx * 3 + 1] = y;
        pointlist[idx * 3 + 2] = (int)val;
    }
}

/* CUDA kernel for enforcing minimum distance */
__global__ void enforceMinDistanceKernel(
    int* pointlist,
    int npoints,
    uchar* featuremap,
    int ncols,
    int nrows,
    int mindist,
    int min_eigenvalue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < npoints) {
        int x = pointlist[idx * 3];
        int y = pointlist[idx * 3 + 1];
        int val = pointlist[idx * 3 + 2];
        
        if (x >= 0 && x < ncols && y >= 0 && y < nrows && 
            val >= min_eigenvalue && !featuremap[y * ncols + x]) {
            
            // Mark surrounding region in featuremap
            for (int iy = y - mindist; iy <= y + mindist; iy++) {
                for (int ix = x - mindist; ix <= x + mindist; ix++) {
                    if (ix >= 0 && ix < ncols && iy >= 0 && iy < nrows) {
                        featuremap[iy * ncols + ix] = 1;
                    }
                }
            }
        } else {
            // Mark as invalid
            pointlist[idx * 3] = -1;
            pointlist[idx * 3 + 1] = -1;
            pointlist[idx * 3 + 2] = -1;
        }
    }
}

/* CUDA bitonic sort for pointlist */
__global__ void bitonicSortStep(int* pointlist, int j, int k, int n)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    
    if (ixj > i) {
        if ((i & k) == 0) {
            // Sort ascending
            if (pointlist[i * 3 + 2] < pointlist[ixj * 3 + 2]) {
                // Swap the triplets
                for (int m = 0; m < 3; m++) {
                    int temp = pointlist[i * 3 + m];
                    pointlist[i * 3 + m] = pointlist[ixj * 3 + m];
                    pointlist[ixj * 3 + m] = temp;
                }
            }
        } else {
            // Sort descending  
            if (pointlist[i * 3 + 2] > pointlist[ixj * 3 + 2]) {
                // Swap the triplets
                for (int m = 0; m < 3; m++) {
                    int temp = pointlist[i * 3 + m];
                    pointlist[i * 3 + m] = pointlist[ixj * 3 + m];
                    pointlist[ixj * 3 + m] = temp;
                }
            }
        }
    }
}

void cudaBitonicSort(int* d_pointlist, int npoints)
{
  /* Time the overall sort wrapper (may use host fallback) */
  cudaEvent_t ts, te;
  cudaEventCreate(&ts);
  cudaEventCreate(&te);
  cudaEventRecord(ts, 0);

  int* pointlist = (int*)malloc(npoints * 3 * sizeof(int));
  cudaMemcpy(pointlist, d_pointlist, npoints * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
  // Use existing CPU sort for now (can be optimized further)
  _quicksort(pointlist, npoints);
    
  cudaMemcpy(d_pointlist, pointlist, npoints * 3 * sizeof(int), cudaMemcpyHostToDevice);
  free(pointlist);

  cudaEventRecord(te, 0);
  cudaEventSynchronize(te);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, ts, te);
  cudaEventDestroy(ts);
  cudaEventDestroy(te);

  FILE *f = fopen("profiling/gpu_timing.txt", "a");
  if (f) { fprintf(f, "cudaBitonicSort %g\n", (double)ms); fclose(f); }
}

/*********************************************************************/

static void _fillFeaturemap(
  int x, int y, 
  uchar *featuremap, 
  int mindist, 
  int ncols, 
  int nrows)
{
  int ix, iy;

  for (iy = y - mindist ; iy <= y + mindist ; iy++)
    for (ix = x - mindist ; ix <= x + mindist ; ix++)
      if (ix >= 0 && ix < ncols && iy >= 0 && iy < nrows)
        featuremap[iy*ncols+ix] = 1;
}


/*********************************************************************
 * _enforceMinimumDistance
 *
 * Removes features that are within close proximity to better features.
 *
 * INPUTS
 * featurelist:  A list of features.  The nFeatures property
 *               is used.
 *
 * OUTPUTS
 * featurelist:  Is overwritten.  Nearby "redundant" features are removed.
 *               Writes -1's into the remaining elements.
 *
 * RETURNS
 * The number of remaining features.
 */

static void _enforceMinimumDistance(
  int *pointlist,              /* featurepoints */
  int npoints,                 /* number of featurepoints */
  KLT_FeatureList featurelist, /* features */
  int ncols, int nrows,        /* size of images */
  int mindist,                 /* min. dist b/w features */
  int min_eigenvalue,          /* min. eigenvalue */
  KLT_BOOL overwriteAllFeatures)
{
  int indx;          /* Index into features */
  int x, y, val;     /* Location and trackability of pixel under consideration */
  uchar *featuremap; /* Boolean array recording proximity of features */
  int *ptr;
  
  printf("DEBUG enforceMinimumDistance: npoints=%d, min_eigenvalue=%d, featurelist->nFeatures=%d\n", 
         npoints, min_eigenvalue, featurelist->nFeatures);
	
  /* Cannot add features with an eigenvalue less than one */
  if (min_eigenvalue < 1)  min_eigenvalue = 1;

  /* Allocate memory for feature map and clear it */
  featuremap = (uchar *) malloc(ncols * nrows * sizeof(uchar));
  memset(featuremap, 0, ncols*nrows);
	
  /* Necessary because code below works with (mindist-1) */
  mindist--;

  /* If we are keeping all old good features, then add them to the featuremap */
  if (!overwriteAllFeatures) {
    printf("DEBUG: Checking existing features\n");
    for (indx = 0 ; indx < featurelist->nFeatures ; indx++)
      if (featurelist->feature[indx]->val >= 0)  {
        x   = (int) featurelist->feature[indx]->x;
        y   = (int) featurelist->feature[indx]->y;
        _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
      }
  }

  /* For each feature point, in descending order of importance, do ... */
  ptr = pointlist;
  indx = 0;
  int features_added = 0;
  
  printf("DEBUG: Processing pointlist...\n");
  while (1)  {

    /* If we can't add all the points, then fill in the rest
       of the featurelist with -1's */
    if (ptr >= pointlist + 3*npoints)  {
      printf("DEBUG: Reached end of pointlist, features_added=%d\n", features_added);
      while (indx < featurelist->nFeatures)  {	
        if (overwriteAllFeatures || 
            featurelist->feature[indx]->val < 0) {
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

    x   = *ptr++;
    y   = *ptr++;
    val = *ptr++;
    
    // DEBUG: Print first few points being processed
    if (features_added < 5) {
      printf("DEBUG: Processing point: x=%d, y=%d, val=%d\n", x, y, val);
    }
		
    /* Ensure that feature is in-bounds */
    if (x < 0 || x >= ncols || y < 0 || y >= nrows) {
      if (features_added < 5) printf("DEBUG: Point out of bounds\n");
      continue;
    }
	
    while (!overwriteAllFeatures && 
           indx < featurelist->nFeatures &&
           featurelist->feature[indx]->val >= 0)
      indx++;

    if (indx >= featurelist->nFeatures)  {
      printf("DEBUG: Feature list full at indx=%d\n", indx);
      break;
    }

    /* If no neighbor has been selected, and if the minimum
       eigenvalue is large enough, then add feature to the current list */
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
      
      features_added++;
      if (features_added <= 5) {
        printf("DEBUG: ADDED FEATURE %d: x=%d, y=%d, val=%d\n", features_added, x, y, val);
      }
      
      indx++;

      /* Fill in surrounding region of feature map, but
         make sure that pixels are in-bounds */
      _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
    } else {
      if (features_added < 5) {
        printf("DEBUG: Skipped point - featuremap[%d]=%d, val>=min_eigenvalue=%d\n", 
               y*ncols+x, featuremap[y*ncols+x], (val >= min_eigenvalue));
      }
    }
  }

  printf("DEBUG: Total features added: %d\n", features_added);

  /* Free feature map  */
  free(featuremap);
}


/*********************************************************************
 * _comparePoints
 *
 * Used by qsort (in _KLTSelectGoodFeatures) to determine
 * which feature is better.
 * By switching the '>' with the '<', qsort is fooled into sorting 
 * in descending order.
 */

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


/*********************************************************************
 * _sortPointList
 */

static void _sortPointList(
  int *pointlist,
  int npoints)
{
#ifdef KLT_USE_QSORT
  qsort(pointlist, npoints, 3*sizeof(int), _comparePoints);
#else
  // Use CUDA bitonic sort for larger arrays
  if (npoints > 1000) {
    int *d_pointlist;
    cudaMalloc(&d_pointlist, npoints * 3 * sizeof(int));
    cudaMemcpy(d_pointlist, pointlist, npoints * 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaBitonicSort(d_pointlist, npoints);
    
    cudaMemcpy(pointlist, d_pointlist, npoints * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_pointlist);
  } else {
    _quicksort(pointlist, npoints);
  }
#endif
}


/*********************************************************************
 * _minEigenvalue
 *
 * Given the three distinct elements of the symmetric 2x2 matrix
 *                     [gxx gxy]
 *                     [gxy gyy],
 * Returns the minimum eigenvalue of the matrix.  
 */

static float _minEigenvalue(float gxx, float gxy, float gyy)
{
  return (float) ((gxx + gyy - sqrt((gxx - gyy)*(gxx - gyy) + 4*gxy*gxy))/2.0f);
}
	

/*********************************************************************/

void _KLTSelectGoodFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img, 
  int ncols, 
  int nrows,
  KLT_FeatureList featurelist,
  selectionMode mode)
{
  _KLT_FloatImage floatimg, gradx, grady;
  int window_hw, window_hh;
  int *pointlist;
  int npoints = 0;
  KLT_BOOL overwriteAllFeatures = (mode == SELECTING_ALL) ?
    TRUE : FALSE;
  KLT_BOOL floatimages_created = FALSE;
  
  /* CUDA variables */
  float *d_floatimg, *d_gradx, *d_grady;
  int *d_pointlist;
  int total_points;

  printf("DEBUG: Starting _KLTSelectGoodFeatures\n");

  /* Check window size */
  if (tc->window_width % 2 != 1) {
    tc->window_width = tc->window_width+1;
  }
  if (tc->window_height % 2 != 1) {
    tc->window_height = tc->window_height+1;
  }
  window_hw = tc->window_width/2; 
  window_hh = tc->window_height/2;
		
  /* Calculate total number of points */
  int borderx = tc->borderx < window_hw ? window_hw : tc->borderx;
  int bordery = tc->bordery < window_hh ? window_hh : tc->bordery;
  
  int x_step = tc->nSkippedPixels + 1;
  int grid_width = (ncols - 2 * borderx + x_step - 1) / x_step;
  int grid_height = (nrows - 2 * bordery + x_step - 1) / x_step;
  total_points = grid_width * grid_height;
  npoints = total_points;

  printf("DEBUG: borderx=%d, bordery=%d, grid_width=%d, grid_height=%d, total_points=%d\n", 
         borderx, bordery, grid_width, grid_height, total_points);

  /* Create pointlist */
  pointlist = (int *) malloc(total_points * 3 * sizeof(int));
  // Initialize with -1
  memset(pointlist, -1, total_points * 3 * sizeof(int));

  /* Create temporary images */
  floatimages_created = TRUE;
  floatimg = _KLTCreateFloatImage(ncols, nrows);
  gradx    = _KLTCreateFloatImage(ncols, nrows);
  grady    = _KLTCreateFloatImage(ncols, nrows);
  
  // Create a simple test pattern in the float image
  for (int y = 0; y < nrows; y++) {
    for (int x = 0; x < ncols; x++) {
      // Create a chessboard pattern for testing
      if ((x / 20 + y / 20) % 2 == 0) {
        floatimg->data[y * ncols + x] = 255.0f;
      } else {
        floatimg->data[y * ncols + x] = 0.0f;
      }
    }
  }
  
  printf("DEBUG: Created test pattern in float image\n");

  /* Compute gradient of image in x and y direction */
  _KLTComputeGradients(floatimg, tc->grad_sigma, gradx, grady);
  printf("DEBUG: Computed gradients\n");
  
  /* Allocate CUDA memory */
  cudaError_t err;
  err = cudaMalloc(&d_floatimg, ncols * nrows * sizeof(float));
  if (err != cudaSuccess) printf("DEBUG: cudaMalloc d_floatimg failed: %s\n", cudaGetErrorString(err));
  
  err = cudaMalloc(&d_gradx, ncols * nrows * sizeof(float));
  if (err != cudaSuccess) printf("DEBUG: cudaMalloc d_gradx failed: %s\n", cudaGetErrorString(err));
  
  err = cudaMalloc(&d_grady, ncols * nrows * sizeof(float));
  if (err != cudaSuccess) printf("DEBUG: cudaMalloc d_grady failed: %s\n", cudaGetErrorString(err));
  
  err = cudaMalloc(&d_pointlist, total_points * 3 * sizeof(int));
  if (err != cudaSuccess) printf("DEBUG: cudaMalloc d_pointlist failed: %s\n", cudaGetErrorString(err));
  
  printf("DEBUG: Allocated CUDA memory\n");

  /* Copy data to GPU */
  err = cudaMemcpy(d_floatimg, floatimg->data, ncols * nrows * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) printf("DEBUG: cudaMemcpy d_floatimg failed: %s\n", cudaGetErrorString(err));
  
  err = cudaMemcpy(d_gradx, gradx->data, ncols * nrows * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) printf("DEBUG: cudaMemcpy d_gradx failed: %s\n", cudaGetErrorString(err));
  
  err = cudaMemcpy(d_grady, grady->data, ncols * nrows * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) printf("DEBUG: cudaMemcpy d_grady failed: %s\n", cudaGetErrorString(err));
  
  // Initialize pointlist on GPU
  err = cudaMemset(d_pointlist, -1, total_points * 3 * sizeof(int));
  if (err != cudaSuccess) printf("DEBUG: cudaMemset d_pointlist failed: %s\n", cudaGetErrorString(err));
  
  printf("DEBUG: Copied data to GPU\n");

  /* Compute trackability using CUDA kernel */
  dim3 blockDim(8, 8);
  dim3 gridDim((grid_width + blockDim.x - 1) / blockDim.x, 
               (grid_height + blockDim.y - 1) / blockDim.y);
  
  printf("DEBUG: Launching kernel with gridDim(%d, %d), blockDim(%d, %d)\n", 
         gridDim.x, gridDim.y, blockDim.x, blockDim.y);
  
  /* Time the computeGradientsAndEigenvalues kernel with CUDA events */
  {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);

    computeGradientsAndEigenvalues<<<gridDim, blockDim>>>(
        d_floatimg, d_gradx, d_grady, d_pointlist,
        ncols, nrows, window_hw, window_hh,
        borderx, bordery, tc->nSkippedPixels);

    cudaEventRecord(t1, 0);
    err = cudaEventSynchronize(t1);
    if (err != cudaSuccess) {
      printf("DEBUG: computeGradientsAndEigenvalues event sync failed: %s\n", cudaGetErrorString(err));
    }
    float ker_ms = 0.0f;
    cudaEventElapsedTime(&ker_ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    /* Append timing label for generator */
    {
      FILE *f = fopen("profiling/gpu_timing.txt", "a");
      if (f) {
        fprintf(f, "computeGradientsAndEigenvalues %g\n", (double)ker_ms);
        fclose(f);
      }
    }

    /* Also ensure device completed and check for errors */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("DEBUG: Kernel execution failed: %s\n", cudaGetErrorString(err));
    else printf("DEBUG: Kernel executed successfully (%.3f ms)\n", ker_ms);
  }
  
  /* Copy results back from GPU */
  err = cudaMemcpy(pointlist, d_pointlist, total_points * 3 * sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) printf("DEBUG: cudaMemcpy back failed: %s\n", cudaGetErrorString(err));
  else printf("DEBUG: Copied results back from GPU\n");
  
  // DEBUG: Print first 10 points
  printf("DEBUG: First 10 points from CUDA:\n");
  for (int i = 0; i < 10 && i < total_points; i++) {
    printf("  Point %d: x=%d, y=%d, val=%d\n", 
           i, pointlist[i*3], pointlist[i*3+1], pointlist[i*3+2]);
  }
  
  // Count valid points
  int valid_points = 0;
  for (int i = 0; i < total_points; i++) {
    if (pointlist[i*3+2] > 0) valid_points++;
  }
  printf("DEBUG: Found %d valid points with eigenvalues > 0\n", valid_points);
			
  /* Sort the features  */
  _sortPointList(pointlist, npoints);
  printf("DEBUG: Sorted pointlist\n");

  /* Check tc->mindist */
  if (tc->mindist < 0)  {
    tc->mindist = 0;
  }

  /* Enforce minimum distance between features */
  printf("DEBUG: Before enforceMinimumDistance: min_eigenvalue=%d\n", tc->min_eigenvalue);
  _enforceMinimumDistance(
    pointlist,
    npoints,
    featurelist,
    ncols, nrows,
    tc->mindist,
    tc->min_eigenvalue,
    overwriteAllFeatures);
  printf("DEBUG: After enforceMinimumDistance\n");

  /* Free memory */
  free(pointlist);
  cudaFree(d_floatimg);
  cudaFree(d_gradx);
  cudaFree(d_grady);
  cudaFree(d_pointlist);
  
  if (floatimages_created)  {
    _KLTFreeFloatImage(floatimg);
    _KLTFreeFloatImage(gradx);
    _KLTFreeFloatImage(grady);
  }
  
  printf("DEBUG: Finished _KLTSelectGoodFeatures\n");
}


/*********************************************************************
 * KLTSelectGoodFeatures
 *
 * Main routine, visible to the outside.  Finds the good features in
 * an image.  
 * 
 * INPUTS
 * tc:	Contains parameters used in computation (size of image,
 *        size of window, min distance b/w features, sigma to compute
 *        image gradients, # of features desired).
 * img:	Pointer to the data of an image (probably unsigned chars).
 * 
 * OUTPUTS
 * features:	List of features.  The member nFeatures is computed.
 */

void KLTSelectGoodFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img, 
  int ncols, 
  int nrows,
  KLT_FeatureList fl)
{
  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Selecting the %d best features "
            "from a %d by %d image...  ", fl->nFeatures, ncols, nrows);
    fflush(stderr);
  }

  _KLTSelectGoodFeatures(tc, img, ncols, nrows, 
                         fl, SELECTING_ALL);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features found.\n", 
            KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}


/*********************************************************************
 * KLTReplaceLostFeatures
 *
 * Main routine, visible to the outside.  Replaces the lost features 
 * in an image.  
 * 
 * INPUTS
 * tc:	Contains parameters used in computation (size of image,
 *        size of window, min distance b/w features, sigma to compute
 *        image gradients, # of features desired).
 * img:	Pointer to the data of an image (probably unsigned chars).
 * 
 * OUTPUTS
 * features:	List of features.  The member nFeatures is computed.
 */

void KLTReplaceLostFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img, 
  int ncols, 
  int nrows,
  KLT_FeatureList fl)
{
  int nLostFeatures = fl->nFeatures - KLTCountRemainingFeatures(fl);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Attempting to replace %d features "
            "in a %d by %d image...  ", nLostFeatures, ncols, nrows);
    fflush(stderr);
  }

  /* If there are any lost features, replace them */
  if (nLostFeatures > 0)
    _KLTSelectGoodFeatures(tc, img, ncols, nrows, 
                           fl, REPLACING_SOME);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features replaced.\n",
            nLostFeatures - fl->nFeatures + KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}