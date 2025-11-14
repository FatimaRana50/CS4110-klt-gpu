/*********************************************************************
 * convolve.c — OpenACC version (computation preserved)
 *********************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

#define MAX_KERNEL_WIDTH 71

typedef struct {
    int width;
    float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

/*********************************************************************
 * _KLTToFloatImage
 *********************************************************************/
void _KLTToFloatImage(KLT_PixelType *img, int ncols, int nrows,
                       _KLT_FloatImage floatimg)
{
    KLT_PixelType *ptrend = img + ncols*nrows;
    float *ptrout = floatimg->data;

    assert(floatimg->ncols >= ncols);
    assert(floatimg->nrows >= nrows);

    floatimg->ncols = ncols;
    floatimg->nrows = nrows;

    while (img < ptrend)
        *ptrout++ = (float)*img++;
}

/*********************************************************************
 * _computeKernels (host only)
 *********************************************************************/
static void _computeKernels(float sigma, ConvolutionKernel *gauss,
                            ConvolutionKernel *gaussderiv)
{
    const float factor = 0.01f;
    int i;
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f;
    float max_gaussderiv = (float)(sigma*exp(-0.5f));

    assert(MAX_KERNEL_WIDTH % 2 == 1);
    assert(sigma >= 0.0);

    /* Compute kernels on host */
    for (i = -hw; i <= hw; i++) {
        float g = expf(-(float)(i*i)/(2.0f*sigma*sigma));
        gauss->data[i + hw] = g;
        gaussderiv->data[i + hw] = -i * g;
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gauss->data[i+hw]/max_gauss) < factor; i++, gauss->width -= 2);

    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gaussderiv->data[i+hw]/max_gaussderiv) < factor; i++, gaussderiv->width -= 2);

    if (gauss->width == MAX_KERNEL_WIDTH || gaussderiv->width == MAX_KERNEL_WIDTH)
        KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d too small for sigma %f",
                 MAX_KERNEL_WIDTH, sigma);

    /* Shift */
    int shift_g = (MAX_KERNEL_WIDTH - gauss->width)/2;
    int shift_d = (MAX_KERNEL_WIDTH - gaussderiv->width)/2;
    for (i=0; i<gauss->width; i++)
        gauss->data[i] = gauss->data[i + shift_g];
    for (i=0; i<gaussderiv->width; i++)
        gaussderiv->data[i] = gaussderiv->data[i + shift_d];

    /* Normalize */
    int hw2 = gaussderiv->width / 2;
    float den = 0.0f;
    for(i=0; i<gauss->width; i++) den += gauss->data[i];
    for(i=0; i<gauss->width; i++) gauss->data[i] /= den;

    den = 0.0f;
    for(i=-hw2; i<=hw2; i++) den -= i * gaussderiv->data[i+hw2];
    for(i=-hw2; i<=hw2; i++) gaussderiv->data[i+hw2] /= den;

    sigma_last = sigma;
}

/*********************************************************************
 * _KLTGetKernelWidths
 *********************************************************************/
void _KLTGetKernelWidths(float sigma, int *gauss_width, int *gaussderiv_width)
{
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    *gauss_width = gauss_kernel.width;
    *gaussderiv_width = gaussderiv_kernel.width;
}

/*********************************************************************
 * _convolveImageHoriz (OpenACC)
 *********************************************************************/
static void _convolveImageHoriz(_KLT_FloatImage imgin,
                                ConvolutionKernel kernel,
                                _KLT_FloatImage imgout)
{
    int nrows = imgin->nrows;
    int ncols = imgin->ncols;
    int radius = kernel.width/2;
    int N = nrows*ncols;

    /* Flatten kernel for safe device access */
    float kernel_data[MAX_KERNEL_WIDTH];
    for(int i=0;i<kernel.width;i++) kernel_data[i] = kernel.data[i];

    #pragma acc parallel loop copyin(imgin->data[0:N], kernel_data[0:MAX_KERNEL_WIDTH]) \
                             copyout(imgout->data[0:N])
    for (int j=0; j<nrows; j++) {
        int row_offset = j*ncols;

        // Left border
        for (int i=0; i<radius; i++)
            imgout->data[row_offset + i] = 0.0f;

        // Middle
        for (int i=radius; i<ncols-radius; i++) {
            float sum = 0.0f;
            int p = row_offset + i - radius;
            for (int k=kernel.width-1; k>=0; k--)
                sum += imgin->data[p + (kernel.width-1 - k)] * kernel_data[k];
            imgout->data[row_offset + i] = sum;
        }

        // Right border
        for (int i=ncols-radius; i<ncols; i++)
            imgout->data[row_offset + i] = 0.0f;
    }
}

/*********************************************************************
 * _convolveImageVert (OpenACC)
 *********************************************************************/
static void _convolveImageVert(_KLT_FloatImage imgin,
                               ConvolutionKernel kernel,
                               _KLT_FloatImage imgout)
{
    int nrows = imgin->nrows;
    int ncols = imgin->ncols;
    int radius = kernel.width/2;
    int N = nrows*ncols;

    float kernel_data[MAX_KERNEL_WIDTH];
    for(int i=0;i<kernel.width;i++) kernel_data[i] = kernel.data[i];

    #pragma acc parallel loop copyin(imgin->data[0:N], kernel_data[0:MAX_KERNEL_WIDTH]) \
                             copyout(imgout->data[0:N])
    for (int i=0; i<ncols; i++) {
        // Top border
        for (int j=0; j<radius; j++)
            imgout->data[j*ncols + i] = 0.0f;

        // Middle
        for (int j=radius; j<nrows-radius; j++) {
            float sum = 0.0f;
            int p = (j-radius)*ncols + i;
            for (int k=kernel.width-1; k>=0; k--) {
                sum += imgin->data[p + (kernel.width-1-k)*ncols] * kernel_data[k];
            }
            imgout->data[j*ncols + i] = sum;
        }

        // Bottom border
        for (int j=nrows-radius; j<nrows; j++)
            imgout->data[j*ncols + i] = 0.0f;
    }
}

/*********************************************************************
 * _convolveSeparate
 *********************************************************************/
static void _convolveSeparate(_KLT_FloatImage imgin,
                              ConvolutionKernel horiz_kernel,
                              ConvolutionKernel vert_kernel,
                              _KLT_FloatImage imgout)
{
    _KLT_FloatImage tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
    int tmpN = tmpimg->ncols * tmpimg->nrows;

    #pragma acc enter data create(tmpimg->data[0:tmpN])

    _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
    _convolveImageVert(tmpimg, vert_kernel, imgout);

    #pragma acc exit data delete(tmpimg->data[0:tmpN])
    _KLTFreeFloatImage(tmpimg);
}

/*********************************************************************
 * _KLTComputeGradients
 *********************************************************************/
void _KLTComputeGradients(_KLT_FloatImage img,
                          float sigma,
                          _KLT_FloatImage gradx,
                          _KLT_FloatImage grady)
{
    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
    _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
}

/*********************************************************************
 * _KLTComputeSmoothedImage
 *********************************************************************/
void _KLTComputeSmoothedImage(_KLT_FloatImage img,
                              float sigma,
                              _KLT_FloatImage smooth)
{
    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}