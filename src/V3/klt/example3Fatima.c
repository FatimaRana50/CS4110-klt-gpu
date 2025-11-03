/**********************************************************************
Tracks 500 best features across all frames in FatimaDataset (001.pgm–122.pgm)
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "pnmio.h"
#include "klt.h"

#ifdef WIN32
int RunExample3()
#else
int main()
#endif
{
    unsigned char *img1, *img2;
    char fnamein[200], fnameout[200];
    KLT_TrackingContext tc;
    KLT_FeatureList fl;
    KLT_FeatureTable ft;

    int nFeatures = 150;
    int startFrame = 1, endFrame = 164;
    int nFrames = endFrame - startFrame + 1;
    int ncols, nrows;
    int i, frame;

    /* --- Initialize KLT context and data structures --- */
    tc = KLTCreateTrackingContext();
    fl = KLTCreateFeatureList(nFeatures);
    ft = KLTCreateFeatureTable(nFrames, nFeatures);

    tc->sequentialMode = TRUE;
    tc->writeInternalImages = FALSE;
    tc->affineConsistencyCheck = -1;  /* set to 2 to enable affine consistency check */

    /* --- Read the first image --- */
    sprintf(fnamein, "fatimadataset2/%03d.pgm", startFrame);
    img1 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
    if (img1 == NULL) {
        fprintf(stderr, "Error: Unable to read %s\n", fnamein);
        return -1;
    }

    img2 = (unsigned char *) malloc(ncols * nrows * sizeof(unsigned char));
    if (img2 == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for img2\n");
        return -1;
    }

    /* --- Select good features in the first frame --- */
    KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
    KLTStoreFeatureList(fl, ft, 0);
    sprintf(fnameout, "feat_%03d.ppm", startFrame);
    KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, fnameout);

    /* --- Track features through remaining frames --- */
    for (i = 1, frame = startFrame + 1; frame <= endFrame; i++, frame++) {
        sprintf(fnamein, "fatimadataset2/%03d.pgm", frame);
        if (pgmReadFile(fnamein, img2, &ncols, &nrows) == NULL) {
            fprintf(stderr, "Warning: Skipping missing frame %s\n", fnamein);
            continue;
        }

        KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

#ifdef REPLACE
        KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif

        KLTStoreFeatureList(fl, ft, i);
        sprintf(fnameout, "feat_%03d.ppm", frame);
        KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);

        /* Swap pointers to reuse memory efficiently */
        unsigned char *tmp = img1;
        img1 = img2;
        img2 = tmp;
    }

    /* --- Write out results --- */
    KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
    KLTWriteFeatureTable(ft, "features.ft", NULL);

    /* --- Cleanup --- */
    KLTFreeFeatureTable(ft);
    KLTFreeFeatureList(fl);
    KLTFreeTrackingContext(tc);
    free(img1);
    free(img2);

    return 0;
}
