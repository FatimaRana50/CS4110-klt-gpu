/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
next images in the dataset (frame_000320.pgm to frame_000600.pgm).
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>     // <-- added for timing
#include "pnmio.h"
#include "klt.h"

/* #define REPLACE */

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
  int startFrame = 320, endFrame = 600;
  int nFrames = endFrame - startFrame + 1;
  int ncols, nrows;
  int i, frame;

  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);
  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set to 2 to turn on affine consistency check */
 
  // Read the first image
  sprintf(fnamein, "images2/frame_%06d.pgm", startFrame);
  img1 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols * nrows * sizeof(unsigned char));

  /* ===============================
        START TIMING BEFORE WORK
     =============================== */
  clock_t start_clock = clock();
  /* =============================== */

  // Select good features in the first frame
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLTStoreFeatureList(fl, ft, 0);
  sprintf(fnameout, "feat_%06d.ppm", startFrame);
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, fnameout);

  // Track through remaining frames
  for (i = 1, frame = startFrame + 1; frame <= endFrame; i++, frame++) {
    sprintf(fnamein, "images2/frame_%06d.pgm", frame);
    pgmReadFile(fnamein, img2, &ncols, &nrows);

    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif

    KLTStoreFeatureList(fl, ft, i);
    sprintf(fnameout, "feat_%06d.ppm", frame);
    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);
  }

  /* ===============================
         END TIMING BEFORE WRITING
     =============================== */
  clock_t end_clock = clock();
  double total_time = (double)(end_clock - start_clock) / CLOCKS_PER_SEC;
  printf("\n[CPU] Feature selection + tracking runtime: %.3f s\n", total_time);
  /* =============================== */

  // Write out results (NOT included in timing)
  KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
  KLTWriteFeatureTable(ft, "features.ft", NULL);

  // Cleanup
  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);

  return 0;
}
