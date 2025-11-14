/*********************************************************************
 * base.h — CUDA/Thrust-safe version
 *********************************************************************/

#ifndef _BASE_H_
#define _BASE_H_

#ifndef uchar
#define uchar unsigned char
#endif
#ifndef schar
#define schar signed char
#endif
#ifndef uint
#define uint unsigned int
#endif
#ifndef ushort
#define ushort unsigned short
#endif
#ifndef ulong
#define ulong unsigned long
#endif

// --------------------------------------------------------------
// Safe max/min definitions that do NOT conflict with <algorithm>
// --------------------------------------------------------------

// If compiling under CUDA C++ or any C++11+ (like Thrust), use std::max/min
#if defined(__CUDACC__) || defined(__cplusplus)
  #include <algorithm>
  using std::max;
  using std::min;
#else
  // Only define macros when in pure C
  #ifndef max
  #define max(a,b) ((a) > (b) ? (a) : (b))
  #endif
  #ifndef min
  #define min(a,b) ((a) < (b) ? (a) : (b))
  #endif
#endif

#define max3(a,b,c) ((a) > (b) ? std::max((a),(c)) : std::max((b),(c)))
#define min3(a,b,c) ((a) < (b) ? std::min((a),(c)) : std::min((b),(c)))

#endif

/* COPY ENTIRE FILE FROM V1, THEN ADD AT THE END: */

/*********************************************************************
 * OpenACC Configuration
 *********************************************************************/

#ifdef _OPENACC
  #include <openacc.h>
  #define KLT_USE_OPENACC 1
#else
  #define KLT_USE_OPENACC 0
#endif

/* OpenACC utility macros */
#ifdef _OPENACC
  #define KLT_ACC_PARALLEL_LOOP _Pragma("acc parallel loop")
  #define KLT_ACC_DATA_COPYIN(ptr, size) _Pragma("acc data copyin(ptr[0:size])")
  #define KLT_ACC_UPDATE_HOST(ptr, size) _Pragma("acc update host(ptr[0:size])")
  #define KLT_ACC_UPDATE_DEVICE(ptr, size) _Pragma("acc update device(ptr[0:size])")
#else
  #define KLT_ACC_PARALLEL_LOOP
  #define KLT_ACC_DATA_COPYIN(ptr, size)
  #define KLT_ACC_UPDATE_HOST(ptr, size)
  #define KLT_ACC_UPDATE_DEVICE(ptr, size)
#endif

/* Performance tuning constants */
#define KLT_ACC_VECTOR_LENGTH 256
#define KLT_ACC_GANG_SIZE 32
