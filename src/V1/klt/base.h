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
