/*********************************************************************
 * error.c
 *
 * Error and warning messages, and system commands.
 *********************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "error.h"

void KLTError(const char *fmt, ...)
{
    va_list args;
    fprintf(stderr, "\n[KLT ERROR] ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    fflush(stderr);
    exit(1);
}



/*********************************************************************
 * KLTWarning
 * 
 * Prints a warning message.
 * 
 * INPUTS
 * exactly like printf
 */

void KLTWarning(const char *fmt, ...)
{
  va_list args;

  va_start(args, fmt);
  fprintf(stderr, "KLT Warning: ");
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  fflush(stderr);
  va_end(args);
}

