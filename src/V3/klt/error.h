/*********************************************************************
 * error.h
 *********************************************************************/

#ifndef _ERROR_H_
#define _ERROR_H_

#ifdef __cplusplus
extern "C" {
#endif

void KLTError(const char *fmt, ...);
void KLTWarning(const char *fmt, ...);
void KLTPrintError(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* _ERROR_H_ */


