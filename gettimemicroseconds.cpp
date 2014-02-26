/* gettimemicroseconds()
 *
 * Get current time of day in microseconds.
 *
 * stamp()
 *
 * Get the current time in nanoseconds
 *
 * Paul Kelly, Gheorghe-Teodor Bercea, Fabio Luporini - Imperial College London - 2014
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <stddef.h>
#include <sys/sysinfo.h>

unsigned long gettimemicroseconds()
{
  struct timeval t;
  unsigned long r;
  (void) gettimeofday(&t, NULL);
  r = t.tv_sec * 1000000;
  r += t.tv_usec;
  return r;
}

long stamp()
{
  struct timespec tv;
  long _stamp;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  _stamp = tv.tv_sec * 1000 * 1000 * 1000 + tv.tv_nsec;
  return _stamp;
}