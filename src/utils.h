#ifndef __IMAGO_UTILS__
#define __IMAGO_UTILS__

// Used to ensure time.h defines CLOCK_MONOTONIC.
#define _POSIX_C_SOURCE 199309L

#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Returns a random float between min and max.
/// \param min The lower bound of the provided range.
/// \param max The upper bound of the provided range.
/// \return A new normally distributed pseudo-random float inside the given range.
float random_float(float min, float max);

/// Returns the current time in nanoseconds since the epoch.
double time_nanos();

/// Returns the current time in microseconds since the epoch.
double time_micros();

/// Returns the current time in milliseconds since the epoch.
double time_millis();

/// Returns the current time in seconds since the epoch.
double time_s();

#ifdef __cplusplus
}
#endif

#endif