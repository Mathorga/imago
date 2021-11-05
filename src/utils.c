#include "utils.h"

#define S_TO_MILLIS 1e3
#define S_TO_MICROS 1e6
#define S_TO_NANOS 1e9

float random_float(float min, float max) {
    float random = ((float) rand()) / ((float) RAND_MAX);

    float range = max - min;
    return (random * range) + min;
}

double time_nanos() {
    return ((double) clock()) / (CLOCKS_PER_SEC / S_TO_NANOS);
}
double time_micros() {
    return ((double) clock()) / (CLOCKS_PER_SEC / S_TO_MICROS);
}
double time_millis() {
    // struct timespec ts;
    // clock_gettime(CLOCK_MONOTONIC, &ts);
    // return ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;

    return ((double) clock()) / (CLOCKS_PER_SEC / S_TO_MILLIS);
}
double time_s() {
    // struct timespec ts;
    // clock_gettime(CLOCK_MONOTONIC, &ts);
    // return ts.tv_sec;

    return ((double) clock()) / CLOCKS_PER_SEC;
}