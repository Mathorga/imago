#include <stdio.h>
#include "imago.h"

int64_t getTimeMillis() {
    struct timespec now;
    timespec_get(&now, TIME_UTC);
    return ((int64_t) now.tv_sec) * 1000 + ((int64_t) now.tv_nsec) / 1000000;
}

int main() {
    // Declare a Corticolumn variable.
    struct Corticolumn column;

    // Initialize the column.
    initColumn(&column, 10);

    printf("Neurons num %d\n", column.neuronsNum);
    printf("Synapses num %d\n", column.synapsesNum);
    
    // for (uint32_t i = 0; i < column.synapsesNum; i++) {
    //     printf("%d\n", column.synapses[i].progress);
    // }

    int64_t startTime = getTimeMillis();
    for (uint32_t i = 0; i < 100; i++) {
        propagate(&column);
        increment(&column);
    }
    int64_t endTime = getTimeMillis();
    printf("Elapsed time: %ld\n", endTime - startTime);

    printf("%d\n", column.synapses[0].progress);

    return 0;
}