#include <stdio.h>
#include "imago.h"

int64_t get_time_millis() {
    struct timespec now;
    timespec_get(&now, TIME_UTC);
    return ((int64_t) now.tv_sec) * 1000 + ((int64_t) now.tv_nsec) / 1000000;
}

int main() {
    // Declare a corticolumn variable.
    struct corticolumn column;

    // Initialize the column.
    init_column(&column, 10000000);

    printf("Neurons num %d\n", column.neurons_num);
    printf("Synapses num %d\n", column.synapses_num);

    int64_t start_time = get_time_millis();
    for (uint32_t i = 0; i < 1; i++) {
        tick(&column);
    }
    int64_t end_time = get_time_millis();
    printf("Elapsed time: %ld ms\n", end_time - start_time);

    printf("%d\n", column.spikes[0].progress);

    return 0;
}