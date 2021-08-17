#include <stdio.h>
#include "imago.h"

int64_t get_time_millis() {
    struct timespec now;
    timespec_get(&now, TIME_UTC);
    return ((int64_t) now.tv_sec) * 1000 + ((int64_t) now.tv_nsec) / 1000000;
}

int main() {
    // Declare a corticolumn variable.
    corticolumn column;

    // Initialize the column.
    dccol_init(&column, 1000000, 20);

    printf("Neurons num \t%d\n", column.neurons_count);
    printf("Synapses num \t%d\n", column.synapses_count);

    int64_t start_time = get_time_millis();
    for (uint32_t i = 0; i < 1; i++) {
        // Fire input neurons.
        

        // Run the model.
        ccol_tick(&column);

        // Read output neurons.
        // TODO
    }
    int64_t end_time = get_time_millis();
    printf("Elapsed time: %ld ms\n", end_time - start_time);

    printf("%d\n", column.spikes[0].progress);

    return 0;
}