#include <imago/imago.h>

int main(int argc, char **argv) {
    int neuronsCount = 100;
    int synapsesDensity = 10;

    // Create network model.
    corticolumn column;
    dccol_init(&column, neuronsCount, synapsesDensity);


    double start_time = time_millis();
    for(int i = 0; i < 100000; i++) {
        // Feed the column and tick it.
        neurons_count_t inputNeuronsCount = 4;
        uint32_t input_neurons[] = {0, 1, 2, 3};
        if (random_float(0, 1) < 0.4f) {
            ccol_feed(&column, input_neurons, inputNeuronsCount, SYNAPSE_DEFAULT_VALUE);
        }
        ccol_tick(&column);
    }
    double end_time = time_millis();


    printf("\nExecution time: %.3f ms\n", end_time - start_time);
}