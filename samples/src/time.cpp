#include <imago/imago.h>

int main(int argc, char **argv) {
    int neuronsCount = 100;
    int synapsesDensity = 10;

    // Create network model.
    braph_t column;
    dbraph_init(&column, neuronsCount, synapsesDensity);


    double start_time = time_millis();
    for(int i = 0; i < 100000; i++) {
        // Feed the column and tick it.
        neurons_count_t inputNeuronsCount = 4;
        neurons_count_t startingInputIndex = 0;
        if (random_float(0, 1) < 0.4f) {
            braph_feed(&column, startingInputIndex, inputNeuronsCount, SYNAPSE_DEFAULT_VALUE);
        }
        braph_tick(&column);
    }
    double end_time = time_millis();


    printf("\nExecution time: %.3f ms\n", end_time - start_time);
}