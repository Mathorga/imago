#include <imago/imago.h>

int main(int argc, char **argv) {
    int neuronsCount = 1000;
    int synapsesDensity = 10;

    // Create network model.
    braph_t braph;
    dbraph_init(&braph, neuronsCount, synapsesDensity);


    double start_time = time_millis();
    for(int i = 0; i < 10; i++) {
        // Feed the braph and tick it.
        neurons_count_t inputNeuronsCount = 4;
        neurons_count_t startingInputIndex = 0;
        if (random_float(0, 1) < 0.4f) {
            braph_feed(&braph, startingInputIndex, inputNeuronsCount, SYNAPSE_DEFAULT_VALUE);
        }
        braph_tick(&braph);
    }
    double end_time = time_millis();


    printf("\nExecution time: %.3f ms\n", end_time - start_time);
}