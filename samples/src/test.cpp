#include <imago/imago.h>
#include <stdio.h>

int main(int argc, char **argv) {
    braph_t column;
    // cudaMalloc(&column, sizeof(braph));

    printf("Started\n");

    braph_init(&column, 10);

    printf("Initialized %zu\n", sizeof(neuron_t));

    braph_feed(&column, 0, 4, SYNAPSE_DEFAULT_VALUE);

    braph_tick(&column);
    
    printf("Ticked\n");

    braph_copy_to_host(&column);
    
    printf("Copied back %d\n", column.neurons[4].threshold);
}