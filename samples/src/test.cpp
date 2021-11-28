#include <imago/imago.h>
#include <stdio.h>

int main(int argc, char **argv) {
    braph_t braph;

    printf("Started\n");

    braph_init(&braph, 10);

    printf("Initialized %zu\n", sizeof(neuron_t));

    braph_feed(&braph, 0, 4, SYNAPSE_DEFAULT_VALUE);

    braph_tick(&braph);
    
    printf("Ticked\n");

    braph_copy_to_host(&braph);
    
    printf("Copied back %d\n", braph.neurons[4].threshold);
}