#include <imago/imago.h>
#include <stdio.h>

int main(int argc, char **argv) {
    corticolumn column;
    // cudaMalloc(&column, sizeof(corticolumn));

    printf("Started\n");

    ccol_init(&column, 10);

    printf("Initialized\n");

    // ccol_feed(&column, input_neurons, 4, SYNAPSE_DEFAULT_VALUE);

    ccol_tick(&column);
    
    printf("Ticked\n");

    ccol_copy_to_host(&column);
    
    printf("Copied back %d\n", column.neurons[15].threshold);
}