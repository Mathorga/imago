#include <stdio.h>
#include "imago.h"

int main() {
    // Declare a Corticolumn variable.
    struct Corticolumn column;

    // Initialize the column.
    initColumn(&column, 100);

    printf("Neurons num %d\n", column.neuronsNum);
    printf("Synapses num %d\n", column.synapsesNum);

    printf("Size of Corticolumn: %zu\n", sizeof(uint32_t**));

    return 0;
}