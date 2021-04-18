#include "imago.h"

void initColumn(struct Corticolumn* column, uint32_t neuronsNum) {
    column->index = 0;
    uint32_t synapsesNum = neuronsNum * 10;

    // Neuron data.
    column->neuronsNum = neuronsNum;
    column->neuronIndexes = (uint32_t*) malloc(sizeof(uint32_t) * neuronsNum);
    column->neuronValues = (int8_t*) malloc(sizeof(int8_t) * neuronsNum);
    column->neuronThresholds = (uint8_t*) malloc(sizeof(uint8_t) * neuronsNum);

    // Synapse data.
    column->synapsesNum = synapsesNum;
    column->synapseIndexes = (uint32_t*) malloc(sizeof(uint32_t) * synapsesNum);
    column->synapseValues = (int8_t*) malloc(sizeof(int8_t) * synapsesNum);
    column->synapsePropagationTimes = (uint8_t*) malloc(sizeof(uint8_t) * synapsesNum);
}

void propagate(uint8_t* propagationTimes, uint8_t* propagationProgresses) {
    
}