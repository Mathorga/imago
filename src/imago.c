#include "imago.h"

void initColumn(struct Corticolumn* column, uint32_t neuronsNum) {
    column->index = 0;

    // Define the density of the column by defining how many output synapses each neuron has.
    uint8_t synapsesPerNeuron = 10;
    uint32_t synapsesNum = neuronsNum * synapsesPerNeuron;

    // Neuron data.
    column->neuronsNum = neuronsNum;
    column->neuronIndexes = (uint32_t*) malloc(sizeof(uint32_t) * neuronsNum);
    column->neuronValues = (int8_t*) malloc(sizeof(int8_t) * neuronsNum);
    column->neuronThresholds = (uint8_t*) malloc(sizeof(uint8_t) * neuronsNum);
    column->neuronInputs = (uint32_t**) malloc(sizeof(uint32_t*) * neuronsNum);
    column->neuronOutputs = (uint32_t**) malloc(sizeof(uint32_t*) * neuronsNum);

    // Synapse data.
    column->synapsesNum = synapsesNum;
    column->synapseIndexes = (uint32_t*) malloc(sizeof(uint32_t) * synapsesNum);
    column->synapseValues = (int8_t*) malloc(sizeof(int8_t) * synapsesNum);
    column->synapsePropagationTimes = (uint8_t*) malloc(sizeof(uint8_t) * synapsesNum);
    column->spikeProgresses = (uint8_t*) malloc(sizeof(uint8_t) * synapsesNum);
}

void propagate(uint8_t* propagationTimes, uint32_t* indexes, uint8_t* progresses, uint32_t synapsesNum) {
    // Loop through synapses.
    for (uint32_t i = 0; i < synapsesNum; i++) {
        if (progresses[i] != SPIKE_DELIVERED && progresses[i] < propagationTimes[i]) {
            // Increment progress if less than propagationTime and not alredy delivered.
            progresses[i]++;
        } else if (progresses[i] >= propagationTimes[i]) {
            // Set progress to SPIKE_DELIVERED if propagation time is reached.
            progresses[i] = SPIKE_DELIVERED;
        }
    }
}

void increment(int8_t* neuronValues,
               uint32_t** neuronInputs,
               uint32_t* synapseIndexes,
               int8_t* synapseValues,
               uint8_t* spikeProgresses,
               uint32_t neuronsNum,
               uint32_t synapsesNum) {
    // Loop through neurons.
    for (uint32_t i = 0; i < neuronsNum; i++) {
        // Decrement value by decay rate.
        neuronValues[i] -= DECAY_RATE;

        // Make sure it does not go below 0.
        if (neuronValues[i] < 0) {
            neuronValues[i] = 0;
        }

        // for (uint32_t j = 0; j < ) {

        // }
    }
}