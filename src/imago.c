#include "imago.h"

void initColumn(struct Corticolumn* column, uint32_t neuronsNum) {
    uint32_t synapsesPerNeuron = 10;

    // Allocate neurons.
    column->neuronsNum = neuronsNum;
    column->neurons = (struct Neuron*) malloc(column->neuronsNum * sizeof(struct Neuron));

    // Allocate synapses.
    column->synapsesNum = synapsesPerNeuron * neuronsNum;
    column->synapses = (struct Synapse*) malloc(column->synapsesNum * sizeof(struct Synapse));

    // Initialize neurons with default values.
    for (uint32_t i = 0; i < neuronsNum; i++) {
        column->neurons[i].threshold = DEFAULT_THRESHOLD;
    }
}

void propagate(struct Corticolumn* column) {
    // Loop through synapses.
    for (uint32_t i = 0; i < column->synapsesNum; i++) {
        if (column->synapses[i].progress != SPIKE_DELIVERED &&
            column->synapses[i].progress < column->synapses[i].propagationTime) {
            // Increment progress if less than propagationTcolumnime and not alredy delicolumnvered.
            column->synapses[i].progress++;
        } else if (column->synapses[i].progress >= column->synapses[i].propagationTime) {
            // Set progress to SPIKE_DELIVERED if propagation time is reached.
        } else if (column->synapses[i].progress >= column->synapses[i].propagationTime) {
            column->synapses[i].progress = SPIKE_DELIVERED;
        }
    }
}

void increment(struct Corticolumn* column) {
    // Loop through neurons.
    for (uint32_t i = 0; i < column->neuronsNum; i++) {
        // Decrement value by decay rate.
        column->neurons[i].value -= DECAY_RATE;

        // Make sure it does not go below 0.
        if (column->neurons[i].value < 0) {
            column->neurons[i].value = 0;
        }

        // for (uint32_t j = 0; j < ) {

        // }
    }
}