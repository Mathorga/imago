#include "imago.h"

void initColumn(struct Corticolumn* column, uint32_t neuronsNum) {
    // Randomize seed.
    // srand(time(NULL));

    // Define synapses per neuron.
    uint32_t synapsesPerNeuron = 10;

    // Allocate neurons.
    column->neuronsNum = neuronsNum;
    column->neurons = (struct Neuron*) malloc(column->neuronsNum * sizeof(struct Neuron));

    // Allocate synapses.
    column->synapsesNum = synapsesPerNeuron * neuronsNum;
    column->synapses = (struct Synapse*) malloc(column->synapsesNum * sizeof(struct Synapse));

    // Initialize neurons with default values.
    for (uint32_t i = 0; i < column->neuronsNum; i++) {
        column->neurons[i].threshold = DEFAULT_THRESHOLD;
        column->neurons[i].value = STARTING_VALUE;
    }

    // Initialize synapses with random values.
    for (uint32_t i = 0; i < column->synapsesNum; i++) {
        // Assign a random input neuron.
        int32_t randomInput = rand() % column->neuronsNum;

        // Assign a random output neuron, different from the input.
        int32_t randomOutput;
        do {
            randomOutput = rand() % column->neuronsNum;
        } while (randomOutput == randomInput);

        column->synapses[i].inputNeuron = randomInput;
        column->synapses[i].outputNeuron = randomOutput;
        column->synapses[i].propagationTime = DEFAULT_PROPAGATION_TIME;
        column->synapses[i].progress = STARTING_PROGRESS;
        column->synapses[i].value = DEFAULT_VALUE;
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
        // Make sure the neuron value does not go below 0.
        if (column->neurons[i].value > 0) {
            // Decrement value by decay rate.
            column->neurons[i].value -= DECAY_RATE;
        }

        // Loop through synapses.
        for (uint32_t j = 0; j < column->synapsesNum; j++) {
            // Only increment neuron value if spike is delivered and synapse outputs to the current neuron.
            if (column->synapses[i].progress == SPIKE_DELIVERED && column->synapses[i].outputNeuron == i) {
                column->neurons[i].value += column->synapses[i].value;
            }
        }
    }
}