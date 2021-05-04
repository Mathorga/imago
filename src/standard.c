#include "standard.h"

void init_column(struct corticolumn* column, uint32_t neurons_num) {
    // Randomize seed.
    // srand(time(NULL));

    // Define synapses per neuron.
    uint32_t synapses_per_neuron = 10;

    // Allocate neurons.
    column->neurons_num = neurons_num;
    column->neurons = (struct neuron*) malloc(column->neurons_num * sizeof(struct neuron));

    // Allocate synapses.
    column->synapses_num = synapses_per_neuron * neurons_num;
    column->synapses = (struct synapse*) malloc(column->synapses_num * sizeof(struct synapse));

    // Initialize neurons with default values.
    for (uint32_t i = 0; i < column->neurons_num; i++) {
        column->neurons[i].threshold = DEFAULT_THRESHOLD;
        column->neurons[i].value = STARTING_VALUE;
    }

    // Initialize synapses with random values.
    for (uint32_t i = 0; i < column->synapses_num; i++) {
        // Assign a random input neuron.
        int32_t randomInput = rand() % column->neurons_num;

        // Assign a random output neuron, different from the input.
        int32_t randomOutput;
        do {
            randomOutput = rand() % column->neurons_num;
        } while (randomOutput == randomInput);

        column->synapses[i].input_neuron = randomInput;
        column->synapses[i].output_neuron = randomOutput;
        column->synapses[i].propagation_time = DEFAULT_PROPAGATION_TIME;
        column->synapses[i].value = DEFAULT_VALUE;
    }
}

void propagate(struct corticolumn* column) {
    // Loop through synapses.
    for (uint32_t i = 0; i < column->synapses_num; i++) {
        if (column->synapses[i].progress != SPIKE_DELIVERED &&
            column->synapses[i].progress < column->synapses[i].propagation_time) {
            // Increment progress if less than propagationTcolumnime and not alredy delicolumnvered.
            column->synapses[i].progress++;
        } else if (column->synapses[i].progress >= column->synapses[i].propagation_time) {
            // Set progress to SPIKE_DELIVERED if propagation time is reached.
        } else if (column->synapses[i].progress >= column->synapses[i].propagation_time) {
            column->synapses[i].progress = SPIKE_DELIVERED;
        }
    }
}

void increment(struct corticolumn* column) {
    // Loop through neurons.
    for (uint32_t i = 0; i < column->neurons_num; i++) {
        // Make sure the neuron value does not go below 0.
        if (column->neurons[i].value > 0) {
            // Decrement value by decay rate.
            column->neurons[i].value -= DECAY_RATE;
        }

        // Loop through synapses.
        for (uint32_t j = 0; j < column->synapses_num; j++) {
            // Only increment neuron value if spike is delivered and synapse outputs to the current neuron.
            if (column->synapses[i].progress == SPIKE_DELIVERED && column->synapses[i].output_neuron == i) {
                column->neurons[i].value += column->synapses[i].value;
            }
        }
    }
}

void fire(struct corticolumn* column) {
    // Loop through neurons and fire those whose value exceeded their threshold.
    for (uint32_t i = 0; i < column->neurons_num; i++) {
        if (column->neurons[i].value > column->neurons[i].threshold) {
            // Loop through synapses and fire them.
        }
    }
}

void tick(struct corticolumn* column) {
    propagate(column);
    increment(column);
    fire(column);
}