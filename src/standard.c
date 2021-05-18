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
    // Loop through spikes.
    for (uint32_t i = 0; i < column->spikes_num; i++) {
        // Retrieve current spike.
        struct spike* current_spike = &(column->spikes[i]);

        // Retrieve reference synapse.
        struct synapse* reference_synapse = &(column->synapses[current_spike->synapse]);

        if (current_spike->progress < reference_synapse->propagation_time &&
            current_spike->progress != SPIKE_DELIVERED) {
            // Increment progress if less than propagationTcolumnime and not alredy delivered.
            current_spike->progress++;
        } else if (current_spike->progress >= reference_synapse->propagation_time) {
            // Set progress to SPIKE_DELIVERED if propagation time is reached.
            current_spike->progress = SPIKE_DELIVERED;
        }
    }
}

void increment(struct corticolumn* column) {
    // Loop through neurons.
    for (uint32_t i = 0; i < column->neurons_num; i++) {
        // Retrieve current neuron.
        struct neuron* current_neuron = &(column->neurons[i]);

        // Make sure the neuron value does not go below 0.
        if (current_neuron->value > 0) {
            // Decrement value by decay rate.
            current_neuron->value -= DECAY_RATE;
        }

        // Loop through spikes.
        for (uint32_t j = 0; j < column->spikes_num; j++) {
            // Retrieve current spike.
            struct spike* current_spike = &(column->spikes[j]);

            // Retrieve reference synapse.
            struct synapse* reference_synapse = &(column->synapses[current_spike->synapse]);

            // Only increment neuron value if spike is delivered and synapse outputs to the current neuron.
            if (reference_synapse->output_neuron == i && current_spike->progress == SPIKE_DELIVERED) {
                current_neuron->value += reference_synapse->value;
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