#include "imago_std.h"

void ccol_init(corticolumn* column, uint32_t neurons_count) {
    dccol_init(column, neurons_count, 10);
}

void dccol_init(corticolumn* column, uint32_t neurons_count, uint16_t synapses_density) {
    // Randomize seed.
    // Comment this if you need consistent result across multiple runs.
    // srand(time(NULL));

    // Allocate neurons.
    column->neurons_count = neurons_count;
    column->neurons = (neuron*) malloc(column->neurons_count * sizeof(neuron));

    // Allocate synapses.
    column->synapses_count = synapses_density * neurons_count;
    column->synapses = (synapse*) malloc(column->synapses_count * sizeof(synapse));

    // Initialize neurons with default values.
    for (uint32_t i = 0; i < column->neurons_count; i++) {
        column->neurons[i].threshold = NEURON_DEFAULT_THRESHOLD;
        column->neurons[i].value = NEURON_STARTING_VALUE;
        column->neurons[i].activity = 0;
    }

    // Initialize synapses with random values.
    for (uint32_t i = 0; i < column->synapses_count; i++) {
        // Assign a random input neuron.
        int32_t random_input = rand() % column->neurons_count;

        // Assign a random output neuron, different from the input.
        int32_t random_output;
        do {
            random_output = rand() % column->neurons_count;
        } while (random_output == random_input);

        column->synapses[i].input_neuron = random_input;
        column->synapses[i].output_neuron = random_output;
        column->synapses[i].propagation_time = SYNAPSE_MIN_PROPAGATION_TIME + (rand() % SYNAPSE_DEFAULT_PROPAGATION_TIME - SYNAPSE_MIN_PROPAGATION_TIME);
        column->synapses[i].value = SYNAPSE_DEFAULT_VALUE;
    }

    // Initialize spikes data.
    column->spikes_count = 0;
    column->spikes = (spike*) malloc(0);
}

void ccol_feed(corticolumn* column, uint32_t* target_neurons, uint32_t targets_count, int8_t value) {
    if (targets_count > column->neurons_count) {
        // TODO Handle error.
        return;
    }

    for (uint32_t i = 0; i < targets_count; i++) {
        column->neurons[target_neurons[i]].value += value;
    }
}

void ccol_propagate(corticolumn* column) {
    // Loop through spikes.
    for (spikes_count_t i = 0; i < column->spikes_count; i++) {
        // Retrieve current spike.
        spike* current_spike = &(column->spikes[i]);

        // Retrieve reference synapse.
        synapse* reference_synapse = &(column->synapses[current_spike->synapse]);

        if (current_spike->progress < reference_synapse->propagation_time &&
            current_spike->progress != SPIKE_DELIVERED) {
            // Increment progress if less than propagation time and not alredy delivered.
            current_spike->progress++;
        } else if (current_spike->progress >= reference_synapse->propagation_time) {
            // Set progress to SPIKE_DELIVERED if propagation time is reached.
            current_spike->progress = SPIKE_DELIVERED;
        }
    }
}

void ccol_increment(corticolumn* column) {
    uint32_t traveling_spikes_count = 0;
    spike* traveling_spikes = (spike*) malloc(traveling_spikes_count * sizeof(spike));

    // Loop through spikes.
    for (spikes_count_t i = 0; i < column->spikes_count; i++) {
        if (column->spikes[i].progress == SPIKE_DELIVERED) {
            // Increment target neuron.
            synapse* reference_synapse = &(column->synapses[column->spikes[i].synapse]);
            neuron* target_neuron = &(column->neurons[reference_synapse->output_neuron]);

            target_neuron->value += reference_synapse->value;
        } else {
            // Save the spike as traveling.
            traveling_spikes_count++;
            traveling_spikes = (spike*) realloc(traveling_spikes, traveling_spikes_count * sizeof(spike));
            traveling_spikes[traveling_spikes_count - 1] = column->spikes[i];
        }
    }

    // Reset spikes.
    free(column->spikes);
    column->spikes = traveling_spikes;
    column->spikes_count = traveling_spikes_count;
}

void ccol_decay(corticolumn* column) {
    // Loop through neurons.
    for (uint32_t i = 0; i < column->neurons_count; i++) {
        // Retrieve current neuron.
        neuron* current_neuron = &(column->neurons[i]);

        // Make sure the neuron value does not go below 0.
        if (current_neuron->value > 0) {
            // Decrement value by decay rate.
            current_neuron->value -= NEURON_DECAY_RATE;
        } else if (current_neuron->value < 0) {
            current_neuron->value += NEURON_DECAY_RATE;
        }
    }
}

void ccol_fire(corticolumn* column) {
    // Loop through synapses and fire spikes on those whose input neuron's value exceeds their threshold.
    for (synapses_count_t i = 0; i < column->synapses_count; i++) {
        neuron* input_neuron = &(column->neurons[column->synapses[i].input_neuron]);
        if (input_neuron->value > input_neuron->threshold) {
            // Create a new spike.
            column->spikes_count++;
            column->spikes = (spike*) realloc(column->spikes, column->spikes_count * sizeof(spike));

            column->spikes[column->spikes_count - 1].progress = 0;
            column->spikes[column->spikes_count - 1].synapse = i;
        }
    }
}

void ccol_relax(corticolumn* column) {
    for (neurons_count_t i = 0; i < column->neurons_count; i++) {
        neuron* current_neuron = &(column->neurons[i]);
        if (current_neuron->value > current_neuron->threshold) {
            // Set neuron value to recovery.
            current_neuron->value = NEURON_RECOVERY_VALUE;

            if (current_neuron->activity < SYNAPSE_GEN_THRESHOLD) {
                current_neuron->activity += NEURON_ACTIVITY_STEP;
            }
        } else {
            if (current_neuron->activity > 0) {
                current_neuron->activity--;
            }
        }
    }
}

void ccol_tick(corticolumn* column) {
    // Update synapses.
    ccol_propagate(column);

    // Update neurons with spikes data.
    ccol_increment(column);

    // Apply decay to all neurons.
    ccol_decay(column);

    // Fire neurons.
    ccol_fire(column);

    // Relax neuron values.
    ccol_relax(column);
}

void ccol_syndel(corticolumn* column) {
    // Allocate tmp vector for synapses.
    synapses_count_t tmp_synapses_count = 0;
    synapse* tmp_synapses = (synapse*) malloc(tmp_synapses_count * sizeof(synapse));

    // Keep track of old indices in order to update them in related spikes.
    synapses_count_t* old_indices = (synapses_count_t*) malloc(tmp_synapses_count * sizeof(synapses_count_t));

    // Loop through synapses.
    for (synapses_count_t i = 0; i < column->synapses_count; i++) {
        synapse* current_synapse = &(column->synapses[i]);

        neuron* input_neuron = &(column->neurons[current_synapse->input_neuron]);
        if (input_neuron->activity > SYNAPSE_DEL_THRESHOLD) {
            // Preserve synapse.
            tmp_synapses = (synapse*) realloc(tmp_synapses, (++tmp_synapses_count) * sizeof(synapse));
            old_indices = (synapses_count_t*) realloc(old_indices, tmp_synapses_count * sizeof(synapses_count_t));
            tmp_synapses[tmp_synapses_count - 1] = *current_synapse;
            old_indices[tmp_synapses_count - 1] = i;
        } else {
            if (rand() % 1000 > 1) {
                // Preserve synapse.
                tmp_synapses = (synapse*) realloc(tmp_synapses, (++tmp_synapses_count) * sizeof(synapse));
                old_indices = (synapses_count_t*) realloc(old_indices, tmp_synapses_count * sizeof(synapses_count_t));
                tmp_synapses[tmp_synapses_count - 1] = *current_synapse;
                old_indices[tmp_synapses_count - 1] = i;
            }
        }
    }

    // Update the column with the new synapses.
    free(column->synapses);
    column->synapses = tmp_synapses;
    column->synapses_count = tmp_synapses_count;

    // There should be no spike on a synapse that exceeded its lifespan, so there's no need to delete related spikes.
    // Synapses' indices change though, so spikes need to update their rereferences.
    for (spikes_count_t i = 0; i < column->spikes_count; i++) {
        spike* current_spike = &(column->spikes[i]);

        // Check if the current spike references any of the moved synapses.
        for (synapses_count_t j = 0; j < tmp_synapses_count; j++) {
            if (current_spike->synapse == old_indices[j]) {
                // Update the old index with the new one.
                current_spike->synapse = j;
            }
        }
    }

    free(old_indices);
}

void ccol_syngen(corticolumn* column) {
    if (column->synapses_count < SYNAPSES_COUNT_MAX) {
        // Loop through neurons.
        for (neurons_count_t i = 0; i < column->neurons_count; i++) {
            neuron* current_neuron = &(column->neurons[i]);
            if (current_neuron->activity > SYNAPSE_GEN_THRESHOLD &&
                current_neuron->activity % NEURON_ACTIVITY_STEP == 0) {
                // Create new synapse.
                column->synapses_count++;
                column->synapses = (synapse*) realloc(column->synapses, column->synapses_count * sizeof(synapse));

                // Assign a random output neuron, different from the input.
                neurons_count_t random_output;
                do {
                    random_output = rand() % column->neurons_count;
                } while (random_output == i);

                column->synapses[column->synapses_count - 1].input_neuron = i;
                column->synapses[column->synapses_count - 1].output_neuron = random_output;
                column->synapses[column->synapses_count - 1].propagation_time = SYNAPSE_MIN_PROPAGATION_TIME + (rand() % SYNAPSE_DEFAULT_PROPAGATION_TIME - SYNAPSE_MIN_PROPAGATION_TIME);
                column->synapses[column->synapses_count - 1].value = SYNAPSE_DEFAULT_VALUE;
            }
        }
    }
}

void ccol_evolve(corticolumn* column) {
    // Delete all unused synapses.
    ccol_syndel(column);

    // Add synapses to busy neurons.
    ccol_syngen(column);
}
