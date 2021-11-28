#include "imago_std.h"

void braph_init(braph_t* braph, uint32_t neurons_count) {
    dbraph_init(braph, neurons_count, 10);
}

void dbraph_init(braph_t* braph, uint32_t neurons_count, uint16_t synapses_density) {
    // Randomize seed.
    // Comment this if you need consistent result across multiple runs.
    // srand(time(NULL));

    // Allocate neurons.
    braph->neurons_count = neurons_count;
    braph->neurons = (neuron_t*) malloc(braph->neurons_count * sizeof(neuron_t));

    // Allocate synapses.
    braph->synapses_count = synapses_density * neurons_count;
    braph->synapses = (synapse_t*) malloc(braph->synapses_count * sizeof(synapse_t));

    // Initialize neurons with default values.
    for (uint32_t i = 0; i < braph->neurons_count; i++) {
        braph->neurons[i].threshold = NEURON_DEFAULT_THRESHOLD;
        braph->neurons[i].value = NEURON_STARTING_VALUE;
        braph->neurons[i].activity = 0;
    }

    // Initialize synapses with random values.
    for (uint32_t i = 0; i < braph->synapses_count; i++) {
        // Assign a random input neuron.
        int32_t random_input = rand() % braph->neurons_count;

        // Assign a random output neuron, different from the input.
        int32_t random_output;
        do {
            random_output = rand() % braph->neurons_count;
        } while (random_output == random_input);

        braph->synapses[i].input_neuron = random_input;
        braph->synapses[i].output_neuron = random_output;
        braph->synapses[i].propagation_time = SYNAPSE_MIN_PROPAGATION_TIME + (rand() % SYNAPSE_DEFAULT_PROPAGATION_TIME - SYNAPSE_MIN_PROPAGATION_TIME);
        braph->synapses[i].value = SYNAPSE_DEFAULT_VALUE;
    }

    // Initialize spikes data.
    braph->spikes_count = 0;
    braph->spikes = (spike_t*) malloc(0);
}

void braph_feed(braph_t* braph, neurons_count_t starting_index, neurons_count_t count, int8_t value) {
    if (count > braph->neurons_count) {
        // TODO Handle error.
        return;
    }

    for (uint32_t i = 0; i < count; i++) {
        braph->neurons[starting_index + i].value += value;
    }
}

void braph_propagate(braph_t* braph) {
    // Loop through spikes.
    for (spikes_count_t i = 0; i < braph->spikes_count; i++) {
        // Retrieve current spike.
        spike_t* current_spike = &(braph->spikes[i]);

        // Retrieve reference synapse.
        synapse_t* reference_synapse = &(braph->synapses[current_spike->synapse]);

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

void braph_increment(braph_t* braph) {
    uint32_t traveling_spikes_count = 0;
    spike_t* traveling_spikes = (spike_t*) malloc(traveling_spikes_count * sizeof(spike_t));

    // Loop through spikes.
    for (spikes_count_t i = 0; i < braph->spikes_count; i++) {
        if (braph->spikes[i].progress == SPIKE_DELIVERED) {
            // Increment target neuron.
            synapse_t* reference_synapse = &(braph->synapses[braph->spikes[i].synapse]);
            neuron_t* target_neuron = &(braph->neurons[reference_synapse->output_neuron]);

            target_neuron->value += reference_synapse->value;
        } else {
            // Save the spike as traveling.
            traveling_spikes_count++;
            traveling_spikes = (spike_t*) realloc(traveling_spikes, traveling_spikes_count * sizeof(spike_t));
            traveling_spikes[traveling_spikes_count - 1] = braph->spikes[i];
        }
    }

    // Reset spikes.
    free(braph->spikes);
    braph->spikes = traveling_spikes;
    braph->spikes_count = traveling_spikes_count;
}

void braph_decay(braph_t* braph) {
    // Loop through neurons.
    for (uint32_t i = 0; i < braph->neurons_count; i++) {
        // Retrieve current neuron.
        neuron_t* current_neuron = &(braph->neurons[i]);

        // Make sure the neuron value does not go below 0.
        if (current_neuron->value > 0) {
            // Decrement value by decay rate.
            current_neuron->value -= NEURON_DECAY_RATE;
        } else if (current_neuron->value < 0) {
            current_neuron->value += NEURON_DECAY_RATE;
        }
    }
}

void braph_fire(braph_t* braph) {
    // Loop through synapses and fire spikes on those whose input neuron's value exceeds their threshold.
    for (synapses_count_t i = 0; i < braph->synapses_count; i++) {
        neuron_t* input_neuron = &(braph->neurons[braph->synapses[i].input_neuron]);
        if (input_neuron->value > input_neuron->threshold) {
            // Create a new spike.
            braph->spikes_count++;
            braph->spikes = (spike_t*) realloc(braph->spikes, braph->spikes_count * sizeof(spike_t));

            braph->spikes[braph->spikes_count - 1].progress = 0;
            braph->spikes[braph->spikes_count - 1].synapse = i;
        }
    }
}

void braph_relax(braph_t* braph) {
    for (neurons_count_t i = 0; i < braph->neurons_count; i++) {
        neuron_t* current_neuron = &(braph->neurons[i]);
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

void braph_tick(braph_t* braph) {
    // Update synapses.
    braph_propagate(braph);

    // Update neurons with spikes data.
    braph_increment(braph);

    // Apply decay to all neurons.
    braph_decay(braph);

    // Fire neurons.
    braph_fire(braph);

    // Relax neuron values.
    braph_relax(braph);
}

void braph_syndel(braph_t* braph) {
    // Allocate tmp vector for synapses.
    synapses_count_t tmp_synapses_count = 0;
    synapse_t* tmp_synapses = (synapse_t*) malloc(tmp_synapses_count * sizeof(synapse_t));

    // Keep track of old indices in order to update them in related spikes.
    synapses_count_t* old_indices = (synapses_count_t*) malloc(tmp_synapses_count * sizeof(synapses_count_t));

    // Loop through synapses.
    for (synapses_count_t i = 0; i < braph->synapses_count; i++) {
        synapse_t* current_synapse = &(braph->synapses[i]);

        neuron_t* input_neuron = &(braph->neurons[current_synapse->input_neuron]);
        if (input_neuron->activity > SYNAPSE_DEL_THRESHOLD) {
            // Preserve synapse.
            tmp_synapses = (synapse_t*) realloc(tmp_synapses, (++tmp_synapses_count) * sizeof(synapse_t));
            old_indices = (synapses_count_t*) realloc(old_indices, tmp_synapses_count * sizeof(synapses_count_t));
            tmp_synapses[tmp_synapses_count - 1] = *current_synapse;
            old_indices[tmp_synapses_count - 1] = i;
        } else {
            if (rand() % 1000 > 1) {
                // Preserve synapse.
                tmp_synapses = (synapse_t*) realloc(tmp_synapses, (++tmp_synapses_count) * sizeof(synapse_t));
                old_indices = (synapses_count_t*) realloc(old_indices, tmp_synapses_count * sizeof(synapses_count_t));
                tmp_synapses[tmp_synapses_count - 1] = *current_synapse;
                old_indices[tmp_synapses_count - 1] = i;
            }
        }
    }

    // Update the braph with the new synapses.
    free(braph->synapses);
    braph->synapses = tmp_synapses;
    braph->synapses_count = tmp_synapses_count;

    // There should be no spike on a synapse that exceeded its lifespan, so there's no need to delete related spikes.
    // Synapses' indices change though, so spikes need to update their rereferences.
    for (spikes_count_t i = 0; i < braph->spikes_count; i++) {
        spike_t* current_spike = &(braph->spikes[i]);

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

void braph_syngen(braph_t* braph) {
    if (braph->synapses_count < SYNAPSES_COUNT_MAX) {
        // Loop through neurons.
        for (neurons_count_t i = 0; i < braph->neurons_count; i++) {
            neuron_t* current_neuron = &(braph->neurons[i]);
            if (current_neuron->activity > SYNAPSE_GEN_THRESHOLD &&
                current_neuron->activity % NEURON_ACTIVITY_STEP == 0) {
                // Create new synapse.
                braph->synapses_count++;
                braph->synapses = (synapse_t*) realloc(braph->synapses, braph->synapses_count * sizeof(synapse_t));

                // Assign a random output neuron, different from the input.
                neurons_count_t random_output;
                do {
                    random_output = rand() % braph->neurons_count;
                } while (random_output == i);

                braph->synapses[braph->synapses_count - 1].input_neuron = i;
                braph->synapses[braph->synapses_count - 1].output_neuron = random_output;
                braph->synapses[braph->synapses_count - 1].propagation_time = SYNAPSE_MIN_PROPAGATION_TIME + (rand() % SYNAPSE_DEFAULT_PROPAGATION_TIME - SYNAPSE_MIN_PROPAGATION_TIME);
                braph->synapses[braph->synapses_count - 1].value = SYNAPSE_DEFAULT_VALUE;
            }
        }
    }
}

void braph_evolve(braph_t* braph) {
    // Delete all unused synapses.
    braph_syndel(braph);

    // Add synapses to busy neurons.
    braph_syngen(braph);
}
