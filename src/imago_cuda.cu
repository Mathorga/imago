#include "imago_cuda.h"

void ccol_feed(corticolumn* column, uint32_t* target_neurons, uint32_t targets_count, int8_t value) {
    if (targets_count > column->neurons_count) {
        // TODO Handle error.
        return;
    }

    for (uint32_t i = 0; i < targets_count; i++) {
        column->neurons[target_neurons[i]].value += value;
    }
}

__global__ void ccol_propagate(corticolumn* column) {
    // Loop through spikes.
    for (uint32_t i = 0; i < column->spikes_count; i++) {
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

void ccol_tick(corticolumn* column) {
    // Update synapses.
    ccol_propagate<<<1, 1>>>(column);

    // Update neurons with spikes data.
    ccol_increment<<<1, 1>>>(column);

    // Apply decay to all neurons.
    ccol_decay<<<1, 1>>>(column);

    // Fire neurons.
    ccol_fire<<<1, 1>>>(column);
}