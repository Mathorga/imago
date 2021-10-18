#include "imago_cuda.h"

__device__ void ccol_feed(corticolumn* column, uint32_t* target_neurons, uint32_t targets_count, int8_t value) {
    if (threadIdx.x >= column->neurons_count) {
        // TODO Handle error.
        return;
    }

    column->neurons[target_neurons[threadIdx.x]].value += value;
}

__device__ void ccol_propagate(corticolumn* column) {
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