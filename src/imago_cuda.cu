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
    // Retrieve current spike.
    spike* current_spike = &(column->spikes[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)]);

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

__global__ void ccol_increment(corticolumn* column, spikes_count_t* traveling_spikes_count, spike** traveling_spikes) {
    if (column->spikes[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)].progress == SPIKE_DELIVERED) {
        // Increment target neuron.
        synapse* reference_synapse = &(column->synapses[column->spikes[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)].synapse]);
        neuron* target_neuron = &(column->neurons[reference_synapse->output_neuron]);

        target_neuron->value += reference_synapse->value;
    } else {
        // Save the spike as traveling.
        (*traveling_spikes_count)++;
        *(traveling_spikes[(*traveling_spikes_count) - 1]) = column->spikes[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)];
    }
}

__global__ void ccol_decay(corticolumn* column) {
    // Retrieve current neuron.
    neuron* current_neuron = &(column->neurons[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)]);

    // Make sure the neuron value does not go below 0.
    if (current_neuron->value > 0) {
        // Decrement value by decay rate.
        current_neuron->value -= NEURON_DECAY_RATE;
    } else if (current_neuron->value < 0) {
        current_neuron->value += NEURON_DECAY_RATE;
    }
}

__global__ void ccol_fire(corticolumn* column) {
    neuron* input_neuron = &(column->neurons[column->synapses[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)].input_neuron]);
    if (input_neuron->value > input_neuron->threshold) {
        // Create a new spike.
        column->spikes_count++;

        column->spikes[column->spikes_count - 1].progress = 0;
        column->spikes[column->spikes_count - 1].synapse = IDX2D(blockIdx.x, threadIdx.x, blockDim.x);
    }
}

__global__ void ccol_relax(corticolumn* column) {
    neuron* current_neuron = &(column->neurons[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)]);
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

void ccol_tick(corticolumn* column) {
    // TODO Copy corticolumn to device.
    // Allocate as much memory as possible, in order not to be forced to realloc.

    // Update synapses.
    ccol_propagate<<<1, column->spikes_count>>>(column);

    uint32_t traveling_spikes_count = 0;
    spike* traveling_spikes;
    cudaMalloc(&traveling_spikes, column->spikes_count * sizeof(spike));

    // Update neurons with spikes data.
    ccol_increment<<<1, column->spikes_count>>>(column, &traveling_spikes_count, traveling_spikes);

    // Reset spikes.
    cudaFree(column->spikes);
    column->spikes = traveling_spikes;
    column->spikes_count = traveling_spikes_count;

    // Apply decay to all neurons.
    ccol_decay<<<1, column->neurons_count>>>(column);

    // Fire neurons.
    ccol_fire<<<1, column->synapses_count>>>(column);

    // Relax neuron values.
    ccol_relax<<<1, column->neurons_count>>>(column);
}