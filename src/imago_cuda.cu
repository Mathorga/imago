#include "imago_cuda.h"
#include "stdio.h"

void ccol_init(braph* column, neurons_count_t neurons_count) {
    dccol_init(column, neurons_count, 10);
}

void dccol_init(braph* column, neurons_count_t neurons_count, uint16_t synapses_density) {
    synapses_count_t synapses_count = neurons_count * synapses_density;

    // Allocate neurons.
    column->neurons_count = neurons_count;
    cudaMalloc((void**) &(column->neurons), neurons_count * sizeof(neuron));
    CUDA_CHECK_ERROR();
    neuron* tmp_neurons = (neuron*) malloc(neurons_count * sizeof(neuron));

    // Initialize neurons with default values.
    for (neurons_count_t i = 0; i < neurons_count; i++) {
        tmp_neurons[i].threshold = NEURON_DEFAULT_THRESHOLD;
        tmp_neurons[i].value = NEURON_STARTING_VALUE;
        tmp_neurons[i].activity = 0;
    }

    // Copy neuron values to device.
    cudaMemcpy(column->neurons, tmp_neurons, neurons_count * sizeof(neuron), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    free(tmp_neurons);

    // Allocate synapses.
    column->synapses_count = synapses_count;
    cudaMalloc((void**) &(column->synapses), synapses_count * sizeof(synapse));
    CUDA_CHECK_ERROR();
    synapse* tmp_synapses = (synapse*) malloc(synapses_count * sizeof(synapse));

    // Initialize synapses with random values.
    for (synapses_count_t i = 0; i < synapses_count; i++) {
        // Assign a random input neuron.
        int32_t random_input = rand() % neurons_count;

        // Assign a random output neuron, different from the input.
        int32_t random_output;
        do {
            random_output = rand() % neurons_count;
        } while (random_output == random_input);

        tmp_synapses[i].input_neuron = random_input;
        tmp_synapses[i].output_neuron = random_output;
        tmp_synapses[i].propagation_time = SYNAPSE_MIN_PROPAGATION_TIME + (rand() % SYNAPSE_DEFAULT_PROPAGATION_TIME - SYNAPSE_MIN_PROPAGATION_TIME);
        tmp_synapses[i].value = SYNAPSE_DEFAULT_VALUE;
    }

    // Copy synapse values to device.
    cudaMemcpy(column->synapses, tmp_synapses, synapses_count * sizeof(synapse), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    free(tmp_synapses);

    // Allocate spikes.
    column->spikes_count = 0;
    cudaMalloc((void**) &(column->spikes), MAX_SPIKES_COUNT * sizeof(spike));
    CUDA_CHECK_ERROR();
    column->traveling_spikes_count = 0;
    cudaMalloc((void**) &(column->traveling_spikes), MAX_SPIKES_COUNT * sizeof(spike));
    CUDA_CHECK_ERROR();
}

void ccol_feed(braph* column, neurons_count_t starting_index, neurons_count_t count, neuron_value_t value) {
    // Copy neurons to host.
    // neuron* tmpNeurons;
    // cudaMemcpy(tmpNeurons, column->neurons[starting_target]);

    // if (targets_count > column->neurons_count) {
    //     // TODO Handle error.
    //     return;
    // }

    // for (uint32_t i = 0; i < targets_count; i++) {
    //     column->neurons[target_neurons[i]].value += value;
    // }
}

__global__ void ccol_propagate(spike* spikes, synapse* synapses) {
    // Retrieve current spike.
    spike* current_spike = &(spikes[IDX2D(threadIdx.x, blockIdx.x, blockDim.x)]);

    // Retrieve reference synapse.
    synapse* reference_synapse = &(synapses[current_spike->synapse]);

    if (current_spike->progress < reference_synapse->propagation_time &&
        current_spike->progress != SPIKE_DELIVERED) {
        // Increment progress if less than propagation time and not alredy delivered.
        current_spike->progress++;
    } else if (current_spike->progress >= reference_synapse->propagation_time) {
        // Set progress to SPIKE_DELIVERED if propagation time is reached.
        current_spike->progress = SPIKE_DELIVERED;
    }
}

__global__ void ccol_increment(spike* spikes, synapse* synapses, neuron* neurons, spike* traveling_spikes, spikes_count_t* traveling_spikes_count) {
    extern __shared__ neuron_value_t traveling_spikes_adds[];

    spike* current_spike = &(spikes[IDX2D(threadIdx.x, blockIdx.x, blockDim.x)]);

    if (current_spike->progress == SPIKE_DELIVERED) {
        // Increment target neuron.
        synapse* reference_synapse = &(synapses[spikes[IDX2D(threadIdx.x, blockIdx.x, blockDim.x)].synapse]);
        neuron* target_neuron = &(neurons[reference_synapse->output_neuron]);

        atomicAdd((uint32_t*) &(target_neuron->value), (uint32_t) reference_synapse->value);
        traveling_spikes_adds[threadIdx.x] = 0;
    } else {
        // Save the spike as traveling.
        traveling_spikes_adds[threadIdx.x] = 1;
    }

    __syncthreads();

    // Reduce all adds on the first thread of the block.
    if (threadIdx.x == blockIdx.x) {
        for (int i = 0; i < blockDim.x; i++) {
            (*traveling_spikes_count) += traveling_spikes_adds[threadIdx.x];

            if (traveling_spikes_adds[threadIdx.x]) {
                traveling_spikes[(*traveling_spikes_count) - 1] = *current_spike;
            }
        }
    }
}

__global__ void ccol_decay(neuron* neurons) {
    // Retrieve current neuron.
    neuron* current_neuron = &(neurons[IDX2D(threadIdx.x, blockIdx.x, blockDim.x)]);

    // Make sure the neuron value does not go below 0.
    if (current_neuron->value > 0) {
        // Decrement value by decay rate.
        current_neuron->value -= NEURON_DECAY_RATE;
    } else if (current_neuron->value < 0) {
        current_neuron->value += NEURON_DECAY_RATE;
    }
}

__global__ void ccol_fire(neuron* neurons, spike* spikes, synapse* synapses, spikes_count_t* spikes_count) {
    extern __shared__ spikes_count_t spikes_adds[];

    neuron* input_neuron = &(neurons[synapses[IDX2D(threadIdx.x, blockIdx.x, blockDim.x)].input_neuron]);

    if (input_neuron->value > input_neuron->threshold) {
        // Create a new spike.
        spikes_adds[threadIdx.x] = 1;
    } else {
        spikes_adds[threadIdx.x] = 0;
    }

    __syncthreads();

    // Reduce all adds on the first thread of the block.
    if (threadIdx.x == blockIdx.x) {
        for (int i = 0; i < blockDim.x; i++) {
            (*spikes_count) += spikes_adds[i];

            // Init the new spike if present.
            if (spikes_adds[i]) {
                spikes[(*spikes_count) - 1].progress = 0;
                spikes[(*spikes_count) - 1].synapse = i;
            }
        }
    }
}

__global__ void ccol_relax(neuron* neurons) {
    neuron* current_neuron = &(neurons[IDX2D(threadIdx.x, blockIdx.x, blockDim.x)]);
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

void ccol_tick(braph* column) {
    // Update synapses.
    if (column->spikes_count > 0) {
        ccol_propagate<<<1, column->spikes_count>>>(column->spikes, column->synapses);
        CUDA_CHECK_ERROR();
    }

    // Update neurons with spikes data.
    if (column->spikes_count > 0) {
        // Copy spikes count to device.
        spikes_count_t* traveling_spikes_count;
        cudaMalloc((void**) &traveling_spikes_count, sizeof(spikes_count_t));
        CUDA_CHECK_ERROR();
        cudaMemcpy(traveling_spikes_count, &(column->traveling_spikes_count), sizeof(spikes_count_t), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        // ccol_increment<<<column->neurons_count, column->spikes_count>>>(column);
        ccol_increment<<<1, column->spikes_count>>>(column->spikes, column->synapses, column->neurons, column->traveling_spikes, traveling_spikes_count);
        CUDA_CHECK_ERROR();

        // Copy back to host.
        cudaMemcpy(&(column->traveling_spikes_count), traveling_spikes_count, sizeof(spikes_count_t), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR();
        cudaFree(traveling_spikes_count);
        CUDA_CHECK_ERROR();
    }

    // Apply decay to all neurons.
    if (column->neurons_count > 0) {
        ccol_decay<<<1, column->neurons_count>>>(column->neurons);
        CUDA_CHECK_ERROR();
    }

    // Fire neurons.
    if (column->synapses_count > 0) {
        // Copy spikes count to device.
        spikes_count_t* spikes_count;
        cudaMalloc((void**) &spikes_count, sizeof(spikes_count_t));
        CUDA_CHECK_ERROR();
        cudaMemcpy(spikes_count, &(column->spikes_count), sizeof(spikes_count_t), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        // Launch kernel.
        ccol_fire<<<1, column->synapses_count, column->synapses_count * sizeof(spikes_count_t)>>>(column->neurons, column->spikes, column->synapses, spikes_count);
        CUDA_CHECK_ERROR();

        // Copy back to host.
        cudaMemcpy(&(column->spikes_count), spikes_count, sizeof(spikes_count_t), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR();
        cudaFree(spikes_count);
        CUDA_CHECK_ERROR();
    }

    // // Relax neuron values.
    if (column->neurons_count > 0) {
        ccol_relax<<<1, column->neurons_count>>>(column->neurons);
        CUDA_CHECK_ERROR();
    }
}


// ONLY FOR DEBUG PURPOSES, REMOVE WHEN NOT NEEDED ANYMORE.
void ccol_copy_to_host(braph* column) {
    neuron* neurons = (neuron*) malloc(column->neurons_count * sizeof(neuron));
    cudaMemcpy(neurons, column->neurons, column->neurons_count * sizeof(neuron), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();

    cudaFree(column->neurons);
    CUDA_CHECK_ERROR();

    column->neurons = neurons;
}