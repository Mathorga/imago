#include "imago_cuda.h"
#include <math.h>
#include <stdio.h>

void braph_init(braph_t* braph, neurons_count_t neurons_count) {
    dbraph_init(braph, neurons_count, 10);
}

void dbraph_init(braph_t* braph, neurons_count_t neurons_count, uint16_t synapses_density) {
    synapses_count_t synapses_count = neurons_count * synapses_density;

    // Allocate neurons.
    braph->neurons_count = neurons_count;
    cudaMalloc((void**) &(braph->neurons), neurons_count * sizeof(neuron_t));
    CUDA_CHECK_ERROR();
    neuron_t* tmp_neurons = (neuron_t*) malloc(neurons_count * sizeof(neuron_t));

    // Initialize neurons with default values.
    for (neurons_count_t i = 0; i < neurons_count; i++) {
        tmp_neurons[i].threshold = NEURON_DEFAULT_THRESHOLD;
        tmp_neurons[i].value = NEURON_STARTING_VALUE;
        tmp_neurons[i].activity = 0;
    }

    // Copy neuron values to device.
    cudaMemcpy(braph->neurons, tmp_neurons, neurons_count * sizeof(neuron_t), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    free(tmp_neurons);

    // Allocate synapses.
    braph->synapses_count = synapses_count;
    cudaMalloc((void**) &(braph->synapses), synapses_count * sizeof(synapse_t));
    CUDA_CHECK_ERROR();
    synapse_t* tmp_synapses = (synapse_t*) malloc(synapses_count * sizeof(synapse_t));

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
    cudaMemcpy(braph->synapses, tmp_synapses, synapses_count * sizeof(synapse_t), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    free(tmp_synapses);

    // Allocate spikes.
    braph->spikes_count = 0;
    cudaMalloc((void**) &(braph->spikes), MAX_SPIKES_COUNT * sizeof(spike_t));
    CUDA_CHECK_ERROR();
    braph->traveling_spikes_count = 0;
    cudaMalloc((void**) &(braph->traveling_spikes), MAX_SPIKES_COUNT * sizeof(spike_t));
    CUDA_CHECK_ERROR();
}

void braph_feed(braph_t* braph, neurons_count_t starting_index, neurons_count_t count, neuron_value_t value) {
    // Copy neurons to host.
    neuron_t* tmp_neurons = (neuron_t*) malloc(count * sizeof(neuron_t));
    cudaMemcpy(tmp_neurons, braph->neurons + starting_index, count * sizeof(neuron_t), cudaMemcpyDeviceToHost);

    if (count > braph->neurons_count) {
        // TODO Handle error.
        return;
        free(tmp_neurons);
    }

    for (uint32_t i = starting_index; i < count; i++) {
        tmp_neurons[i].value += value;
    }

    // Copy neurons back to device.
    cudaMemcpy(braph->neurons + starting_index, tmp_neurons, count * sizeof(neuron_t), cudaMemcpyHostToDevice);

    free(tmp_neurons);
}

__global__ void braph_propagate(spike_t* spikes, synapse_t* synapses) {
    // Retrieve current spike.
    spike_t* current_spike = &(spikes[IDX2D(threadIdx.x, blockIdx.x, blockDim.x)]);

    // Retrieve reference synapse.
    synapse_t* reference_synapse = &(synapses[current_spike->synapse]);

    if (current_spike->progress < reference_synapse->propagation_time &&
        current_spike->progress != SPIKE_DELIVERED) {
        // Increment progress if less than propagation time and not alredy delivered.
        current_spike->progress++;
    } else if (current_spike->progress >= reference_synapse->propagation_time) {
        // Set progress to SPIKE_DELIVERED if propagation time is reached.
        current_spike->progress = SPIKE_DELIVERED;
    }
}

__global__ void braph_increment(spike_t* spikes,
                                synapse_t* synapses,
                                neuron_t* neurons,
                                spike_t* traveling_spikes,
                                spikes_count_t* traveling_spikes_count,
                                spikes_count_t spikes_count) {
    extern __shared__ neuron_value_t traveling_spikes_adds[];

    // Cut exceeding threads.
    spikes_count_t spike_id = IDX2D(threadIdx.x, blockIdx.x, blockDim.x);
    if (spike_id >= spikes_count) {
        return;
    }

    spike_t* current_spike = &(spikes[spike_id]);

    if (current_spike->progress == SPIKE_DELIVERED) {
        // Increment target neuron.
        synapse_t* reference_synapse = &(synapses[spikes[IDX2D(threadIdx.x, blockIdx.x, blockDim.x)].synapse]);
        neuron_t* target_neuron = &(neurons[reference_synapse->output_neuron]);

        // atomicAdd((uint32_t*) &(target_neuron->value), (uint32_t) reference_synapse->value);
        target_neuron->value += reference_synapse->value;
        traveling_spikes_adds[threadIdx.x] = 0;
    } else {
        // Save the spike as traveling.
        traveling_spikes_adds[threadIdx.x] = 1;
    }

    __syncthreads();

    // Reduce all adds on the first thread of the block.
    if (threadIdx.x == blockIdx.x) {
        for (int i = 0; i < blockDim.x; i++) {
            // atomicAdd(traveling_spikes_count, traveling_spikes_adds[threadIdx.x]);
            (*traveling_spikes_count) += traveling_spikes_adds[threadIdx.x];

            if (traveling_spikes_adds[threadIdx.x]) {
                traveling_spikes[(*traveling_spikes_count) - 1] = *current_spike;
            }
        }
    }
}

__global__ void braph_decay(neuron_t* neurons) {
    // Retrieve current neuron.
    neuron_t* current_neuron = &(neurons[IDX2D(threadIdx.x, blockIdx.x, blockDim.x)]);

    // Make sure the neuron value does not go below 0.
    if (current_neuron->value > 0) {
        // Decrement value by decay rate.
        current_neuron->value -= NEURON_DECAY_RATE;
    } else if (current_neuron->value < 0) {
        current_neuron->value += NEURON_DECAY_RATE;
    }
}

__global__ void braph_fire(neuron_t* neurons, spike_t* spikes, synapse_t* synapses, spikes_count_t* spikes_count) {
    extern __shared__ spikes_count_t spikes_adds[];

    neuron_t* input_neuron = &(neurons[synapses[IDX2D(threadIdx.x, blockIdx.x, blockDim.x)].input_neuron]);

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

__global__ void braph_relax(neuron_t* neurons) {
    neuron_t* current_neuron = &(neurons[IDX2D(threadIdx.x, blockIdx.x, blockDim.x)]);
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

void braph_tick(braph_t* braph) {
    // Update synapses.
    if (braph->spikes_count > 0) {

        int blocks_count = ceil((float) braph->spikes_count / 1024);
        blocks_count = blocks_count <= 0 ? 1 : blocks_count;
        int threads_count = ceil((float) braph->spikes_count / blocks_count);
        braph_propagate<<<blocks_count, threads_count>>>(braph->spikes, braph->synapses);
        CUDA_CHECK_ERROR();
    }

    // Update neurons with spikes data.
    if (braph->spikes_count > 0) {
        // Copy spikes count to device.
        spikes_count_t* traveling_spikes_count;
        cudaMalloc((void**) &traveling_spikes_count, sizeof(spikes_count_t));
        CUDA_CHECK_ERROR();
        cudaMemcpy(traveling_spikes_count, &(braph->traveling_spikes_count), sizeof(spikes_count_t), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        int blocks_count = ceil((float) braph->spikes_count / 1024);
        blocks_count = blocks_count <= 0 ? 1 : blocks_count;
        int threads_count = ceil((float) braph->spikes_count / blocks_count);
        braph_increment<<<blocks_count, threads_count, threads_count * sizeof(neuron_value_t)>>>(braph->spikes,
                                                                                                       braph->synapses,
                                                                                                       braph->neurons,
                                                                                                       braph->traveling_spikes,
                                                                                                       traveling_spikes_count,
                                                                                                       braph->spikes_count);
        CUDA_CHECK_ERROR();

        // Copy back to host.
        cudaMemcpy(&(braph->traveling_spikes_count), traveling_spikes_count, sizeof(spikes_count_t), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR();
        cudaFree(traveling_spikes_count);
        CUDA_CHECK_ERROR();
    }

    // Apply decay to all neurons.
    if (braph->neurons_count > 0) {
        braph_decay<<<1, braph->neurons_count>>>(braph->neurons);
        CUDA_CHECK_ERROR();
    }

    // Fire neurons.
    if (braph->synapses_count > 0) {
        // Copy spikes count to device.
        spikes_count_t* spikes_count;
        cudaMalloc((void**) &spikes_count, sizeof(spikes_count_t));
        CUDA_CHECK_ERROR();
        cudaMemcpy(spikes_count, &(braph->spikes_count), sizeof(spikes_count_t), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        // Launch kernel.
        braph_fire<<<1, braph->synapses_count, braph->synapses_count * sizeof(spikes_count_t)>>>(braph->neurons, braph->spikes, braph->synapses, spikes_count);
        CUDA_CHECK_ERROR();

        // Copy back to host.
        cudaMemcpy(&(braph->spikes_count), spikes_count, sizeof(spikes_count_t), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR();
        cudaFree(spikes_count);
        CUDA_CHECK_ERROR();
    }

    // // Relax neuron values.
    if (braph->neurons_count > 0) {
        braph_relax<<<1, braph->neurons_count>>>(braph->neurons);
        CUDA_CHECK_ERROR();
    }
}


// ONLY FOR DEBUG PURPOSES, REMOVE WHEN NOT NEEDED ANYMORE.
void braph_copy_to_host(braph_t* braph) {
    neuron_t* neurons = (neuron_t*) malloc(braph->neurons_count * sizeof(neuron_t));
    cudaMemcpy(neurons, braph->neurons, braph->neurons_count * sizeof(neuron_t), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();

    cudaFree(braph->neurons);
    CUDA_CHECK_ERROR();

    braph->neurons = neurons;
}