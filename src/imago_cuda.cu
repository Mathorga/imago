#include "imago_cuda.h"
#include "stdio.h"

void ccol_init(corticolumn* column, neurons_count_t neurons_count) {
    dccol_init(column, neurons_count, 10);
}

void dccol_init(corticolumn* column, neurons_count_t neurons_count, uint16_t synapses_density) {
    synapses_count_t synapses_count = neurons_count * synapses_density;

    // Allocate neurons.
    column->neurons_count = neurons_count;
    cudaMalloc((void**) &(column->neuron_thresholds), neurons_count * sizeof(neuron_threshold_t));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**) &(column->neuron_values), neurons_count * sizeof(neuron_value_t));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**) &(column->neuron_activities), neurons_count * sizeof(neuron_activity_t));
    CUDA_CHECK_ERROR();
    neuron_threshold_t* h_neuron_thresholds = (neuron_threshold_t*) malloc(neurons_count * sizeof(neuron_threshold_t));
    neuron_value_t* h_neuron_values = (neuron_value_t*) malloc(neurons_count * sizeof(neuron_value_t));
    neuron_activity_t* h_neuron_activities = (neuron_activity_t*) malloc(neurons_count * sizeof(neuron_activity_t));

    // Initialize neurons with default values.
    for (neurons_count_t i = 0; i < neurons_count; i++) {
        h_neuron_thresholds[i] = NEURON_DEFAULT_THRESHOLD;
        h_neuron_values[i] = NEURON_STARTING_VALUE;
        h_neuron_activities[i] = 0;
    }

    // Copy neuron values to device.
    cudaMemcpy(column->neuron_thresholds, h_neuron_thresholds, neurons_count * sizeof(neuron_threshold_t), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(column->neuron_values, h_neuron_values, neurons_count * sizeof(neuron_value_t), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(column->neuron_activities, h_neuron_activities, neurons_count * sizeof(neuron_activity_t), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    free(h_neuron_thresholds);
    free(h_neuron_values);
    free(h_neuron_activities);

    // Allocate synapses.
    column->synapses_count = synapses_count;
    cudaMalloc((void**) &(column->synapse_propagation_times), synapses_count * sizeof(synapse_propagation_time_t));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**) &(column->synapse_values), synapses_count * sizeof(synapse_value_t));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**) &(column->synapse_input_neurons), synapses_count * sizeof(neurons_count_t));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**) &(column->synapse_output_neurons), synapses_count * sizeof(neurons_count_t));
    CUDA_CHECK_ERROR();
    synapse_propagation_time_t* h_synapse_propagation_times = (synapse_propagation_time_t*) malloc(synapses_count * sizeof(synapse_propagation_time_t));
    synapse_value_t* h_synapse_values = (synapse_value_t*) malloc(synapses_count * sizeof(synapse_value_t));
    neurons_count_t* h_synapse_input_neurons = (neurons_count_t*) malloc(synapses_count * sizeof(neurons_count_t));
    neurons_count_t* h_synapse_output_neurons = (neurons_count_t*) malloc(synapses_count * sizeof(neurons_count_t));

    // Initialize synapses with random values.
    for (synapses_count_t i = 0; i < synapses_count; i++) {
        // Assign a random input neuron.
        int32_t random_input = rand() % neurons_count;

        // Assign a random output neuron, different from the input.
        int32_t random_output;
        do {
            random_output = rand() % neurons_count;
        } while (random_output == random_input);

        h_synapse_propagation_times[i] = SYNAPSE_MIN_PROPAGATION_TIME + (rand() % SYNAPSE_DEFAULT_PROPAGATION_TIME - SYNAPSE_MIN_PROPAGATION_TIME);
        h_synapse_values[i] = SYNAPSE_DEFAULT_VALUE;
        h_synapse_input_neurons[i] = random_input;
        h_synapse_output_neurons[i] = random_output;
    }

    // Copy synapse values to device.
    cudaMemcpy(column->synapse_propagation_times, h_synapse_propagation_times, synapses_count * sizeof(synapse_propagation_time_t), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(column->synapse_values, h_synapse_values, synapses_count * sizeof(synapse_value_t), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(column->synapse_input_neurons, h_synapse_input_neurons, synapses_count * sizeof(neurons_count_t), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(column->synapse_output_neurons, h_synapse_output_neurons, synapses_count * sizeof(neurons_count_t), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    free(h_synapse_propagation_times);
    free(h_synapse_values);
    free(h_synapse_input_neurons);
    free(h_synapse_output_neurons);

    // Allocate spikes.
    column->spikes_count = 0;
    cudaMalloc((void**) &(column->spike_progresses), MAX_SPIKES_COUNT * sizeof(spike_progress_t));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**) &(column->spike_synapses), MAX_SPIKES_COUNT * sizeof(synapses_count_t));
    CUDA_CHECK_ERROR();
    column->traveling_spikes_count = 0;
    cudaMalloc((void**) &(column->traveling_spike_progresses), MAX_SPIKES_COUNT * sizeof(spike_progress_t));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**) &(column->traveling_spike_synapses), MAX_SPIKES_COUNT * sizeof(synapses_count_t));
    CUDA_CHECK_ERROR();
}

void ccol_feed(corticolumn* column, neurons_count_t* target_neurons, neurons_count_t targets_count, int8_t value) {
    if (targets_count > column->neurons_count) {
        // TODO Handle error.
        return;
    }

    for (uint32_t i = 0; i < targets_count; i++) {
        column->neuron_values[target_neurons[i]] += value;
    }
}

__global__ void ccol_propagate(spike_progress_t* spike_progresses, synapses_count_t* spike_synapses, synapse_propagation_time_t* synapse_propagation_times) {
    // Retrieve current spike data.
    spike_progress_t* progress = &(spike_progresses[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)]);

    // Synapse data is only read, not modified, so no pointer is needed.
    synapses_count_t synapse = spike_synapses[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)];
    synapse_propagation_time_t synapse_propagation_time = synapse_propagation_times[synapse];

    if (*progress < synapse_propagation_time &&
        *progress != SPIKE_DELIVERED) {
        // Increment progress if less than propagation time and not alredy delivered.
        (*progress)++;
    } else if (*progress >= synapse_propagation_time) {
        // Set progress to SPIKE_DELIVERED if propagation time is reached.
        (*progress) = SPIKE_DELIVERED;
    }
}

__global__ void ccol_increment(spike_progress_t* spike_progresses,
                               synapses_count_t* spike_synapses,
                               synapse_value_t* synapse_values,
                               neurons_count_t* synapse_output_neurons,
                               neuron_value_t* neuron_values,
                               spike_progress_t* traveling_spike_progresses,
                               synapses_count_t* traveling_spike_synapses,
                               spikes_count_t* traveling_spikes_count) {
    extern __shared__ neuron_value_t traveling_spikes_adds[];

    spike_progress_t* progress = &(spike_progresses[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)]);
    synapses_count_t synapse = spike_synapses[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)];

    if ((*progress) == SPIKE_DELIVERED) {
        // Increment target neuron.
        synapse_value_t synapse_value = synapse_values[synapse];
        neurons_count_t synapse_output_neuron = synapse_output_neurons[synapse];
        neuron_value_t* neuron_value = &(neuron_values[synapse_output_neuron]);

        atomicAdd((uint32_t*) &(neuron_value), (uint32_t) synapse_value);
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
                traveling_spike_progresses[(*traveling_spikes_count) - 1] = *progress;
                traveling_spike_synapses[(*traveling_spikes_count) - 1] = synapse;
            }
        }
    }
}

__global__ void ccol_decay(neuron_value_t* neuron_values) {
    // Retrieve current neuron data.
    neuron_value_t* value = &(neuron_values[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)]);

    // Make sure the neuron value does not go below 0.
    if (*value > 0) {
        // Decrement value by decay rate.
        (*value) -= NEURON_DECAY_RATE;
    } else if (*value < 0) {
        (*value) += NEURON_DECAY_RATE;
    }
}

// __global__ void ccol_fire(neuron* neurons, spike* spikes, synapse* synapses, spikes_count_t* spikes_count) {
//     extern __shared__ spikes_count_t spikes_adds[];

//     neuron* input_neuron = &(neurons[synapses[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)].input_neuron]);

//     if (input_neuron->value > input_neuron->threshold) {
//         // Create a new spike.
//         spikes_adds[threadIdx.x] = 1;
//     } else {
//         spikes_adds[threadIdx.x] = 0;
//     }

//     __syncthreads();

//     // Reduce all adds on the first thread of the block.
//     if (threadIdx.x == blockIdx.x) {
//         for (int i = 0; i < blockDim.x; i++) {
//             (*spikes_count) += spikes_adds[i];

//             // Init the new spike if present.
//             if (spikes_adds[i]) {
//                 spikes[(*spikes_count) - 1].progress = 0;
//                 spikes[(*spikes_count) - 1].synapse = i;
//             }
//         }
//     }
// }

// __global__ void ccol_relax(corticolumn* column) {
//     neuron* current_neuron = &(column->neurons[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)]);
//     if (current_neuron->value > current_neuron->threshold) {
//         // Set neuron value to recovery.
//         current_neuron->value = NEURON_RECOVERY_VALUE;

//         if (current_neuron->activity < SYNAPSE_GEN_THRESHOLD) {
//             current_neuron->activity += NEURON_ACTIVITY_STEP;
//         }
//     } else {
//         if (current_neuron->activity > 0) {
//             current_neuron->activity--;
//         }
//     }
// }

void ccol_tick(corticolumn* column) {
    // Update synapses.
    if (column->spikes_count > 0) {
        ccol_propagate<<<1, column->spikes_count>>>(column->spike_progresses, column->spike_synapses, column->synapse_propagation_times);
        CUDA_CHECK_ERROR();
    }

    printf("propagate done\n");

    // Update neurons with spikes data.
    if (column->spikes_count > 0) {
        // Copy spikes count to device.
        spikes_count_t* traveling_spikes_count;
        cudaMalloc((void**) &traveling_spikes_count, sizeof(spikes_count_t));
        CUDA_CHECK_ERROR();
        cudaMemcpy(traveling_spikes_count, &(column->traveling_spikes_count), sizeof(spikes_count_t), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        // ccol_increment<<<column->neurons_count, column->spikes_count>>>(column);
        ccol_increment<<<1, column->spikes_count>>>(column->spike_progresses,
                                                    column->spike_synapses,
                                                    column->synapse_values,
                                                    column->synapse_output_neurons,
                                                    column->neuron_values,
                                                    column->traveling_spike_progresses,
                                                    column->traveling_spike_synapses,
                                                    traveling_spikes_count);
        CUDA_CHECK_ERROR();

        // Copy back to host.
        cudaMemcpy(&(column->traveling_spikes_count), traveling_spikes_count, sizeof(spikes_count_t), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR();
        cudaFree(traveling_spikes_count);
        CUDA_CHECK_ERROR();
    }

    printf("increment done %d\n", column->traveling_spikes_count);

    // Apply decay to all neurons.
    if (column->neurons_count > 0) {
        ccol_decay<<<1, column->neurons_count>>>(column->neuron_values);
        CUDA_CHECK_ERROR();
        cudaDeviceSynchronize();
    }

    printf("decay done\n");

    // Fire neurons.
    // if (column->synapses_count > 0) {
    //     // Copy spikes count to device.
    //     spikes_count_t* spikes_count;
    //     cudaMalloc((void**) &spikes_count, sizeof(spikes_count_t));
    //     CUDA_CHECK_ERROR();
    //     cudaMemcpy(spikes_count, &(column->spikes_count), sizeof(spikes_count_t), cudaMemcpyHostToDevice);
    //     CUDA_CHECK_ERROR();

    //     // Launch kernel.
    //     ccol_fire<<<1, column->synapses_count>>>(column->neurons, column->spikes, column->synapses, spikes_count);
    //     CUDA_CHECK_ERROR();

    //     // Copy back to host.
    //     cudaMemcpy(&(column->spikes_count), spikes_count, sizeof(spikes_count_t), cudaMemcpyDeviceToHost);
    //     CUDA_CHECK_ERROR();
    //     cudaFree(spikes_count);
    //     CUDA_CHECK_ERROR();
    // }

    printf("fire done\n");

    // // Relax neuron values.
    // if (column->neurons_count > 0) {
    //     ccol_relax<<<1, column->neurons_count>>>(column);
    //     CUDA_CHECK_ERROR();
    // }

    // printf("relax done\n");
}


// ONLY FOR DEBUG PURPOSES, REMOVE WHEN NOT NEEDED ANYMORE.
void ccol_copy_to_host(corticolumn* column) {
    // neuron* neurons = (neuron*) malloc(column->neurons_count * sizeof(neuron));
    // cudaMemcpy(neurons, column->neurons, column->neurons_count * sizeof(neuron), cudaMemcpyDeviceToHost);
    // CUDA_CHECK_ERROR();

    // cudaFree(column->neurons);
    // CUDA_CHECK_ERROR();

    // column->neurons = neurons;
}