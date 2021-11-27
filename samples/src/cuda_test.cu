#include <stdint.h>
#include <stdio.h>

// Column values.
#define SYNAPSES_COUNT_MAX 0x00000FFFu

// Neuron values.
#define NEURON_DEFAULT_THRESHOLD 0xCCu
#define NEURON_STARTING_VALUE 0x00u
#define NEURON_DECAY_RATE 0x01u
#define NEURON_RECOVERY_VALUE -0x77
#define NEURON_LIFESPAN 0x1111u
#define NEURON_ACTIVITY_MAX 0xFFFFu
#define NEURON_ACTIVITY_STEP 0x0033u

// Synapse values.
#define SYNAPSE_DEFAULT_VALUE 0x22
#define SYNAPSE_MIN_PROPAGATION_TIME 0x11u
#define SYNAPSE_DEFAULT_PROPAGATION_TIME 0x32u
#define SYNAPSE_STARTING_PROGRESS 0x00u
#define SYNAPSE_DEL_THRESHOLD 0x00FFu
#define SYNAPSE_GEN_THRESHOLD 0x1100u

// Spike values.
#define SPIKE_DELIVERED -1
#define SPIKE_IDLE -2


// Translate an id wrapping it to the provided size (pacman effect).
// [i] is the given index.
// [n] is the size over which to wrap.
#define IDX(i, n) (i < 0 ? (i % n) : (n + (i % n)))

// Translates bidimensional indexes to a monodimensional one.
// |i| is the row index.
// |j| is the column index.
// |m| is the number of columns (length of the rows).
#define IDX2D(i, j, m) ((m * j) + i)

// Translates tridimensional indexes to a monodimensional one.
// |i| is the index in the first dimension.
// |j| is the index in the second dimension.
// |k| is the index in the third dimension.
// |m| is the size of the first dimension.
// |n| is the size of the second dimension.
#define IDX3D(i, j, k, m, n) ((m * n * k) + (m * j) + i)

#define CUDA_CHECK_ERROR() {                                                                                \
            cudaError_t e = cudaGetLastError();                                                             \
            if (e != cudaSuccess) {                                                                         \
                printf("Cuda failure %s(%d): %d(%s)\n", __FILE__, __LINE__ - 1, e, cudaGetErrorString(e));  \
                exit(0);                                                                                    \
            }                                                                                               \
        }

// Maximum number of spikes.
#define MAX_SPIKES_COUNT 0xFFFFFFu

// Neuron data types.
typedef uint8_t neuron_threshold_t;
typedef int16_t neuron_value_t;
typedef uint16_t neuron_activity_t;

// Synapse data types.
typedef uint8_t synapse_propagation_time_t;
typedef int8_t synapse_value_t;

// Spike data types.
typedef int16_t spike_progress_t;

// Corticolumn data types.
typedef uint32_t neurons_count_t;
typedef uint32_t synapses_count_t;
typedef uint32_t spikes_count_t;


typedef struct {
    // Threshold value. The neuron fires if value goes above it.
    neuron_threshold_t threshold;

    // Actual value of the neuron. If it goes above threshold, then the neuron fires.
    neuron_value_t value;

    // The activity level of the neuron (direct match for the firing rate);
    neuron_activity_t activity;
} neuron;

typedef struct {
    // Propagation time of spikes along the synapse.
    synapse_propagation_time_t propagation_time;

    // Value of the synapse. This is what influences the output neuron.
    synapse_value_t value;

    // Index of the input neuron.
    neurons_count_t input_neuron;

    // Index of the output neuron.
    neurons_count_t output_neuron;
} synapse;

typedef struct {
    // Progress of the current spike along the synapse.
    spike_progress_t progress;

    // Reference synapse.
    synapses_count_t synapse;
} spike;

// Defines the building block of the brain intelligence: the minimum sensory-motor learning model.
typedef struct {
    // The number of neuron in the corticolumn (also defines the number of synapses).
    neurons_count_t neurons_count;

    // Actual neurons in the corticolumn. The size is defined by neuronsNum.
    neuron* neurons;

    // Amount of synapses in the corticolumn.
    synapses_count_t synapses_count;

    // Synapses in the corticolumn. This size is defined by synapsesNum.
    synapse* synapses;

    spikes_count_t spikes_count;

    spike* spikes;

    spikes_count_t traveling_spikes_count;

    spike* traveling_spikes;
} corticolumn;

void dccol_init(corticolumn* column, neurons_count_t neurons_count, uint16_t synapses_density) {
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

void ccol_init(corticolumn* column, neurons_count_t neurons_count) {
    dccol_init(column, neurons_count, 10);
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

__global__ void ccol_propagate(spike* spikes, synapse* synapses) {
    // Retrieve current spike.
    spike* current_spike = &(spikes[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)]);

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

    spike* current_spike = &(spikes[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)]);

    if (current_spike->progress == SPIKE_DELIVERED) {
        // Increment target neuron.
        synapse* reference_synapse = &(synapses[spikes[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)].synapse]);
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
    neuron* current_neuron = &(neurons[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)]);

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

    neuron* input_neuron = &(neurons[synapses[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)].input_neuron]);

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
    // Update synapses.
    if (column->spikes_count > 0) {
        ccol_propagate<<<1, column->spikes_count>>>(column->spikes, column->synapses);
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
        ccol_increment<<<1, column->spikes_count>>>(column->spikes, column->synapses, column->neurons, column->traveling_spikes, traveling_spikes_count);
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
        ccol_decay<<<1, column->neurons_count>>>(column->neurons);
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
    neuron* neurons = (neuron*) malloc(column->neurons_count * sizeof(neuron));
    cudaMemcpy(neurons, column->neurons, column->neurons_count * sizeof(neuron), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();

    cudaFree(column->neurons);
    CUDA_CHECK_ERROR();

    column->neurons = neurons;
}

int main(int argc, char **argv) {
    corticolumn column;
    // cudaMalloc(&column, sizeof(corticolumn));

    printf("Started\n");

    ccol_init(&column, 10);

    printf("Initialized %zu\n", sizeof(neuron));

    // ccol_feed(&column, input_neurons, 4, SYNAPSE_DEFAULT_VALUE);

    ccol_tick(&column);
    
    printf("Ticked\n");

    ccol_copy_to_host(&column);
    
    printf("Copied back %d\n", column.neurons[15].threshold);
}