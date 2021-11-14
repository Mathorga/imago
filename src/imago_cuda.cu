#include "imago_cuda.h"
#include "stdio.h"

void ccol_init(corticolumn* column, neurons_count_t neurons_count) {
    dccol_init(column, neurons_count, 10);
}

void dccol_init(corticolumn* column, neurons_count_t neurons_count, uint16_t synapses_density) {
    synapses_count_t synapses_count = neurons_count * synapses_density;

    // Allocate neurons.
    cudaMemcpy(&(column->neurons_count), &neurons_count, sizeof(neurons_count_t), cudaMemcpyHostToDevice);
    cudaMalloc(&(column->neurons), neurons_count * sizeof(neuron));
    neuron* tmp_neurons = (neuron*) malloc(neurons_count * sizeof(neuron));

    // Initialize neurons with default values.
    for (neurons_count_t i = 0; i < neurons_count; i++) {
        tmp_neurons[i].threshold = NEURON_DEFAULT_THRESHOLD;
        tmp_neurons[i].value = NEURON_STARTING_VALUE;
        tmp_neurons[i].activity = 0;
    }

    // Copy neuron values to device.
    cudaMemcpy(column->neurons, tmp_neurons, neurons_count * sizeof(neuron), cudaMemcpyHostToDevice);

    // Allocate synapses.
    cudaMemcpy(&(column->synapses_count), &synapses_count, sizeof(synapses_count_t), cudaMemcpyHostToDevice);
    cudaMalloc(&(column->synapses), synapses_count * sizeof(synapse));
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

    // Allocate spikes.
    cudaMemcpy(&(column->spikes_count), 0, sizeof(spikes_count_t), cudaMemcpyHostToDevice);
    cudaMalloc(&(column->spikes), MAX_SPIKES_COUNT * sizeof(spike));
    cudaMemcpy(&(column->traveling_spikes_count), 0, sizeof(spikes_count_t), cudaMemcpyHostToDevice);
    cudaMalloc(&(column->traveling_spikes), MAX_SPIKES_COUNT * sizeof(spike));
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

__global__ void ccol_increment(corticolumn* column) {
    if (column->spikes[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)].progress == SPIKE_DELIVERED) {
        // Increment target neuron.
        synapse* reference_synapse = &(column->synapses[column->spikes[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)].synapse]);
        neuron* target_neuron = &(column->neurons[reference_synapse->output_neuron]);

        target_neuron->value += reference_synapse->value;
    } else {
        // Save the spike as traveling.
        column->traveling_spikes_count++;
        column->traveling_spikes[column->traveling_spikes_count - 1] = column->spikes[IDX2D(blockIdx.x, threadIdx.x, blockDim.x)];
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
    // Update synapses.
    ccol_propagate<<<1, column->spikes_count>>>(column);

    // Update neurons with spikes data.
    ccol_increment<<<1, column->spikes_count>>>(column);

    // Apply decay to all neurons.
    ccol_decay<<<1, column->neurons_count>>>(column);

    // Fire neurons.
    ccol_fire<<<1, column->synapses_count>>>(column);

    // Relax neuron values.
    ccol_relax<<<1, column->neurons_count>>>(column);
}