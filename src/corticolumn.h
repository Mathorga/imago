/*
*****************************************************************
corticolumn.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __CORTICOLUMN__
#define __CORTICOLUMN__

#include <stdint.h>

// Neuron values.
#define DEFAULT_THRESHOLD 0xCCu
#define STARTING_VALUE 0x00u
#define DECAY_RATE 1

// Synapse values.
#define DEFAULT_VALUE -0x22
#define DEFAULT_PROPAGATION_TIME 0x22u
#define STARTING_PROGRESS 0x00u

// Spyke values.
#define SPIKE_DELIVERED -1
#define SPIKE_IDLE -2

struct neuron {
    // Threshold value. The neuron fires if value goes above it.
    uint8_t threshold;

    // Actual value of the neuron. If it goes above threshold, then the neuron fires.
    uint8_t value;
};

struct synapse {
    // Propagation time of spikes along the synapse.
    uint8_t propagation_time;

    // Value of the synapse. This is what influences the output neuron.
    int8_t value;

    // Index of the input neuron.
    uint32_t input_neuron;

    // Index of the output neuron.
    uint32_t output_neuron;
};

struct spike {
    // Progress of the current spike along the synapse.
    int16_t progress;

    // Reference synapse.
    uint32_t synapse;
};

// Defines the building block of the brain intelligence: the minimum sensory-motor learning model.
struct corticolumn {
    // The number of neuron in the corticolumn (also defines the number of synapses).
    uint32_t neurons_num;

    // Actual neurons in the corticolumn. The size is defined by neuronsNum.
    struct neuron* neurons;

    // Amount of synapses in the corticolumn.
    uint32_t synapses_num;

    // Synapses in the corticolumn. This size is defined by synapsesNum.
    struct synapse* synapses;

    uint32_t spikes_num;

    struct spike* spikes;
};

#endif