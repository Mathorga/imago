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
#define RECOVERY_VALUE -0x77
#define NEURON_LIFESPAN 0x1111u
#define NEURON_INACTIVITY_MAX 0xFFFFu

// Synapse values.
#define DEFAULT_VALUE 0x22
#define MIN_PROPAGATION_TIME 0x11u
#define DEFAULT_PROPAGATION_TIME 0x32u
#define STARTING_PROGRESS 0x00u
#define SYNAPSE_LIFESPAN 0x00FFu

// Spike values.
#define SPIKE_DELIVERED -1
#define SPIKE_IDLE -2

#ifdef __cplusplus
extern "C" {
#endif

// Neuron data types.
typedef uint8_t neuron_threshold_t;
typedef int16_t neuron_value_t;
typedef uint16_t neuron_inactivity_t;

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

    // The number of timesteps since the last firing.
    neuron_inactivity_t inactivity;
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
} corticolumn;

#ifdef __cplusplus
}
#endif

#endif
