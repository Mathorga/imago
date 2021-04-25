/*
*****************************************************************
imago.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __IMAGO__
#define __IMAGO__

#include <stdint.h>
#include <stdlib.h>

#define SPIKE_DELIVERED -1
#define SPIKE_IDLE -2

#define DECAY_RATE 1

struct Neuron {
    // Threshold value. The neuron fires if value goes above it.
    uint8_t threshold;

    // Actual value of the neuron. If it goes above threshold, then the neuron fires.
    uint8_t value;
};

struct Synapse {
    // Propagation time of spikes along the synapse.
    uint8_t propagationTime;

    // Progress of the current spike along the synapse.
    int16_t progress;

    // Value of the synapse. This is what influences the output neuron.
    int8_t value;

    // Index of the input neuron.
    uint32_t inputNeuron;

    // Index of the output neuron.
    uint32_t outputNeuron;
};

// Defines the building block of the brain intelligence: the minimum sensory-motor learning model.
struct Corticolumn {
    // The number of neuron in the corticolumn (also defines the number of synapses).
    uint32_t neuronsNum;

    // Actual neurons in the corticolumn. The size is defined by neuronsNum.
    struct Neuron* neurons;

    // Amount of synapses per neuron. The size is defined by neuronsNum.
    uint32_t* synapsesNums;

    // Synapses in the corticolumn. This is an array of synapses (more than one for each neuron). The size is defined by neuronsNum.
    // Each element is an array of actual synapses. Each size is defined by the corresponding element in synapsesNum.
    struct Synapses** synapses;
};

// Initializes the given Corticolumn with default values.
void initColumn(struct Corticolumn* column, uint32_t neuronsNum);

// Propagates synapse spikes to post-synaptic neurons.
void propagate(uint8_t* propagationTimes,
               uint32_t* indexes,
               uint8_t* progresses,
               uint32_t synapsesNum);

// Increments neuron values with spikes from input synapses and decrements them by decay.
void increment(int8_t* neuronValues,
               uint32_t** neuronInputs,
               uint32_t* synapseIndexes,
               int8_t* synapseValues,
               uint8_t* spikeProgresses,
               uint32_t neuronsNum,
               uint32_t synapsesNum);

// Triggers neuron firing if values exceeds threshold.
void fire();

// Copies a whole corticolumn to device in order to run kernels on it.
void copyToDevice();

// Performs a full cycle over the network corticolumn.
void tick(struct Corticolumn* column);

#endif