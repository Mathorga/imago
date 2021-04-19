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
    uint16_t index;
    uint8_t threshold;
    
};

struct Corticolumn {
    // Index of the corticolumn.
    uint16_t index;

    // ---------------Neurons data---------------
    // Threshold values for neurons in the corticolumn.
    uint8_t* neuronThresholds;

    // Actual neuron potential values in the current corticolumn.
    int8_t* neuronValues;

    // Neuron indexes, used to identify synapses' targets.
    uint32_t* neuronIndexes;

    // Neuron input synapses (synapse indexes).
    uint32_t** neuronInputs;

    // Sizes of neuron input synapses.
    uint32_t* neuronInputsSizes;

    // Neuron output synapses (synapse indexes).
    uint32_t** neuronOutputs;

    // Amount of neuron in the corticolumn.
    uint32_t neuronsNum;

    // ---------------Synapses data---------------
    // Propagation times for synapses in the corticolumn.
    uint8_t* synapsePropagationTimes;

    // Progress of spikes along synapses.
    uint8_t* spikeProgresses;

    // Actual values of synapses in the corticolumn.
    int8_t* synapseValues;

    // Synapse indexes, used to identify neurons' outputs.
    uint32_t* synapseIndexes;

    // Amount of synapses in the corticolumn.
    uint32_t synapsesNum;
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