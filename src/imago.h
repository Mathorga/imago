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

    // Neuron output synapses (synapse indexes).
    uint32_t* neuronOutputs;

    // Amount of neuron in the corticolumn.
    uint32_t neuronsNum;

    // ---------------Synapses data---------------
    // Propagation times for synapses in the corticolumn.
    uint8_t* synapsePropagationTimes;

    // Propagation progresses for synapses in the corticolumn (goes from 0 to propagation time).
    uint8_t* synapsePropagationProgresses;

    // Actual values of synapses in the corticolumn.
    int8_t* synapseValues;

    // Synapse indexes, used to identify neurons' outputs.
    uint32_t* synapseIndexes;

    // Synapse targets (neuron indexes).
    uint32_t* synapseTargets;

    // Amount of synapses in the corticolumn.
    uint32_t synapsesNum;
};

// Initializes the given Corticolumn with default values.
void initColumn(struct Corticolumn* column, uint32_t neuronsNum);

// Propagates synapse spikes to post-synaptic neurons.
void propagate(uint8_t* propagationTimes, uint8_t* propagationProgresses);

// Increments neuron values with spikes from input synapses.
void increment();

// Triggers neuron firing if values exceeds threshold.
void fire();

// Copies a whole corticolumn to device in order to run kernels on it.
void copyToDevice();

// Performs a full cycle over the network corticolumn.
void tick();

#endif