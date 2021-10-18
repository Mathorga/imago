/*
*****************************************************************
imago_cuda.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __IMAGO_CUDA__
#define __IMAGO_CUDA__

#include "corticolumn.h"

// Initialization functions:

// Initializes the given corticolumn with default values.
void ccol_init(corticolumn* column, uint32_t neurons_count);

// Initializes the given corticolumn specifying the synapses density (synapses per neuron).
void dccol_init(corticolumn* column, uint32_t neurons_count, uint16_t synapses_density);


// Execution functions:

// Feeds external spikes to the specified neurons.
__device__ void ccol_feed(corticolumn* column, uint32_t* target_neurons, uint32_t targets_count, int8_t value);

// Propagates synapse spikes according to their progress.
__device__ void ccol_propagate(corticolumn* column);

// Increments neuron values with spikes from input synapses.
__device__ void ccol_increment(corticolumn* column);

// Decrements all neurons values by decay.
__device__ void ccol_decay(corticolumn* column);

// Triggers neuron firing if values exceeds threshold.
__device__ void ccol_fire(corticolumn* column);

// Performs a full run cycle over the network corticolumn.
__global__ void ccol_tick(corticolumn* column);


// Learning functions:

// Deletes all unused synapses.
__global__ void ccol_syndel(corticolumn* column);

// Adds synapses to busy neurons (those that fire frequently).
__global__ void ccol_syngen(corticolumn* column);

// Performs a full evolution cycle over the network corticolumn.
__global__ void ccol_evolve(corticolumn* column);

#endif