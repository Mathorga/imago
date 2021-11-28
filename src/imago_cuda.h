/*
*****************************************************************
imago_cuda.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __IMAGO_CUDA__
#define __IMAGO_CUDA__

#include "braph.h"

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

#define CUDA_CHECK_ERROR() {                                                                                  \
            cudaError_t e = cudaGetLastError();                                                             \
            if (e != cudaSuccess) {                                                                         \
                printf("Cuda failure %s(%d): %d(%s)\n", __FILE__, __LINE__ - 1, e, cudaGetErrorString(e));  \
                exit(0);                                                                                    \
            }                                                                                               \
        }

// Maximum number of spikes.
#define MAX_SPIKES_COUNT 0xFFFFFFu

#ifdef __cplusplus
extern "C" {
#endif

// Initialization functions:

/// Initializes the given braph with default values.
void braph_init(braph* column, neurons_count_t neurons_count);

/// Initializes the given braph specifying the synapses density (synapses per neuron).
void dbraph_init(braph* column, neurons_count_t neurons_count, uint16_t synapses_density);


// Execution functions:

/// Feeds external spikes to the specified neurons.
/// \param column 
void braph_feed(braph* column, neurons_count_t starting_index, neurons_count_t count, int8_t value);

/// Propagates synapse spikes according to their progress.
__global__ void braph_propagate(spike* spikes, synapse* synapses);

/// Increments neuron values with spikes from input synapses.
__global__ void braph_increment(spike* spikes, synapse* synapses, neuron* neurons, spike* traveling_spikes, spikes_count_t* traveling_spikes_count);

/// Decrements all neurons values by decay.
__global__ void braph_decay(neuron* neurons);

/// Triggers neuron firing if values exceeds threshold.
__global__ void braph_fire(neuron* neurons, spike* spikes, synapse* synapses, spikes_count_t* spikes_count);

/// Relaxes value to neurons that exceeded their threshold.
__global__ void braph_relax(neuron* neurons);

/// Performs a full run cycle over the network braph.
/// \param column The braph on which to perform the processing operations.
/// column must reside in device memory.
void braph_tick(braph* column);


// Learning functions:

/// Deletes all unused synapses.
__global__ void braph_syndel(braph* column);

/// Adds synapses to busy neurons (those that fire frequently).
__global__ void braph_syngen(braph* column);

/// Performs a full evolution cycle over the network braph.
__global__ void braph_evolve(braph* column);




// ONLY FOR DEBUG PURPOSES, REMOVE WHEN NOT NEEDED ANYMORE.
void braph_copy_to_host(braph* column);


#ifdef __cplusplus
}
#endif

#endif