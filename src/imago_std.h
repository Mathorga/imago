/*
*****************************************************************
imago_std.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __IMAGO_STD__
#define __IMAGO_STD__

#include <stdint.h>
#include <stdlib.h>
// TODO Remove in release.
#include <stdio.h>
#include <time.h>
#include "braph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialization functions:

/// Initializes the given braph with default values.
void braph_init(braph_t* braph, uint32_t neurons_count);

/// Initializes the given braph specifying the synapses density (synapses per neuron).
void dbraph_init(braph_t* braph, uint32_t neurons_count, uint16_t synapses_density);


// Execution functions:

/// Feeds external spikes to the specified neurons.
void braph_feed(braph_t* braph, neurons_count_t starting_index, neurons_count_t count, int8_t value);

/// Propagates synapse spikes according to their progress.
void braph_propagate(braph_t* braph);

/// Increments neuron values with spikes from input synapses.
void braph_increment(braph_t* braph);

/// Decrements all neurons values by decay.
void braph_decay(braph_t* braph);

/// Triggers neuron firing if values exceeds threshold.
void braph_fire(braph_t* braph);

/// Relaxes value to neurons that exceeded their threshold.
void braph_relax(braph_t* braph);

/// Performs a full run cycle over the network braph.
void braph_tick(braph_t* braph);


// Learning functions:

/// Deletes all unused synapses.
void braph_syndel(braph_t* braph);

/// Adds synapses to busy neurons (those that fire frequently).
void braph_syngen(braph_t* braph);

/// Performs a full evolution cycle over the network braph.
void braph_evolve(braph_t* braph);




// ONLY FOR DEBUG PURPOSES, REMOVE WHEN NOT NEEDED ANYMORE.
void braph_copy_to_host(braph_t* braph);

#ifdef __cplusplus
}
#endif

#endif