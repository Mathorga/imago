/*
*****************************************************************
imago.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __STANDARD__
#define __STANDARD__

#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "corticolumn.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialization methods.
// Initializes the given corticolumn with default values.
void ccol_init(corticolumn* column, uint32_t neurons_count);

// Initializes the given corticolumn specifying the synapses density (synapses per neuron).
void dccol_init(corticolumn* column, uint32_t neurons_count, uint16_t synapses_density);


// Editing methods.
// TODO


// Execution methods.
// Propagates synapse spikes according to their progress.
void propagate(corticolumn* column);

// Increments neuron values with spikes from input synapses and decrements them by decay.
void increment(corticolumn* column);

// Triggers neuron firing if values exceeds threshold.
void fire(corticolumn* column);

// Performs a full cycle over the network corticolumn.
void tick(corticolumn* column);

#ifdef __cplusplus
}
#endif

#endif