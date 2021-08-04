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

// Initializes the given corticolumn with default values.
void init_column(corticolumn* column, uint32_t neurons_num);

// Propagates synapse spikes according to their progress.
void propagate(corticolumn* column);

// Increments neuron values with spikes from input synapses and decrements them by decay.
void increment(corticolumn* column);

// Triggers neuron firing if values exceeds threshold.
void fire(corticolumn* column);

// Performs a full cycle over the network corticolumn.
void tick(corticolumn* column);

#endif