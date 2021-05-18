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
void init_column(struct corticolumn* column, uint32_t neurons_num);

// Propagates synapse spikes according to their progress.
void propagate(struct corticolumn* column);

// Increments neuron values with spikes from input synapses and decrements them by decay.
void increment(struct corticolumn* column);

// Triggers neuron firing if values exceeds threshold.
void fire(struct corticolumn* column);

// Performs a full cycle over the network corticolumn.
void tick(struct corticolumn* column);

#endif