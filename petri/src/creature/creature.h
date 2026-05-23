#pragma once

#include <vector>
#include <cstdint>
#include "brain/brain_config.h"

struct Genome {
    // Dense connection weight matrix of size N_NEURONS * N_NEURONS
    std::vector<float> weights;

    Genome() {
        weights.resize(N_NEURONS * N_NEURONS, 0.0f);
    }
};

struct Creature {
    uint32_t id;
    float x;
    float y;
    float angle;         // Facing direction in radians (0 to 2*PI)
    float speed;         // Current speed
    bool alive;
    uint32_t brain_idx;  // Offset index in the GPU brain pool
    Genome genome;

    // Phase 1 stats
    float energy;
    int generation;
};
