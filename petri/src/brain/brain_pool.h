#pragma once

#include <vector>
#include <cstdint>
#include "creature/creature.h"
#include "brain/brain_config.h"

class BrainPool {
public:
    BrainPool(int max_creatures);
    ~BrainPool();

    // Allocate resources on GPU
    void init();

    // Upload weight matrices from CPU creatures to GPU
    void upload_genomes(const std::vector<Creature>& creatures);

    // Upload sensory current inputs for all creatures
    void upload_sensory_inputs(const std::vector<float>& sensory_inputs);

    // Run one physical tick of LIF neuron simulation on GPU
    void step();

    // Download motor output firing states from GPU
    void download_motor_outputs(std::vector<uint8_t>& out_motor_fired);

private:
    int max_creatures_;
    bool initialized_;

    // GPU Pointers
    float* d_v_ = nullptr;
    uint8_t* d_fired_prev_ = nullptr;
    uint8_t* d_fired_curr_ = nullptr;
    float* d_current_injected_ = nullptr;
    float* d_weights_ = nullptr;

    // Helper to free GPU memory
    void free_resources();
};
