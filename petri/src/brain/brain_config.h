#pragma once

// Brain architecture dimensions
static constexpr int N_NEURONS = 128;      // Neurons per creature
static constexpr int N_SENSORY = 8;        // Inputs (0 to 7)
static constexpr int N_MOTOR = 8;          // Outputs (N_NEURONS-8 to N_NEURONS-1)

// Physics and LIF Neuron constants
static constexpr float V_REST = 0.0f;
static constexpr float V_THRESH = 1.0f;
static constexpr float LEAK_DECAY = 0.85f;  // Volts leak per tick
static constexpr float RECEPTOR_GAIN = 1.5f; // Scaling factor for sensory input current

// Weight scale for random initialization
static constexpr float WEIGHT_INIT_MIN = -0.5f;
static constexpr float WEIGHT_INIT_MAX = 0.5f;
