#include "brain/brain_pool.h"
#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel to advance LIF neurons for all creatures by 1 tick
__global__ void lif_step_kernel(
    float* v,
    const uint8_t* fired_prev,
    uint8_t* fired_curr,
    const float* current_injected,
    const float* weights,
    int num_creatures
) {
    int global_neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_neuron_idx >= num_creatures * N_NEURONS) return;

    int creature_idx = global_neuron_idx / N_NEURONS;
    int neuron_idx = global_neuron_idx % N_NEURONS;

    // 1. Accumulate synaptic inputs from previous tick's spikes
    float syn_current = 0.0f;
    int weight_offset = creature_idx * N_NEURONS * N_NEURONS;
    int neuron_offset = creature_idx * N_NEURONS;

    for (int j = 0; j < N_NEURONS; j++) {
        if (fired_prev[neuron_offset + j]) {
            // weight matrix index: j * N_NEURONS + neuron_idx (weight from j to neuron_idx)
            syn_current += weights[weight_offset + j * N_NEURONS + neuron_idx];
        }
    }

    // 2. Add external sensory input if this is a sensory neuron
    float sensory_current = 0.0f;
    if (neuron_idx < N_SENSORY) {
        sensory_current = current_injected[neuron_offset + neuron_idx];
    }

    // 3. Compute new membrane potential
    float new_v = v[global_neuron_idx] * LEAK_DECAY + syn_current + sensory_current;

    // 4. Threshold & spike generation
    if (new_v >= V_THRESH) {
        v[global_neuron_idx] = V_REST;
        fired_curr[global_neuron_idx] = 1;
    } else {
        if (new_v < V_REST) {
            new_v = V_REST; // Clamp to rest
        }
        v[global_neuron_idx] = new_v;
        fired_curr[global_neuron_idx] = 0;
    }
}

BrainPool::BrainPool(int max_creatures) 
    : max_creatures_(max_creatures), initialized_(false) {}

BrainPool::~BrainPool() {
    free_resources();
}

void BrainPool::init() {
    if (initialized_) return;

    cudaError_t err;

    err = cudaMalloc(&d_v_, max_creatures_ * N_NEURONS * sizeof(float));
    if (err != cudaSuccess) std::cerr << "CUDA malloc v failed: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_fired_prev_, max_creatures_ * N_NEURONS * sizeof(uint8_t));
    if (err != cudaSuccess) std::cerr << "CUDA malloc fired_prev failed: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_fired_curr_, max_creatures_ * N_NEURONS * sizeof(uint8_t));
    if (err != cudaSuccess) std::cerr << "CUDA malloc fired_curr failed: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_current_injected_, max_creatures_ * N_NEURONS * sizeof(float));
    if (err != cudaSuccess) std::cerr << "CUDA malloc current_injected failed: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_weights_, max_creatures_ * N_NEURONS * N_NEURONS * sizeof(float));
    if (err != cudaSuccess) std::cerr << "CUDA malloc weights failed: " << cudaGetErrorString(err) << "\n";

    // Set initial memory
    cudaMemset(d_v_, 0, max_creatures_ * N_NEURONS * sizeof(float));
    cudaMemset(d_fired_prev_, 0, max_creatures_ * N_NEURONS * sizeof(uint8_t));
    cudaMemset(d_fired_curr_, 0, max_creatures_ * N_NEURONS * sizeof(uint8_t));
    cudaMemset(d_current_injected_, 0, max_creatures_ * N_NEURONS * sizeof(float));
    cudaMemset(d_weights_, 0, max_creatures_ * N_NEURONS * N_NEURONS * sizeof(float));

    initialized_ = true;
}

void BrainPool::upload_genomes(const std::vector<Creature>& creatures) {
    if (!initialized_) return;

    std::vector<float> host_weights;
    host_weights.reserve(max_creatures_ * N_NEURONS * N_NEURONS);

    for (const auto& c : creatures) {
        if (c.alive) {
            host_weights.insert(host_weights.end(), c.genome.weights.begin(), c.genome.weights.end());
        } else {
            // Fill with zeros for empty slots
            host_weights.insert(host_weights.end(), N_NEURONS * N_NEURONS, 0.0f);
        }
    }
    // Pad if active creatures is less than max_creatures_
    if (creatures.size() < (size_t)max_creatures_) {
        host_weights.insert(host_weights.end(), (max_creatures_ - creatures.size()) * N_NEURONS * N_NEURONS, 0.0f);
    }

    cudaMemcpy(d_weights_, host_weights.data(), host_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void BrainPool::upload_sensory_inputs(const std::vector<float>& sensory_inputs) {
    if (!initialized_) return;

    // input sensory_inputs contains (num_creatures * N_SENSORY) entries
    // we map them to the first N_SENSORY spots of d_current_injected_
    int num_creatures = sensory_inputs.size() / N_SENSORY;
    std::vector<float> host_injections(max_creatures_ * N_NEURONS, 0.0f);

    for (int i = 0; i < num_creatures; i++) {
        for (int j = 0; j < N_SENSORY; j++) {
            host_injections[i * N_NEURONS + j] = sensory_inputs[i * N_SENSORY + j];
        }
    }

    cudaMemcpy(d_current_injected_, host_injections.data(), host_injections.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void BrainPool::step() {
    if (!initialized_) return;

    // Launch configuration: 1 thread per neuron, blocks of 256
    int total_neurons = max_creatures_ * N_NEURONS;
    int threads_per_block = 256;
    int blocks = (total_neurons + threads_per_block - 1) / threads_per_block;

    lif_step_kernel<<<blocks, threads_per_block>>>(
        d_v_,
        d_fired_prev_,
        d_fired_curr_,
        d_current_injected_,
        d_weights_,
        max_creatures_
    );

    // Synchronize to ensure step is complete
    cudaDeviceSynchronize();

    // Swap fired_prev and fired_curr for temporal dynamics
    std::swap(d_fired_prev_, d_fired_curr_);
}

void BrainPool::download_motor_outputs(std::vector<uint8_t>& out_motor_fired) {
    if (!initialized_) return;

    // Download the entire fired buffer (simple and fast enough)
    std::vector<uint8_t> host_fired(max_creatures_ * N_NEURONS);
    cudaMemcpy(host_fired.data(), d_fired_prev_, host_fired.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Extract only the motor outputs (last N_MOTOR neurons per creature)
    out_motor_fired.resize(max_creatures_ * N_MOTOR);
    for (int i = 0; i < max_creatures_; i++) {
        for (int j = 0; j < N_MOTOR; j++) {
            out_motor_fired[i * N_MOTOR + j] = host_fired[i * N_NEURONS + (N_NEURONS - N_MOTOR) + j];
        }
    }
}

void BrainPool::free_resources() {
    if (d_v_) cudaFree(d_v_);
    if (d_fired_prev_) cudaFree(d_fired_prev_);
    if (d_fired_curr_) cudaFree(d_fired_curr_);
    if (d_current_injected_) cudaFree(d_current_injected_);
    if (d_weights_) cudaFree(d_weights_);

    d_v_ = nullptr;
    d_fired_prev_ = nullptr;
    d_fired_curr_ = nullptr;
    d_current_injected_ = nullptr;
    d_weights_ = nullptr;
    initialized_ = false;
}
