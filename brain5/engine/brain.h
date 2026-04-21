/*
 * brain5/engine/brain.h
 * 
 * The Neon Cortex — A pure Leaky Integrate-and-Fire (LIF) neural network
 * simulating language ingestion via Sparse Distributed Representations (SDR).
 * "Neurons that fire together, wire together."
 */

#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

// ── Physics Constants ──────────────────────────────────────
static constexpr float V_REST = 0.0f;           // Resting membrane potential
static constexpr float V_THRESH = 1.0f;         // Spiking threshold
static constexpr float LEAK_DECAY = 0.90f;      // Voltage retention per tick
static constexpr float TRACE_DECAY = 0.85f;     // STDP trace retention per tick
static constexpr float RECEPTOR_CURRENT = 1.5f; // Initial current injection for text
static constexpr float A_PLUS = 0.05f;          // STDP LTP (strengthening)
static constexpr float A_MINUS = 0.02f;         // STDP LTD (weakening)
static constexpr float MAX_WEIGHT = 2.0f;       // Synapse weight limit

struct Edge {
    int node;
    int syn_idx; // Index into the global synapse_weights array
};

struct Brain {
    int N;                      
    int K;                      
    int tick = 0;

    // ── Node arrays (Struct of Arrays) ──
    std::vector<float> v;
    std::vector<float> trace_pre;
    std::vector<float> trace_post;
    std::vector<bool> fired;

    // ── Synapse Graph (CSR style but with object edge lists for easy building) ──
    // In a pure CSR, out_edges is a single flat array. For simplicity and 
    // exact equivalence to vectors, we use vector of vectors. This is still 
    // vastly faster than unordered_map as memory is contiguous per-node.
    std::vector<std::vector<Edge>> out_edges;
    std::vector<std::vector<Edge>> in_edges;
    std::vector<float> syn_weights;

    // ── NLP Vocabulary Map ──
    std::unordered_map<std::string, std::vector<int>> word_receptors;
    std::vector<std::string> id_to_word;
    std::unordered_map<std::string, int> word_ids;

    std::mt19937 rng;

    Brain(int num_neurons, int sparsity_k, unsigned seed = 42) : 
        N(num_neurons), K(sparsity_k), rng(seed) 
    {
        v.resize(N, V_REST);
        trace_pre.resize(N, 0.0f);
        trace_post.resize(N, 0.0f);
        fired.resize(N, false);
        out_edges.resize(N);
        in_edges.resize(N);

        // Biological constraints: cap maximum physical fan-out
        int edges_per_neuron = std::min(N, 100); 
        std::uniform_int_distribution<int> n_dist(0, N - 1);
        // Extremely weak initial weights to prevent global seizures
        std::uniform_real_distribution<float> w_dist(0.0001f, 0.005f);

        // Pre-allocate approximate synapse pool
        syn_weights.reserve(N * edges_per_neuron);

        for (int i = 0; i < N; i++) {
            // Fast unique random sampling: shuffle an identity array
            std::vector<int> candidates(N);
            for(int j=0; j<N; j++) candidates[j] = j;
            std::shuffle(candidates.begin(), candidates.end(), rng);

            int added = 0;
            for (int j : candidates) {
                if (i == j) continue;
                
                int syn_idx = syn_weights.size();
                float weight = w_dist(rng);
                syn_weights.push_back(weight);

                out_edges[i].push_back({j, syn_idx});
                in_edges[j].push_back({i, syn_idx});
                
                added++;
                if (added >= edges_per_neuron) break;
            }
        }
    }

    // ── Neuromodulation (The World Model) ──
    float dopamine = 1.0f;          // Global learning multiplier
    float last_surprise = 0.0f;     // How unexpected was the latest sensory input?

    void register_word(const std::string& word) {
        if (word_ids.count(word)) return;
        
        std::vector<int> receptors;
        std::uniform_int_distribution<int> dist(0, N - 1);
        
        std::unordered_set<int> picked;
        while ((int)picked.size() < K) {
            picked.insert(dist(rng));
        }
        for (int id : picked) receptors.push_back(id);

        word_receptors[word] = receptors;
        word_ids[word] = id_to_word.size();
        id_to_word.push_back(word);
    }

    // Sensory Input: Predicts surprise and injects current
    void inject_word(const std::string& word) {
        if (word_receptors.find(word) == word_receptors.end()) return;
        
        const auto& receptors = word_receptors[word];
        
        // 1. Calculate Surprise (Prediction Error)
        // If the brain expected this word, these neurons would already 
        // have slightly elevated voltage from predictive cascades.
        float expected_voltage = 0.0f;
        for (int id : receptors) {
            expected_voltage += v[id];
        }
        expected_voltage /= K;

        // If expectation was 0.0, surprise is 1.0. If expectation was 0.5 (high), surprise drops.
        last_surprise = std::max(0.0f, 1.0f - (expected_voltage / 0.5f));
        
        // Dopamine rush = high learning on surprise. Boredom = no learning when predicted.
        dopamine = 0.1f + (last_surprise * 0.9f); 

        // 2. Inject raw sensory electricity
        for (int id : receptors) {
            v[id] += RECEPTOR_CURRENT;
        }
    }

    int step() {
        tick++;
        int total_spikes = 0;
        std::vector<int> current_spikers;

        // 1. Update voltages, traces, and calculate spikes
        for (int i = 0; i < N; i++) {
            fired[i] = false;
            
            v[i] *= LEAK_DECAY;
            trace_pre[i] *= TRACE_DECAY;
            trace_post[i] *= TRACE_DECAY;

            if (v[i] >= V_THRESH) {
                fired[i] = true;
                v[i] = V_REST; 
                current_spikers.push_back(i);
                total_spikes++;
            }
        }

        // Dopamine naturally decays over time
        dopamine = std::max(0.1f, dopamine * 0.99f);

        // 2. Propagate current & apply STDP (Modulated by Dopamine World Model)
        for (int i : current_spikers) {
            // STDP rule 1: LTD (Post before Pre)
            for (const Edge& edge : out_edges[i]) {
                int j = edge.node;
                int s = edge.syn_idx;
                float ltd = A_MINUS * trace_post[j] * dopamine;
                syn_weights[s] = std::max(0.0f, syn_weights[s] - ltd);
                
                // Inject electrical current downstream (this forms the future prediction!)
                v[j] += syn_weights[s];
            }
            trace_pre[i] = 1.0f; 
        }

        for (int j : current_spikers) {
            // STDP rule 2: LTP (Pre before Post)
            for (const Edge& edge : in_edges[j]) {
                int i = edge.node;
                int s = edge.syn_idx;
                float ltp = A_PLUS * trace_pre[i] * dopamine;
                syn_weights[s] = std::min((float)MAX_WEIGHT, syn_weights[s] + ltp);
            }
            trace_post[j] = 1.0f; 
        }

        return total_spikes;
    }
};
