/*
 * brain5/engine/test.cpp
 * Scaling Laws Test for Hebbian STDP Learning
 */

#include "brain.h"
#include <iostream>
#include <chrono>
#include <map>

using namespace std;

// Corpus for sensory input
const vector<vector<string>> CORPUS = {
    {"cats", "are", "furry", "animals", "that", "meow"},
    {"dogs", "are", "furry", "animals", "that", "bark"},
    {"birds", "are", "animals", "that", "fly", "and", "chirp"},
    {"fish", "are", "animals", "that", "swim", "in", "water"},
    {"sharks", "are", "large", "fish", "with", "teeth"},
    {"cats", "and", "dogs", "are", "common", "pets"}
};

void run_scaling_experiment(int N, int K) {
    cout << "\n============================================\n";
    cout << ">> EXPERIMENT: " << N << " Neurons (Sparsity K=" << K << ")\n";
    cout << "============================================\n";

    auto t_start = chrono::high_resolution_clock::now();

    Brain brain(N, K, 42); // seed 42

    // Register all words in corpus
    for (const auto& sentence : CORPUS) {
        for (const auto& w : sentence) {
            brain.register_word(w);
        }
    }

    // --- Phase 1: LEARNING ---
    cout << "Phase 1: Learning (STDP Plasticity Active)...\n";
    
    // We will do 5 epochs of the corpus to let Hebbian weights settle
    int total_learning_spikes = 0;
    for (int epoch = 0; epoch < 5; epoch++) {
        for (const auto& sentence : CORPUS) {
            // "Rest" between sentences to clear traces
            for(int t=0; t<50; t++) brain.step(); 

            // Read words sequentially
            for (const auto& w : sentence) {
                // First tick: inject sensory data and measure prediction surprise!
                brain.inject_word(w);
                if (epoch > 3) { // On the last epoch
                    cout << "    Word '" << w << "' -> Surprise: " << brain.last_surprise << ", Dopamine: " << brain.dopamine << "\n";
                }
                
                // Let physics run for 20 ticks (word duration)
                for(int t=0; t<20; t++) {
                    total_learning_spikes += brain.step();
                }
            }
            }
        }
    }

    auto t_mid = chrono::high_resolution_clock::now();
    double ms_learn = chrono::duration<double, milli>(t_mid - t_start).count();
    cout << "  Learning time: " << ms_learn << " ms\n";
    cout << "  Total spikes during learning: " << total_learning_spikes << "\n";

    // --- Phase 2: RECALL & ASSOCIATIONS ---
    // We test what "cat" associates with, without updating weights.
    // To do this, we stimulate "cat", let it run, and count spikes across all word receptors.
    
    cout << "\nPhase 2: Semantic Recall Test\n";
    
    // List of words to probe
    vector<string> probes = {"cats", "dogs", "fish"};

    for (const string& probe : probes) {
        // Reset ALL voltages and traces before probe
        for (int i=0; i<N; i++) {
            brain.v[i] = 0.0f;
            brain.trace_pre[i] = 0.0f;
            brain.trace_post[i] = 0.0f;
        }

        // We temporarily "disable" plasticity in the step loop purely for the test
        // By just running a custom step that doesn't modify weights.
        
        // Inject current to probe
        for (int id : brain.word_receptors[probe]) {
            brain.v[id] += RECEPTOR_CURRENT;
        }

        map<string, int> word_spikes;

        // Run recall physics for 30 ticks
        for(int t=0; t<30; t++) {
            // Simplified Step without STDP learning
            vector<int> spikers;
            for (int i = 0; i < N; i++) {
                brain.v[i] *= LEAK_DECAY;
                if (brain.v[i] >= V_THRESH) {
                    brain.v[i] = V_REST; 
                    spikers.push_back(i);
                }
            }

            for (int i : spikers) {
                // Check if this neuron belongs to any word
                for (const auto& [w, receptors] : brain.word_receptors) {
                    for (int rec_id : receptors) {
                        if (rec_id == i && w != probe) {
                            word_spikes[w]++;
                        }
                    }
                }

                // Propagate
                for (const auto& edge : brain.out_edges[i]) {
                    brain.v[edge.node] += brain.syn_weights[edge.syn_idx];
                }
            }
        }

        // Sort results
        vector<pair<int, string>> results;
        for (const auto& [w, count] : word_spikes) {
            results.push_back({count, w});
        }
        sort(results.rbegin(), results.rend());

        cout << "  > Activating \033[1;36m'" << probe << "'\033[0m sends electricity to: ";
        if (results.empty()) {
            cout << "nothing (silence)\n";
        } else {
            for(int i=0; i<min(5, (int)results.size()); i++) {
                if (i > 0) cout << ", ";
                cout << results[i].second << "(" << results[i].first << " spikes)";
            }
            cout << "\n";
        }
    }

int main() {
    cout << "\033[1;35m";
    cout << "============================================\n";
    cout << "BRAIN 5: The Neon Cortex - Scaling Laws\n";
    cout << "============================================\n";
    cout << "\033[0m\n";

    // 100 Neurons: High collision chance. K=5 means 5% of brain per word.
    run_scaling_experiment(100, 5);

    // 1000 Neurons: Modest collision chance. K=20 means 2% of brain per word.
    run_scaling_experiment(1000, 20);

    // 10000 Neurons: Very low collision chance. K=50 means 0.5% of brain per word.
    run_scaling_experiment(10000, 50);

    return 0;
}
