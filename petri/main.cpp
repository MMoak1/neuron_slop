#include "sim/simulation.h"
#include "render/renderer.h"

int main() {
    // 1. Initialize simulation and spawns creatures
    Simulation sim;
    sim.init();

    // 2. Initialize graphics and open game window
    Renderer renderer(sim);
    renderer.init();

    // 3. Execution loop
    while (!renderer.should_close()) {
        sim.step();     // Compute sensory inputs, run CUDA step, apply physics
        renderer.render(); // Handle camera controls, select creatures, render frames
    }

    return 0;
}
