#include "sim/simulation.h"
#include <random>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

Simulation::Simulation() {
    brain_pool_ = std::make_unique<BrainPool>(MAX_CREATURES);
}

Simulation::~Simulation() {}

void Simulation::init() {
    tick_count_ = 0;
    world_.init();
    brain_pool_->init();

    creatures_.clear();
    creatures_.resize(MAX_CREATURES);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos_x_dist(World::CELL_SIZE * 2.0f, (World::WIDTH - 2) * World::CELL_SIZE);
    std::uniform_real_distribution<float> pos_y_dist(World::CELL_SIZE * 2.0f, (World::HEIGHT - 2) * World::CELL_SIZE);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> w_dist(WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);

    // Populate creatures
    for (int i = 0; i < MAX_CREATURES; i++) {
        Creature& c = creatures_[i];
        c.id = i;
        c.brain_idx = i;
        c.alive = true;
        c.energy = 1.0f;
        c.generation = 0;
        c.speed = 0.0f;

        // Find a walkable spawn position
        int attempts = 0;
        float rx = 0, ry = 0;
        do {
            rx = pos_x_dist(rng);
            ry = pos_y_dist(rng);
            attempts++;
        } while (!world_.is_walkable(rx, ry, CREATURE_RADIUS) && attempts < 100);

        c.x = rx;
        c.y = ry;
        c.angle = angle_dist(rng);

        // Initialize random weights for recurrent brain connections
        for (int j = 0; j < N_NEURONS * N_NEURONS; j++) {
            c.genome.weights[j] = w_dist(rng);
        }
    }

    // Upload the initial brains to GPU
    brain_pool_->upload_genomes(creatures_);
}

void Simulation::step() {
    tick_count_++;

    // 1. Gather and upload sensory inputs
    update_sensory_inputs();

    // 2. Perform LIF Neuron updates on GPU
    brain_pool_->step();

    // 3. Download motor outputs to CPU
    std::vector<uint8_t> motor_outputs;
    brain_pool_->download_motor_outputs(motor_outputs);

    // 4. Resolve creature actions (movements and collisions)
    resolve_creature_actions(motor_outputs);
}

void Simulation::update_sensory_inputs() {
    std::vector<float> sensory_inputs(MAX_CREATURES * N_SENSORY, 0.0f);

    for (int i = 0; i < MAX_CREATURES; i++) {
        const Creature& c = creatures_[i];
        if (!c.alive) continue;

        int sensory_offset = i * N_SENSORY;

        // Cast rays in three directions relative to heading: forward, left 45, right 45
        float dist_f = cast_ray(c.x, c.y, c.angle, MAX_VIEW_DIST);
        float dist_l = cast_ray(c.x, c.y, c.angle - M_PI / 4.0f, MAX_VIEW_DIST);
        float dist_r = cast_ray(c.x, c.y, c.angle + M_PI / 4.0f, MAX_VIEW_DIST);

        // Inputs 0, 1, 2: Wall proximity (close = 1.0, far = 0.0)
        sensory_inputs[sensory_offset + 0] = (1.0f - (dist_f / MAX_VIEW_DIST)) * RECEPTOR_GAIN;
        sensory_inputs[sensory_offset + 1] = (1.0f - (dist_l / MAX_VIEW_DIST)) * RECEPTOR_GAIN;
        sensory_inputs[sensory_offset + 2] = (1.0f - (dist_r / MAX_VIEW_DIST)) * RECEPTOR_GAIN;

        // Input 3: Energy (constant 1.0f in Phase 1)
        sensory_inputs[sensory_offset + 3] = c.energy * RECEPTOR_GAIN;

        // Input 4: Health / survival state (dummy 1.0f)
        sensory_inputs[sensory_offset + 4] = 1.0f * RECEPTOR_GAIN;

        // Input 5: Normalized speed
        sensory_inputs[sensory_offset + 5] = (c.speed / 3.0f) * RECEPTOR_GAIN;

        // Input 6, 7: Sin/Cos of facing angle for directional orientation
        sensory_inputs[sensory_offset + 6] = std::sin(c.angle) * RECEPTOR_GAIN;
        sensory_inputs[sensory_offset + 7] = std::cos(c.angle) * RECEPTOR_GAIN;
    }

    brain_pool_->upload_sensory_inputs(sensory_inputs);
}

void Simulation::resolve_creature_actions(const std::vector<uint8_t>& motor_outputs) {
    for (int i = 0; i < MAX_CREATURES; i++) {
        Creature& c = creatures_[i];
        if (!c.alive) continue;

        int motor_offset = i * N_MOTOR;

        // Motor 0: Forward thrust
        if (motor_outputs[motor_offset + 0]) {
            c.speed += 0.3f;
        }

        // Motor 1: Turn left
        if (motor_outputs[motor_offset + 1]) {
            c.angle -= 0.06f;
        }

        // Motor 2: Turn right
        if (motor_outputs[motor_offset + 2]) {
            c.angle += 0.06f;
        }

        // Friction and speed capping
        c.speed *= 0.88f;
        if (c.speed > 2.5f) {
            c.speed = 2.5f;
        }

        // Wrap angle
        if (c.angle < 0.0f) c.angle += 2.0f * M_PI;
        if (c.angle >= 2.0f * M_PI) c.angle -= 2.0f * M_PI;

        // Calculate next candidate position
        float prev_x = c.x;
        float prev_y = c.y;
        
        c.x += c.speed * std::cos(c.angle);
        c.y += c.speed * std::sin(c.angle);

        // Resolve collision with rocks/borders
        world_.resolve_collision(c.x, c.y, CREATURE_RADIUS);
        
        // If stuck or fully stopped by obstacle, zero speed
        float dx = c.x - prev_x;
        float dy = c.y - prev_y;
        if (dx*dx + dy*dy < 0.001f && c.speed > 0.0f) {
            c.speed = 0.0f;
        }
    }
}

float Simulation::cast_ray(float start_x, float start_y, float angle_rad, float max_dist) const {
    float dx = std::cos(angle_rad);
    float dy = std::sin(angle_rad);

    float step = 3.0f; // Step size in pixels
    float dist = 0.0f;

    while (dist < max_dist) {
        float px = start_x + dx * dist;
        float py = start_y + dy * dist;

        int tx = static_cast<int>(px / World::CELL_SIZE);
        int ty = static_cast<int>(py / World::CELL_SIZE);

        if (world_.get_tile(tx, ty) == TILE_ROCK) {
            return dist;
        }

        dist += step;
    }

    return max_dist;
}
