#pragma once

#include <vector>
#include <memory>
#include "world/world.h"
#include "creature/creature.h"
#include "brain/brain_pool.h"

class Simulation {
public:
    static constexpr int MAX_CREATURES = 500;
    static constexpr float CREATURE_RADIUS = 5.0f;
    static constexpr float MAX_VIEW_DIST = 160.0f; // Pixels

    Simulation();
    ~Simulation();

    void init();
    void step();

    const World& get_world() const { return world_; }
    const std::vector<Creature>& get_creatures() const { return creatures_; }
    int get_tick_count() const { return tick_count_; }

    // Cast a ray from position in direction, return distance to rock obstacle
    float cast_ray(float start_x, float start_y, float angle_rad, float max_dist) const;

private:
    World world_;
    std::vector<Creature> creatures_;
    std::unique_ptr<BrainPool> brain_pool_;

    int tick_count_ = 0;

    void update_sensory_inputs();
    void resolve_creature_actions(const std::vector<uint8_t>& motor_outputs);
};
