#pragma once

#include "sim/simulation.h"
#include "raylib.h"

class Renderer {
public:
    static constexpr int SCREEN_WIDTH = 1280;
    static constexpr int SCREEN_HEIGHT = 800;

    Renderer(Simulation& sim);
    ~Renderer();

    void init();
    void render();
    bool should_close() const;

private:
    Simulation& sim_;
    Camera2D camera_;
    
    // Selection state
    int selected_creature_id_ = -1;

    void handle_input();
    void draw_world();
    void draw_creatures();
    void draw_hud();
};
