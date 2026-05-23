#include "render/renderer.h"
#include <string>
#include <cmath>
#include <algorithm>

Renderer::Renderer(Simulation& sim) : sim_(sim) {
    camera_ = { 0 };
}

Renderer::~Renderer() {
    CloseWindow();
}

void Renderer::init() {
    // Enable Anti-Aliasing (Multisampling)
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Petri Dish: Spiking Neural Ecosystem (Phase 1)");
    SetTargetFPS(60);

    // Center camera on world
    camera_.target = { (World::WIDTH * World::CELL_SIZE) / 2.0f, (World::HEIGHT * World::CELL_SIZE) / 2.0f };
    camera_.offset = { SCREEN_WIDTH / 2.0f, SCREEN_HEIGHT / 2.0f };
    camera_.rotation = 0.0f;
    camera_.zoom = 0.4f;
}

bool Renderer::should_close() const {
    return WindowShouldClose();
}

void Renderer::render() {
    handle_input();

    BeginDrawing();
    ClearBackground(GetColor(0x111115FF)); // Dark cosmic background

    BeginMode2D(camera_);

    draw_world();
    draw_creatures();

    EndMode2D();

    draw_hud();

    EndDrawing();
}

void Renderer::handle_input() {
    // 1. Mouse Drag Panning (Right or Middle Click)
    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
        Vector2 delta = GetMouseDelta();
        // Scale panning speed by current zoom level
        delta.x = -delta.x / camera_.zoom;
        delta.y = -delta.y / camera_.zoom;
        camera_.target.x += delta.x;
        camera_.target.y += delta.y;
    }

    // 2. Mouse Wheel Zoom (focus zoom on mouse position)
    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        Vector2 mouse_world_pos = GetScreenToWorld2D(GetMousePosition(), camera_);
        
        // Adjust zoom
        float zoom_factor = 1.15f;
        if (wheel > 0) camera_.zoom *= zoom_factor;
        else camera_.zoom /= zoom_factor;

        // Clamp zoom level
        camera_.zoom = std::clamp(camera_.zoom, 0.05f, 5.0f);

        // Center zoom on mouse pointer
        Vector2 new_mouse_world_pos = GetScreenToWorld2D(GetMousePosition(), camera_);
        camera_.target.x += (mouse_world_pos.x - new_mouse_world_pos.x);
        camera_.target.y += (mouse_world_pos.y - new_mouse_world_pos.y);
    }

    // 3. Click to Select Creature (Left Click)
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_world = GetScreenToWorld2D(GetMousePosition(), camera_);
        
        int best_id = -1;
        float best_dist_sq = 1e9f;
        float selection_radius = 12.0f; // Click tolerance in world space

        const auto& creatures = sim_.get_creatures();
        for (const auto& c : creatures) {
            if (!c.alive) continue;
            float dx = mouse_world.x - c.x;
            float dy = mouse_world.y - c.y;
            float dist_sq = dx*dx + dy*dy;
            if (dist_sq < selection_radius * selection_radius && dist_sq < best_dist_sq) {
                best_id = c.id;
                best_dist_sq = dist_sq;
            }
        }

        selected_creature_id_ = best_id;
    }

    // 4. Hotkeys
    if (IsKeyPressed(KEY_R)) {
        // Reset camera view
        camera_.target = { (World::WIDTH * World::CELL_SIZE) / 2.0f, (World::HEIGHT * World::CELL_SIZE) / 2.0f };
        camera_.zoom = 0.4f;
    }
}

void Renderer::draw_world() {
    const World& world = sim_.get_world();

    // Draw Grid Lines (Blueprint style, very subtle)
    Color grid_color = GetColor(0x1c1c24FF);
    for (int x = 0; x <= World::WIDTH; x++) {
        float px = x * World::CELL_SIZE;
        DrawLineV({ px, 0 }, { px, World::HEIGHT * World::CELL_SIZE }, grid_color);
    }
    for (int y = 0; y <= World::HEIGHT; y++) {
        float py = y * World::CELL_SIZE;
        DrawLineV({ 0, py }, { World::WIDTH * World::CELL_SIZE, py }, grid_color);
    }

    // Draw Tiles (only Rocks in Phase 1)
    for (int y = 0; y < World::HEIGHT; y++) {
        for (int x = 0; x < World::WIDTH; x++) {
            TileType tile = world.get_tile(x, y);
            if (tile == TILE_ROCK) {
                float px = x * World::CELL_SIZE;
                float py = y * World::CELL_SIZE;
                
                // Draw Rock tile with premium outline and gradient-like structure
                DrawRectangle(px, py, World::CELL_SIZE, World::CELL_SIZE, GetColor(0x282c35FF));
                // Lighter top-left highlight border
                DrawLine(px, py, px + World::CELL_SIZE, py, GetColor(0x3e4554FF));
                DrawLine(px, py, px, py + World::CELL_SIZE, GetColor(0x3e4554FF));
                // Darker bottom-right border
                DrawLine(px + World::CELL_SIZE - 1, py, px + World::CELL_SIZE - 1, py + World::CELL_SIZE, GetColor(0x1a1c22FF));
                DrawLine(px, py + World::CELL_SIZE - 1, px + World::CELL_SIZE, py + World::CELL_SIZE - 1, GetColor(0x1a1c22FF));
            }
        }
    }
}

void Renderer::draw_creatures() {
    const auto& creatures = sim_.get_creatures();

    for (const auto& c : creatures) {
        if (!c.alive) continue;

        // Draw sensory vision rays for the selected creature
        if (c.id == selected_creature_id_) {
            float view_len = Simulation::MAX_VIEW_DIST;
            float fl = sim_.cast_ray(c.x, c.y, c.angle, view_len);
            float ll = sim_.cast_ray(c.x, c.y, c.angle - 3.14159f / 4.0f, view_len);
            float lr = sim_.cast_ray(c.x, c.y, c.angle + 3.14159f / 4.0f, view_len);

            // Draw clean laser lines
            DrawLineV({ c.x, c.y }, { c.x + std::cos(c.angle) * fl, c.y + std::sin(c.angle) * fl }, GetColor(0x00ffffff));
            DrawLineV({ c.x, c.y }, { c.x + std::cos(c.angle - 3.14159f/4.0f) * ll, c.y + std::sin(c.angle - 3.14159f/4.0f) * ll }, GetColor(0x00ff8844));
            DrawLineV({ c.x, c.y }, { c.x + std::cos(c.angle + 3.14159f/4.0f) * lr, c.y + std::sin(c.angle + 3.14159f/4.0f) * lr }, GetColor(0x00ff8844));
        }

        // Color based on active speed (visualizing energy/movement)
        float speed_factor = c.speed / 2.5f;
        Color body_color = GetColor(0x00f0ffFF); // Cyan
        if (speed_factor > 0.6f) {
            body_color = GetColor(0x00ff88FF); // Vibrant green
        } else if (speed_factor < 0.1f) {
            body_color = GetColor(0x4facfeFF); // Deep blue
        }

        // Selected highlights
        if (c.id == selected_creature_id_) {
            DrawCircleSectorLines({ c.x, c.y }, Simulation::CREATURE_RADIUS + 6.0f, 0.0f, 360.0f, 24, GetColor(0xffd700FF));
            body_color = GetColor(0xffd700FF); // Selection Gold
        }

        // Draw creature body circle
        DrawCircle(c.x, c.y, Simulation::CREATURE_RADIUS, body_color);
        DrawCircleLines(c.x, c.y, Simulation::CREATURE_RADIUS, GetColor(0x0a0a0dFF));

        // Draw facing direction vector (line)
        float dir_x = c.x + std::cos(c.angle) * (Simulation::CREATURE_RADIUS + 3.0f);
        float dir_y = c.y + std::sin(c.angle) * (Simulation::CREATURE_RADIUS + 3.0f);
        DrawLineEx({ c.x, c.y }, { dir_x, dir_y }, 2.0f, GetColor(0xff007fFF));
    }
}

void Renderer::draw_hud() {
    // 1. Simulation Statistics Card (Top Left)
    DrawRectangleRounded({ 15, 15, 280, 110 }, 0.12f, 8, GetColor(0x18181ce0));
    DrawRectangleRoundedLines({ 15, 15, 280, 110 }, 0.12f, 8, 1, GetColor(0x353a4788));
    
    DrawText("PETRI DISH ECOSYSTEM", 30, 25, 15, GetColor(0xffd700FF));
    
    std::string ticks_str = "Tick Count: " + std::to_string(sim_.get_tick_count());
    DrawText(ticks_str.c_str(), 30, 48, 13, RAYWHITE);

    std::string count_str = "Active Population: " + std::to_string(Simulation::MAX_CREATURES);
    DrawText(count_str.c_str(), 30, 68, 13, RAYWHITE);

    std::string help_str = "R: Reset Camera | R-Drag: Pan | Scroll: Zoom";
    DrawText(help_str.c_str(), 30, 92, 10, LIGHTGRAY);

    // 2. Creature Inspector Card (Top Right, glassmorphism)
    if (selected_creature_id_ != -1) {
        const auto& creatures = sim_.get_creatures();
        const Creature& c = creatures[selected_creature_id_];

        DrawRectangleRounded({ SCREEN_WIDTH - 315, 15, 300, 180 }, 0.08f, 8, GetColor(0x18181ce8));
        DrawRectangleRoundedLines({ SCREEN_WIDTH - 315, 15, 300, 180 }, 0.08f, 8, 1, GetColor(0xffd700aa));

        DrawText("CREATURE INSPECTOR", SCREEN_WIDTH - 295, 25, 15, GetColor(0xffd700FF));
        
        std::string id_str = "Creature ID: #" + std::to_string(c.id);
        DrawText(id_str.c_str(), SCREEN_WIDTH - 295, 48, 13, RAYWHITE);

        std::string pos_str = "Position: (" + std::to_string((int)c.x) + ", " + std::to_string((int)c.y) + ")";
        DrawText(pos_str.c_str(), SCREEN_WIDTH - 295, 68, 13, RAYWHITE);

        std::string speed_str = "Velocity: " + std::to_string(c.speed).substr(0, 4) + " px/t";
        DrawText(speed_str.c_str(), SCREEN_WIDTH - 295, 88, 13, RAYWHITE);

        std::string heading_str = "Heading: " + std::to_string(c.angle * 180.0f / 3.14159f).substr(0, 5) + " deg";
        DrawText(heading_str.c_str(), SCREEN_WIDTH - 295, 108, 13, RAYWHITE);

        std::string energy_str = "Energy level: " + std::to_string(c.energy * 100.0f).substr(0, 5) + "%";
        DrawText(energy_str.c_str(), SCREEN_WIDTH - 295, 128, 13, RAYWHITE);

        DrawText("Brain type: Fully-recurrent LIF", SCREEN_WIDTH - 295, 155, 11, GRAY);
    }
}
