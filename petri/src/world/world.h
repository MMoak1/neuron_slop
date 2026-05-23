#pragma once

#include <vector>
#include <cstdint>

enum TileType : uint8_t {
    TILE_EMPTY = 0,
    TILE_ROCK = 1
};

class World {
public:
    static constexpr int WIDTH = 128;
    static constexpr int HEIGHT = 128;
    static constexpr float CELL_SIZE = 16.0f; // Dimension of each tile in world space

    World();
    ~World();

    void init();

    // Check if a point (x, y) in world space collides with obstacles or borders
    bool is_walkable(float x, float y, float radius) const;

    // Check collision and adjust position sliding along obstacles
    void resolve_collision(float& x, float& y, float radius) const;

    TileType get_tile(int tx, int ty) const;

private:
    std::vector<TileType> grid_;

    bool is_in_grid(int tx, int ty) const {
        return tx >= 0 && tx < WIDTH && ty >= 0 && ty < HEIGHT;
    }
};
