#include "world/world.h"
#include <random>
#include <cmath>
#include <algorithm>

World::World() {
    grid_.resize(WIDTH * HEIGHT, TILE_EMPTY);
}

World::~World() {}

void World::init() {
    // Fill with empty space
    std::fill(grid_.begin(), grid_.end(), TILE_EMPTY);

    // 1. Build border walls
    for (int x = 0; x < WIDTH; x++) {
        grid_[0 * WIDTH + x] = TILE_ROCK;
        grid_[(HEIGHT - 1) * WIDTH + x] = TILE_ROCK;
    }
    for (int y = 0; y < HEIGHT; y++) {
        grid_[y * WIDTH + 0] = TILE_ROCK;
        grid_[y * WIDTH + (WIDTH - 1)] = TILE_ROCK;
    }

    // 2. Generate random rocks (obstacles)
    std::mt19937 rng(1337); // Seed for reproducible world layout
    std::uniform_int_distribution<int> x_dist(5, WIDTH - 6);
    std::uniform_int_distribution<int> y_dist(5, HEIGHT - 6);
    std::uniform_int_distribution<int> size_dist(2, 5);

    int num_rock_clusters = 50;
    for (int c = 0; c < num_rock_clusters; c++) {
        int cx = x_dist(rng);
        int cy = y_dist(rng);
        int size = size_dist(rng);

        for (int dy = -size; dy <= size; dy++) {
            for (int dx = -size; dx <= size; dx++) {
                if (dx*dx + dy*dy <= size*size) {
                    int tx = cx + dx;
                    int ty = cy + dy;
                    if (is_in_grid(tx, ty)) {
                        grid_[ty * WIDTH + tx] = TILE_ROCK;
                    }
                }
            }
        }
    }
}

bool World::is_walkable(float x, float y, float radius) const {
    // Check out of bounds
    if (x - radius < 0 || x + radius >= WIDTH * CELL_SIZE ||
        y - radius < 0 || y + radius >= HEIGHT * CELL_SIZE) {
        return false;
    }

    // Grid coordinates range
    int tx_start = static_cast<int>((x - radius) / CELL_SIZE);
    int tx_end   = static_cast<int>((x + radius) / CELL_SIZE);
    int ty_start = static_cast<int>((y - radius) / CELL_SIZE);
    int ty_end   = static_cast<int>((y + radius) / CELL_SIZE);

    for (int ty = ty_start; ty <= ty_end; ++ty) {
        for (int tx = tx_start; tx <= tx_end; ++tx) {
            if (!is_in_grid(tx, ty) || grid_[ty * WIDTH + tx] == TILE_ROCK) {
                // Circle-AABB intersection check
                float cell_x = tx * CELL_SIZE;
                float cell_y = ty * CELL_SIZE;
                
                // Closest point in AABB
                float closest_x = std::max(cell_x, std::min(x, cell_x + CELL_SIZE));
                float closest_y = std::max(cell_y, std::min(y, cell_y + CELL_SIZE));
                
                float dx = x - closest_x;
                float dy = y - closest_y;
                if (dx*dx + dy*dy < radius*radius) {
                    return false;
                }
            }
        }
    }
    return true;
}

void World::resolve_collision(float& x, float& y, float radius) const {
    // If it's already walkable, no need to push out
    if (is_walkable(x, y, radius)) return;

    // Check cells surrounding the creature's coordinate
    int center_tx = static_cast<int>(x / CELL_SIZE);
    int center_ty = static_cast<int>(y / CELL_SIZE);

    for (int ty = center_ty - 1; ty <= center_ty + 1; ++ty) {
        for (int tx = center_tx - 1; tx <= center_tx + 1; ++tx) {
            if (is_in_grid(tx, ty) && grid_[ty * WIDTH + tx] == TILE_ROCK) {
                float cell_x = tx * CELL_SIZE;
                float cell_y = ty * CELL_SIZE;
                
                // Find closest point on this rock tile AABB
                float closest_x = std::max(cell_x, std::min(x, cell_x + CELL_SIZE));
                float closest_y = std::max(cell_y, std::min(y, cell_y + CELL_SIZE));
                
                float dx = x - closest_x;
                float dy = y - closest_y;
                float dist_sq = dx*dx + dy*dy;
                
                if (dist_sq < radius*radius && dist_sq > 0.0001f) {
                    float dist = std::sqrt(dist_sq);
                    float overlap = radius - dist;
                    // Push out
                    x += (dx / dist) * overlap;
                    y += (dy / dist) * overlap;
                }
            }
        }
    }

    // Final boundary check and clamping
    float min_coord = radius + CELL_SIZE;
    float max_x = (WIDTH - 1) * CELL_SIZE - radius;
    float max_y = (HEIGHT - 1) * CELL_SIZE - radius;
    if (x < min_coord) x = min_coord;
    if (x > max_x) x = max_x;
    if (y < min_coord) y = min_coord;
    if (y > max_y) y = max_y;
}

TileType World::get_tile(int tx, int ty) const {
    if (is_in_grid(tx, ty)) {
        return grid_[ty * WIDTH + tx];
    }
    return TILE_ROCK;
}
