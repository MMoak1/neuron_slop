/*
 * brain.cpp — Hopfield Associative Memory, 10,000 neurons
 *
 * Each "idea" is a pattern of which neurons are on (+1) or off (-1).
 * We store them via Hebbian learning ("neurons that fire together wire together").
 * Given a corrupted/partial cue, the network dynamics recover the nearest stored idea.
 *
 * No weight matrix stored — we use the pattern-overlap trick:
 *   h_i = Σ_μ  pattern_μ[i] * ⟨pattern_μ, state⟩
 * and update s_i = sign(h_i).
 * Overlaps are maintained incrementally, so each full sweep is O(N·P).
 *
 * Compile:  g++ -O2 -o brain brain.cpp
 * Run:      ./brain
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

// ── Constants ───────────────────────────────────────────────
static constexpr int SIDE = 100;
static constexpr int N    = SIDE * SIDE;  // 10,000 neurons

using Pattern = std::vector<int8_t>;  // +1 or -1

// ── Pattern Generators ──────────────────────────────────────

Pattern make_circle() {
    Pattern p(N, -1);
    for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++) {
            int dx = x - 50, dy = y - 50;
            if (dx*dx + dy*dy <= 30*30)
                p[y*SIDE + x] = 1;
        }
    return p;
}

Pattern make_cross() {
    Pattern p(N, -1);
    for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++) {
            bool h = (y >= 42 && y <= 58);
            bool v = (x >= 42 && x <= 58);
            if (h || v) p[y*SIDE + x] = 1;
        }
    return p;
}

Pattern make_X() {
    Pattern p(N, -1);
    for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++) {
            if (abs(x - y) <= 7 || abs(x - (SIDE-1-y)) <= 7)
                p[y*SIDE + x] = 1;
        }
    return p;
}

Pattern make_ring() {
    Pattern p(N, -1);
    for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++) {
            int dx = x - 50, dy = y - 50;
            int r2 = dx*dx + dy*dy;
            if (r2 >= 25*25 && r2 <= 38*38)
                p[y*SIDE + x] = 1;
        }
    return p;
}

Pattern make_stripes() {
    Pattern p(N, -1);
    for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++) {
            if ((y / 10) % 2 == 0)
                p[y*SIDE + x] = 1;
        }
    return p;
}

Pattern make_diamond() {
    Pattern p(N, -1);
    for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++) {
            if (abs(x - 50) + abs(y - 50) <= 35)
                p[y*SIDE + x] = 1;
        }
    return p;
}

// ── Display ─────────────────────────────────────────────────
// Uses half-block characters: ▀ ▄ █ to pack 2 rows per line

void display(const Pattern& p, const char* label) {
    printf("\n  \033[1;33m%s\033[0m\n", label);
    for (int y = 0; y < SIDE; y += 2) {
        printf("  ");
        for (int x = 0; x < SIDE; x++) {
            bool top = p[y * SIDE + x] > 0;
            bool bot = (y + 1 < SIDE) && p[(y+1) * SIDE + x] > 0;

            if      ( top &&  bot) printf("\033[97m█\033[0m");
            else if ( top && !bot) printf("\033[97m▀\033[0m");
            else if (!top &&  bot) printf("\033[97m▄\033[0m");
            else                   printf("\033[38;5;236m·\033[0m");
        }
        printf("\n");
    }
}

// side-by-side display: show two patterns next to each other
void display_pair(const Pattern& a, const char* labelA,
                  const Pattern& b, const char* labelB) {
    // print labels
    printf("\n  \033[1;33m%-50s %s\033[0m\n", labelA, labelB);
    for (int y = 0; y < SIDE; y += 2) {
        printf("  ");
        // left pattern
        for (int x = 0; x < SIDE; x++) {
            bool top = a[y * SIDE + x] > 0;
            bool bot = (y + 1 < SIDE) && a[(y+1) * SIDE + x] > 0;
            if      ( top &&  bot) printf("\033[97m█\033[0m");
            else if ( top && !bot) printf("\033[97m▀\033[0m");
            else if (!top &&  bot) printf("\033[97m▄\033[0m");
            else                   printf("\033[38;5;236m·\033[0m");
        }
        printf("    ");
        // right pattern
        for (int x = 0; x < SIDE; x++) {
            bool top = b[y * SIDE + x] > 0;
            bool bot = (y + 1 < SIDE) && b[(y+1) * SIDE + x] > 0;
            if      ( top &&  bot) printf("\033[97m█\033[0m");
            else if ( top && !bot) printf("\033[97m▀\033[0m");
            else if (!top &&  bot) printf("\033[97m▄\033[0m");
            else                   printf("\033[38;5;236m·\033[0m");
        }
        printf("\n");
    }
}

// ── Overlap (dot product / N) ───────────────────────────────

float calc_overlap(const Pattern& a, const Pattern& b) {
    int dot = 0;
    for (int i = 0; i < N; i++) dot += (int)a[i] * (int)b[i];
    return (float)dot / N;
}

// ── Corruption ──────────────────────────────────────────────

Pattern corrupt(const Pattern& p, float frac, unsigned seed = 42) {
    std::mt19937 rng(seed);
    Pattern noisy = p;
    int nFlip = (int)(N * frac);

    std::vector<int> idx(N);
    for (int i = 0; i < N; i++) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);

    for (int i = 0; i < nFlip; i++)
        noisy[idx[i]] *= -1;
    return noisy;
}

// ── Hopfield Recall ─────────────────────────────────────────
// Asynchronous updates with incremental overlap tracking.
// Each full sweep is O(N · P) — blazing fast for 10k neurons.

struct RecallResult {
    int sweeps;
    double ms;
    int totalFlips;
};

RecallResult recall(Pattern& state,
                    const std::vector<Pattern>& memories,
                    const std::vector<const char*>& names,
                    int maxSweeps = 30,
                    bool verbose = true)
{
    std::mt19937 rng(12345);
    int P = (int)memories.size();

    // precompute overlaps: m[μ] = ⟨memory_μ, state⟩
    std::vector<int> m(P, 0);
    for (int mu = 0; mu < P; mu++)
        for (int i = 0; i < N; i++)
            m[mu] += memories[mu][i] * state[i];

    std::vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i;

    int totalFlips = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int sweep = 0; sweep < maxSweeps; sweep++) {
        std::shuffle(order.begin(), order.end(), rng);
        int flips = 0;

        for (int idx = 0; idx < N; idx++) {
            int i = order[idx];

            // local field: h_i = Σ_μ memory_μ[i] · m[μ]
            int h = 0;
            for (int mu = 0; mu < P; mu++)
                h += memories[mu][i] * m[mu];

            int8_t newS = (h >= 0) ? 1 : -1;

            if (newS != state[i]) {
                int delta = newS - state[i]; // ±2
                for (int mu = 0; mu < P; mu++)
                    m[mu] += memories[mu][i] * delta;
                state[i] = newS;
                flips++;
            }
        }

        totalFlips += flips;

        if (verbose) {
            printf("    sweep %2d │ %5d flips │ ", sweep + 1, flips);
            for (int mu = 0; mu < P; mu++) {
                float ov = (float)m[mu] / N;
                // highlight the dominant overlap
                if (ov > 0.9f)
                    printf("\033[32m%s:%.3f\033[0m ", names[mu], ov);
                else if (ov > 0.5f)
                    printf("\033[33m%s:%.3f\033[0m ", names[mu], ov);
                else
                    printf("\033[90m%s:%.3f\033[0m ", names[mu], ov);
            }
            printf("\n");
        }

        if (flips == 0) {
            if (verbose) printf("    \033[32m✓ converged\033[0m\n");
            break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return { totalFlips > 0 ? -1 : 0, ms, totalFlips };
}

// ═══════════════════════════════════════════════════════════════
int main() {
    printf("\033[1;36m");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║   HOPFIELD ASSOCIATIVE MEMORY — 10,000 NEURONS       ║\n");
    printf("║   \"neurons that fire together wire together\"          ║\n");
    printf("║                                                       ║\n");
    printf("║   Store ideas as neuron patterns.                     ║\n");
    printf("║   Corrupt them. Watch the brain recover the memory.   ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\033[0m\n");

    // ── Store ideas ──
    std::vector<Pattern>     memories;
    std::vector<const char*> names;

    memories.push_back(make_circle());   names.push_back("CIRCLE");
    memories.push_back(make_cross());    names.push_back("CROSS");
    memories.push_back(make_X());        names.push_back("X-SHAPE");
    memories.push_back(make_ring());     names.push_back("RING");
    memories.push_back(make_stripes()); names.push_back("STRIPES");
    memories.push_back(make_diamond()); names.push_back("DIAMOND");

    int P = (int)memories.size();
    printf("  Stored \033[1m%d ideas\033[0m across \033[1m%d neurons\033[0m\n", P, N);
    printf("  Theoretical capacity: ~%d ideas (N / 2·ln(N))\n\n",
           (int)(N / (2.0 * log(N))));

    // ── Show stored patterns ──
    printf("══ STORED IDEAS ══════════════════════════════════════\n");
    for (int mu = 0; mu < P; mu++) {
        int active = 0;
        for (int i = 0; i < N; i++) if (memories[mu][i] > 0) active++;
        char buf[128];
        snprintf(buf, sizeof(buf), "%s  (%d neurons active, %.0f%%)",
                 names[mu], active, 100.0f * active / N);
        display(memories[mu], buf);
    }

    // ── Overlap matrix ──
    printf("\n══ IDEA SIMILARITY ═══════════════════════════════════\n");
    printf("  (ideal: 1.0 on diagonal, near 0.0 off-diagonal)\n\n");
    printf("  %10s", "");
    for (int mu = 0; mu < P; mu++) printf(" %8s", names[mu]);
    printf("\n");
    for (int mu = 0; mu < P; mu++) {
        printf("  %10s", names[mu]);
        for (int nu = 0; nu < P; nu++) {
            float ov = calc_overlap(memories[mu], memories[nu]);
            if (mu == nu)
                printf("  \033[1;32m%5.3f\033[0m", ov);
            else if (fabs(ov) > 0.2)
                printf("  \033[33m%5.3f\033[0m", ov);
            else
                printf("  \033[90m%5.3f\033[0m", ov);
        }
        printf("\n");
    }

    // ── Recall tests at various noise levels ──
    printf("\n══ RECALL FROM NOISY MEMORY ═══════════════════════════\n\n");

    struct Test { int idea; float noise; };
    Test tests[] = {
        {0, 0.20f}, // circle, 20% noise
        {1, 0.30f}, // cross, 30%
        {2, 0.25f}, // X-shape, 25%
        {3, 0.30f}, // ring, 30%
        {5, 0.35f}, // diamond, 35%
        {0, 0.45f}, // circle pushed hard
    };

    for (auto& t : tests) {
        char lbl[128];
        snprintf(lbl, sizeof(lbl),
                 "Recalling \"%s\" from %.0f%% noise", names[t.idea], t.noise * 100);
        printf("  ┌─ %s ──\n", lbl);

        Pattern state = corrupt(memories[t.idea], t.noise,
                                t.idea * 31 + (int)(t.noise * 137));

        float initOv = calc_overlap(state, memories[t.idea]);
        printf("    initial overlap: %.3f\n", initOv);

        auto res = recall(state, memories, names, 30, true);

        // find best match
        int bestMu = 0; float bestOv = -2;
        for (int mu = 0; mu < P; mu++) {
            float ov = calc_overlap(state, memories[mu]);
            if (ov > bestOv) { bestOv = ov; bestMu = mu; }
        }

        if (bestMu == t.idea && bestOv > 0.95)
            printf("    \033[1;32m✓ Recalled \"%s\"  (overlap %.4f, %.1fms)\033[0m\n",
                   names[bestMu], bestOv, res.ms);
        else if (bestMu == t.idea)
            printf("    \033[1;33m~ Partial recall \"%s\"  (overlap %.4f, %.1fms)\033[0m\n",
                   names[bestMu], bestOv, res.ms);
        else
            printf("    \033[1;31m✗ Got \"%s\" instead  (overlap %.4f, %.1fms)\033[0m\n",
                   names[bestMu], bestOv, res.ms);
        printf("  └──────────────────────────────────────────\n\n");
    }

    // ── Visual demo ──
    printf("══ VISUAL DEMO ═══════════════════════════════════════\n");

    int demo = 5; // diamond
    float demoNoise = 0.30f;

    Pattern noisy = corrupt(memories[demo], demoNoise, 777);

    display_pair(memories[demo], "ORIGINAL",
                 noisy,          "CORRUPTED (30% noise)");

    printf("\n  Running recall...\n");
    recall(noisy, memories, names, 30, true);

    display_pair(memories[demo], "ORIGINAL",
                 noisy,          "AFTER RECALL");

    printf("  final overlap: \033[1;32m%.4f\033[0m\n", calc_overlap(noisy, memories[demo]));

    // ── Association test ──
    printf("\n══ ASSOCIATION TEST ═══════════════════════════════════\n");
    printf("  What happens if we mix two ideas?\n");
    printf("  Top half = CIRCLE, bottom half = CROSS\n\n");

    Pattern mixed(N, -1);
    for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++) {
            int i = y * SIDE + x;
            mixed[i] = (y < SIDE/2) ? memories[0][i]   // circle top
                                    : memories[1][i];  // cross bottom
        }

    display(mixed, "MIXED INPUT");
    printf("  initial overlaps: ");
    for (int mu = 0; mu < P; mu++)
        printf("%s=%.3f ", names[mu], calc_overlap(mixed, memories[mu]));
    printf("\n\n");

    recall(mixed, memories, names, 30, true);

    int bestMu = 0; float bestOv = -2;
    for (int mu = 0; mu < P; mu++) {
        float ov = calc_overlap(mixed, memories[mu]);
        if (ov > bestOv) { bestOv = ov; bestMu = mu; }
    }

    display(mixed, "AFTER RECALL");
    printf("\n  \033[1;36m→ The brain associated this with: \"%s\" (overlap %.3f)\033[0m\n",
           names[bestMu], bestOv);
    printf("  The network settled into the nearest stored idea — that's associative memory.\n\n");

    return 0;
}
