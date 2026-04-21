/*
 * brain3.cpp — A Brain That Learns From Experience
 *
 * The brain starts knowing NOTHING.
 * A tiny world generates experiences (encounters with things).
 * The brain observes which features co-occur and discovers:
 *   - Concepts (clusters of co-occurring features)
 *   - Names (labels that attach to clusters)
 *   - Categories (shared structure across concepts)
 *
 * NO knowledge is hardcoded into the brain. Everything is learned.
 *
 * Compile: g++ -O2 -std=c++17 -o brain3 brain3.cpp
 */

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <random>
#include <numeric>

static std::mt19937 g_rng(42);

// ════════════════════════════════════════════════════════════
// THE WORLD — the brain has zero access to this structure.
// It only sees streams of features from experiences.
// ════════════════════════════════════════════════════════════

struct Entity {
    std::string name; // label — brain must LEARN this
    std::vector<std::string> properties;
    std::vector<std::string> behaviors;
    float frequency; // how often encountered
};

static std::vector<Entity> g_world = {
    {"cat",     {"furry","small","four_legs","whiskers","tail","soft","warm","eyes"},
                {"meows","purrs","stretches","climbs","pounces"}, 1.0f},
    {"dog",     {"furry","medium","four_legs","floppy_ears","tail","warm","snout","eyes"},
                {"barks","wags_tail","fetches","pants","sniffs"}, 1.0f},
    {"bird",    {"feathered","small","two_legs","wings","beak","light","eyes"},
                {"chirps","flies","pecks","nests","hops"}, 0.8f},
    {"fish",    {"scaly","small","fins","cold","wet","slimy","eyes"},
                {"swims","bubbles","glides"}, 0.6f},
    {"snake",   {"scaly","long","no_legs","cold","flexible","eyes"},
                {"slithers","hisses","coils","strikes"}, 0.3f},
    {"horse",   {"furry","large","four_legs","hooves","tail","strong","warm","mane"},
                {"gallops","neighs","trots","grazes"}, 0.4f},
    {"tree",    {"tall","green","bark_texture","leaves","roots","still","woody"},
                {"sways","grows","sheds_leaves"}, 0.7f},
    {"flower",  {"colorful","small","soft","fragrant","petals","green","delicate"},
                {"blooms","sways","wilts","attracts_bees"}, 0.5f},
    {"car",     {"metallic","large","four_wheels","engine","fast","shiny","heavy"},
                {"drives","honks","accelerates","idles"}, 0.8f},
    {"bicycle", {"metallic","medium","two_wheels","pedals","light","thin"},
                {"rides","coasts","clicks"}, 0.5f},
    {"rock",    {"hard","heavy","grey","cold","still","rough","solid"},
                {"sits","erodes"}, 0.4f},
    {"fire",    {"hot","bright","orange","dangerous","flickering","light_emitting"},
                {"burns","crackles","spreads","glows"}, 0.3f},
    {"rain",    {"wet","cold","falling","grey_sky","drops","fresh"},
                {"falls","splashes","drips","soaks"}, 0.4f},
    {"sun",     {"bright","hot","yellow","round","far_away","large","light_emitting"},
                {"shines","warms","rises","sets"}, 0.3f},
};

// Generate one experience: a noisy, partial encounter with an entity.
struct Experience {
    std::vector<std::string> features; // everything observed together
    std::string true_entity;           // for our stats only — brain never sees this
};

Experience generate_experience() {
    // weighted random entity selection
    float total = 0;
    for (auto& e : g_world) total += e.frequency;
    float roll = std::uniform_real_distribution<float>(0, total)(g_rng);
    float cum = 0;
    Entity* ent = &g_world[0];
    for (auto& e : g_world) { cum += e.frequency; if (cum >= roll) { ent = &e; break; } }

    Experience exp;
    exp.true_entity = ent->name;

    // observe a random SUBSET of properties (60-90%)
    for (auto& p : ent->properties)
        if (std::uniform_real_distribution<float>(0,1)(g_rng) < 0.75f)
            exp.features.push_back(p);

    // observe a random SUBSET of behaviors (40-70%)
    for (auto& b : ent->behaviors)
        if (std::uniform_real_distribution<float>(0,1)(g_rng) < 0.55f)
            exp.features.push_back(b);

    // sometimes hear the label (50% of the time)
    if (std::uniform_real_distribution<float>(0,1)(g_rng) < 0.50f)
        exp.features.push_back("called:" + ent->name);

    return exp;
}

// ════════════════════════════════════════════════════════════
// THE BRAIN — starts completely empty
// ════════════════════════════════════════════════════════════

struct Brain {
    // feature registry — grows as new features are encountered
    std::unordered_map<std::string, int> feat2idx;
    std::vector<std::string> idx2feat;
    int nf = 0;

    // co-occurrence matrix and counts
    std::vector<std::vector<int>> cooccur; // cooccur[i][j] = times i and j appeared together
    std::vector<int> count;                // count[i] = times feature i appeared at all
    int totalExp = 0;

    // learned concept clusters
    struct Concept {
        std::vector<int> members;   // feature indices in this cluster
        std::string label;          // discovered name (if any)
        int observations;           // how many experiences contributed
    };
    std::vector<Concept> concepts;

    int feat(const std::string& f) {
        auto it = feat2idx.find(f);
        if (it != feat2idx.end()) return it->second;
        int id = nf++;
        feat2idx[f] = id;
        idx2feat.push_back(f);
        cooccur.resize(nf);
        for (auto& row : cooccur) row.resize(nf, 0);
        count.resize(nf, 0);
        return id;
    }

    // ── Learn from one experience ──────────────────────────
    void observe(const Experience& exp) {
        totalExp++;
        std::vector<int> ids;
        for (auto& f : exp.features) ids.push_back(feat(f));

        for (int i : ids) count[i]++;
        for (int i = 0; i < (int)ids.size(); i++)
            for (int j = i + 1; j < (int)ids.size(); j++) {
                cooccur[ids[i]][ids[j]]++;
                cooccur[ids[j]][ids[i]]++;
            }
    }

    // ── Conditional probability P(j | i) ───────────────────
    float condProb(int i, int j) const {
        if (count[i] == 0) return 0;
        return (float)cooccur[i][j] / count[i];
    }

    // Pointwise mutual information
    float pmi(int i, int j) const {
        if (cooccur[i][j] == 0) return -10;
        float pij = (float)cooccur[i][j] / totalExp;
        float pi = (float)count[i] / totalExp;
        float pj = (float)count[j] / totalExp;
        return log2f(pij / (pi * pj));
    }

    // ── Discover concepts via clustering ───────────────────
    void discoverConcepts(float pmiThreshold = 1.5f, int minClusterSize = 3) {
        concepts.clear();
        std::vector<bool> assigned(nf, false);

        // find seed: unassigned feature with strongest max-PMI connection
        while (true) {
            int bestSeed = -1;
            float bestPMI = -999;

            for (int i = 0; i < nf; i++) {
                if (assigned[i] || count[i] < 3) continue;
                for (int j = i + 1; j < nf; j++) {
                    if (assigned[j] || count[j] < 3) continue;
                    float p = pmi(i, j);
                    if (p > bestPMI) { bestPMI = p; bestSeed = i; }
                }
            }

            if (bestSeed < 0 || bestPMI < pmiThreshold) break;

            // grow cluster from seed
            Concept c;
            c.observations = count[bestSeed];
            std::vector<int> cluster = {bestSeed};
            assigned[bestSeed] = true;

            // add features with high avg PMI to existing cluster members
            bool grew = true;
            while (grew) {
                grew = false;
                for (int i = 0; i < nf; i++) {
                    if (assigned[i] || count[i] < 3) continue;
                    // avg PMI with cluster members
                    float avgP = 0;
                    for (int m : cluster) avgP += pmi(i, m);
                    avgP /= cluster.size();

                    if (avgP > pmiThreshold * 0.7f) {
                        cluster.push_back(i);
                        assigned[i] = true;
                        grew = true;
                    }
                }
            }

            if ((int)cluster.size() < minClusterSize) {
                for (int m : cluster) assigned[m] = false;
                // mark seed so we don't re-pick it
                assigned[bestSeed] = true;
                continue;
            }

            c.members = cluster;

            // check if any label features are in the cluster
            for (int m : cluster) {
                if (idx2feat[m].substr(0, 7) == "called:") {
                    c.label = idx2feat[m].substr(7);
                    break;
                }
            }

            concepts.push_back(c);
        }
    }

    // ── Query: what do you know about these features? ──────
    struct QueryResult {
        int conceptIdx;
        float matchScore;
    };

    std::vector<QueryResult> query(const std::vector<std::string>& queryFeatures) const {
        // convert to indices
        std::set<int> qset;
        for (auto& f : queryFeatures) {
            auto it = feat2idx.find(f);
            if (it != feat2idx.end()) qset.insert(it->second);
        }

        std::vector<QueryResult> results;
        for (int ci = 0; ci < (int)concepts.size(); ci++) {
            auto& c = concepts[ci];
            // how many query features appear in this concept?
            int hits = 0;
            for (int m : c.members)
                if (qset.count(m)) hits++;
            float score = qset.empty() ? 0 : (float)hits / qset.size();
            if (score > 0.01f)
                results.push_back({ci, score});
        }

        std::sort(results.begin(), results.end(),
                  [](auto& a, auto& b) { return a.matchScore > b.matchScore; });
        return results;
    }

    // ── Recognize: what is this novel thing? ───────────────
    std::vector<QueryResult> recognize(const std::vector<std::string>& features) const {
        return query(features);
    }

    // ── Complete: given partial features, what else? ────────
    std::vector<std::pair<std::string, float>> complete(
            const std::vector<std::string>& known) const {
        // activate known features, find associated features via co-occurrence
        std::set<int> knownIdx;
        for (auto& f : known) {
            auto it = feat2idx.find(f);
            if (it != feat2idx.end()) knownIdx.insert(it->second);
        }

        std::vector<std::pair<std::string, float>> predictions;
        for (int j = 0; j < nf; j++) {
            if (knownIdx.count(j)) continue;
            // average conditional probability P(j | known features)
            float avg = 0;
            for (int k : knownIdx) avg += condProb(k, j);
            if (!knownIdx.empty()) avg /= knownIdx.size();
            if (avg > 0.05f)
                predictions.push_back({idx2feat[j], avg});
        }

        std::sort(predictions.begin(), predictions.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });
        return predictions;
    }
};

// ── Printing helpers ───────────────────────────────────────

void print_bar(float v, int w = 20, const char* col = "\033[36m") {
    int f = (int)(v * w);
    printf("%s", col);
    for (int i = 0; i < f; i++) printf("█");
    printf("\033[90m");
    for (int i = f; i < w; i++) printf("░");
    printf("\033[0m");
}

// ════════════════════════════════════════════════════════════
int main() {
    printf("\033[1;35m");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║   A BRAIN THAT LEARNS FROM EXPERIENCE                    ║\n");
    printf("║                                                          ║\n");
    printf("║   The brain starts completely empty.                     ║\n");
    printf("║   It observes a world. It forms its own concepts.        ║\n");
    printf("║   Nothing is hardcoded. Everything is learned.           ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("\033[0m\n");

    Brain brain;

    printf("══ THE WORLD ═════════════════════════════════════════════\n\n");
    printf("  The world has %d types of things.\n", (int)g_world.size());
    printf("  The brain knows NONE of them. It will experience them.\n\n");

    // ── Generate experiences ────────────────────────────────
    int N_EXP = 2000;
    printf("══ EXPERIENCING THE WORLD ════════════════════════════════\n\n");
    printf("  Generating %d experiences...\n\n", N_EXP);

    // show first few experiences
    printf("  \033[90mSample experiences (what the brain actually sees):\033[0m\n\n");

    std::unordered_map<std::string, int> entityCounts;

    for (int i = 0; i < N_EXP; i++) {
        Experience exp = generate_experience();
        entityCounts[exp.true_entity]++;

        if (i < 8) {
            printf("    exp %3d │ ", i + 1);
            for (int f = 0; f < (int)exp.features.size(); f++) {
                if (f > 0) printf(", ");
                // color labels differently
                if (exp.features[f].substr(0, 7) == "called:")
                    printf("\033[1;33m%s\033[0m", exp.features[f].c_str());
                else
                    printf("%s", exp.features[f].c_str());
            }
            printf("\n");
        }
        if (i == 8) printf("    ...\n");

        brain.observe(exp);
    }

    printf("\n  Brain now has %d unique features from %d experiences.\n\n",
           brain.nf, brain.totalExp);

    // ── Show what the brain learned about co-occurrence ─────
    printf("══ CO-OCCURRENCE MATRIX (top pairs) ══════════════════════\n\n");
    printf("  These feature pairs appeared together most often.\n");
    printf("  The brain discovered this structure, not us.\n\n");

    struct FPair { int i, j; float p; };
    std::vector<FPair> pairs;
    for (int i = 0; i < brain.nf; i++)
        for (int j = i+1; j < brain.nf; j++)
            if (brain.pmi(i,j) > 2.0f)
                pairs.push_back({i, j, brain.pmi(i,j)});
    std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return a.p > b.p; });

    for (int k = 0; k < std::min(15, (int)pairs.size()); k++) {
        auto& p = pairs[k];
        printf("    %-20s ↔ %-20s  PMI: ",
               brain.idx2feat[p.i].c_str(), brain.idx2feat[p.j].c_str());
        print_bar(p.p / 8.0f, 15, "\033[32m");
        printf(" %.2f\n", p.p);
    }

    // ── Discover concepts ───────────────────────────────────
    printf("\n══ CONCEPT DISCOVERY ══════════════════════════════════════\n\n");
    printf("  Clustering co-occurring features into concepts...\n");
    printf("  The brain is figuring out what 'things' exist in the world.\n\n");

    brain.discoverConcepts(1.8f, 4);

    for (int ci = 0; ci < (int)brain.concepts.size(); ci++) {
        auto& c = brain.concepts[ci];
        if (c.label.empty())
            printf("  \033[1;36mConcept %d (unnamed)\033[0m\n", ci);
        else
            printf("  \033[1;36mConcept %d → discovered name: \"%s\"\033[0m\n",
                   ci, c.label.c_str());

        printf("    features: ");
        int printed = 0;
        for (int m : c.members) {
            if (brain.idx2feat[m].substr(0,7) == "called:") continue;
            if (printed > 0) printf(", ");
            printf("%s", brain.idx2feat[m].c_str());
            printed++;
        }
        printf("\n\n");
    }

    printf("  The brain discovered \033[1m%d concepts\033[0m from pure experience.\n",
           (int)brain.concepts.size());

    // ── Query the brain ─────────────────────────────────────
    printf("\n══ ASKING THE BRAIN QUESTIONS ═════════════════════════════\n\n");
    printf("  We give it features. It tells us what it knows.\n\n");

    struct Q { const char* question; std::vector<std::string> features; };
    Q questions[] = {
        {"What meows?",                       {"meows"}},
        {"What's furry with four legs?",      {"furry", "four_legs"}},
        {"What swims?",                       {"swims"}},
        {"What's metallic with wheels?",      {"metallic", "four_wheels"}},
        {"What's hot and bright?",            {"hot", "bright"}},
    };

    for (auto& q : questions) {
        printf("  Q: \033[1m%s\033[0m\n", q.question);
        auto results = brain.query(q.features);
        if (results.empty()) {
            printf("    Brain: \"I don't know.\"\n\n");
            continue;
        }
        for (int i = 0; i < std::min(3, (int)results.size()); i++) {
            auto& c = brain.concepts[results[i].conceptIdx];
            printf("    → ");
            if (!c.label.empty())
                printf("\033[1;32m\"%s\"\033[0m ", c.label.c_str());
            else
                printf("\033[33m(unnamed concept)\033[0m ");
            print_bar(results[i].matchScore, 10, "\033[32m");
            printf(" %.0f%% match\n", results[i].matchScore * 100);
        }
        printf("\n");
    }

    // ── Pattern completion ──────────────────────────────────
    printf("══ PATTERN COMPLETION ════════════════════════════════════\n\n");
    printf("  Give partial info → brain predicts what else is true.\n");
    printf("  This is recall from experience, not lookup.\n\n");

    struct PC { const char* label; std::vector<std::string> known; };
    PC completions[] = {
        {"I see something furry and small...",     {"furry", "small"}},
        {"Something in the sky that chirps...",    {"chirps"}},
        {"It's metallic and has an engine...",     {"metallic", "engine"}},
        {"I see something scaly and cold...",      {"scaly", "cold"}},
    };

    for (auto& pc : completions) {
        printf("  \033[1m%s\033[0m\n", pc.label);
        auto preds = brain.complete(pc.known);
        printf("    Brain predicts: ");
        int shown = 0;
        for (auto& [feat, prob] : preds) {
            if (shown >= 8) break;
            if (feat.substr(0,7) == "called:") {
                printf("\033[1;33m%s(%.0f%%)\033[0m ", feat.c_str(), prob*100);
            } else {
                printf("%s(%.0f%%) ", feat.c_str(), prob*100);
            }
            shown++;
        }
        printf("\n\n");
    }

    // ── Novel entity test ───────────────────────────────────
    printf("══ GENERALIZATION — HANDLE THE UNKNOWN ═══════════════════\n\n");
    printf("  Present something the brain has NEVER experienced.\n");
    printf("  It should recognize similarities from what it has learned.\n\n");

    struct Novel { const char* desc; std::vector<std::string> features; };
    Novel novels[] = {
        {"A wolf: furry, large, four_legs, howls, wild, warm, tail",
         {"furry","large","four_legs","warm","tail"}},
        {"A submarine: metallic, large, swims, heavy, engine",
         {"metallic","large","swims","heavy","engine"}},
        {"A candle: hot, bright, small, wax, flickering, glows",
         {"hot","bright","small","flickering","glows"}},
    };

    for (auto& n : novels) {
        printf("  Novel: \033[1m%s\033[0m\n", n.desc);

        auto results = brain.recognize(n.features);
        if (!results.empty()) {
            auto& best = brain.concepts[results[0].conceptIdx];
            printf("    Closest concept: ");
            if (!best.label.empty())
                printf("\033[1;32m\"%s\"\033[0m", best.label.c_str());
            else
                printf("\033[33m(unnamed)\033[0m");
            printf(" (%.0f%% match)\n", results[0].matchScore * 100);
        }

        auto preds = brain.complete(n.features);
        printf("    Brain also guesses: ");
        int shown = 0;
        for (auto& [f, p] : preds) {
            if (shown >= 6) break;
            printf("%s(%.0f%%) ", f.c_str(), p*100);
            shown++;
        }
        printf("\n\n");
    }

    // ── Summary ─────────────────────────────────────────────
    printf("\033[1;35m══════════════════════════════════════════════════════════\033[0m\n");
    printf("  This brain was born knowing nothing.\n");
    printf("  After %d experiences, it discovered %d concepts.\n",
           brain.totalExp, (int)brain.concepts.size());
    printf("  It can recognize, predict, and generalize.\n\n");
    printf("  Everything it knows, it learned by watching the world.\n");
    printf("  No one told it cats meow — it saw cats meow, over and over,\n");
    printf("  and formed that association from the statistics of experience.\n");
    printf("\n  Is that 'knowing'? Is that different from an LLM?\n");
    printf("  ...maybe. The structure is different:\n");
    printf("    LLM: P(meow | the cat) from text statistics\n");
    printf("    This: P(meow | furry,small,whiskers) from world experience\n");
    printf("  Same math. Different grounding.\n");
    printf("\033[1;35m══════════════════════════════════════════════════════════\033[0m\n\n");

    return 0;
}
