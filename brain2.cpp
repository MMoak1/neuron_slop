/*
 * brain2.cpp — Sparse Distributed Memory + Associative Thinking
 *
 * The big ideas:
 *   1. An "idea" is NOT one neuron. It's a sparse cloud of ~400 active
 *      neurons out of 10,000. This is called a Sparse Distributed
 *      Representation (SDR). It's how the neocortex actually works.
 *
 *   2. Similar ideas share neurons. "dog" and "cat" both light up the
 *      "animal" neurons, the "furry" neurons, the "pet" neurons.
 *      Generalization falls out for free — novel inputs naturally
 *      overlap with known concepts.
 *
 *   3. Ideas are connected. Activating "dog" partially activates
 *      "park", "bone", "loyal". This IS associative thinking.
 *      Follow the chain and you get a stream of consciousness.
 *
 *   4. Categories emerge. We never explicitly code "animals" as a
 *      category — but all animal concepts share a neural substrate,
 *      so the category EXISTS in the overlap.
 *
 * Compile: g++ -O2 -std=c++17 -o brain2 brain2.cpp
 * Run:     ./brain2
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <algorithm>
#include <random>
#include <functional>
#include <numeric>

// ── Core Parameters ────────────────────────────────────────
static constexpr int N = 10000;                // total neurons
static constexpr int NEURONS_PER_FEATURE = 40; // neurons per feature
static constexpr int MAX_THINK_STEPS = 12;

static std::mt19937 g_rng(42);

// ── Sparse Distributed Representation ──────────────────────
// An SDR is just a sorted list of which neurons are "on."

using SDR = std::vector<int>;

// overlap: how many neurons in common (raw count)
int sdr_overlap(const SDR& a, const SDR& b) {
    int i = 0, j = 0, count = 0;
    while (i < (int)a.size() && j < (int)b.size()) {
        if      (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else { count++; i++; j++; }
    }
    return count;
}

// overlap as fraction of the smaller SDR
float sdr_similarity(const SDR& a, const SDR& b) {
    int shared = sdr_overlap(a, b);
    int minSz = std::min((int)a.size(), (int)b.size());
    return minSz > 0 ? (float)shared / minSz : 0.0f;
}

// ── Feature → Neuron Mapping ───────────────────────────────
// Each feature (a word like "furry") maps to a fixed random subset
// of neurons. This mapping is created once and reused.

static std::unordered_map<std::string, SDR> g_featureMap;

const SDR& get_feature_neurons(const std::string& feature) {
    auto it = g_featureMap.find(feature);
    if (it != g_featureMap.end()) return it->second;

    // create random neuron subset for this feature
    std::unordered_set<int> used;
    SDR neurons;
    while ((int)neurons.size() < NEURONS_PER_FEATURE) {
        int n = g_rng() % N;
        if (used.insert(n).second)
            neurons.push_back(n);
    }
    std::sort(neurons.begin(), neurons.end());
    g_featureMap[feature] = neurons;
    return g_featureMap[feature];
}

// Build an SDR from a list of features (union of their neurons)
SDR encode(const std::vector<std::string>& features) {
    std::set<int> active;
    for (auto& f : features)
        for (int n : get_feature_neurons(f))
            active.insert(n);
    return SDR(active.begin(), active.end());
}

// ── Concept ────────────────────────────────────────────────
struct Concept {
    std::string name;
    std::string category;
    std::vector<std::string> features;
    SDR sdr;
};

// ── Association ────────────────────────────────────────────
struct Association {
    int from;       // concept index
    int to;         // concept index
    float strength; // how strong the link is
};

// ── The Brain ──────────────────────────────────────────────
struct Brain {
    std::vector<Concept> concepts;
    std::vector<Association> associations;

    // index by name
    std::unordered_map<std::string, int> nameIndex;

    int addConcept(const std::string& name, const std::string& category,
                   const std::vector<std::string>& features) {
        int idx = (int)concepts.size();
        Concept c;
        c.name = name;
        c.category = category;
        c.features = features;
        c.sdr = encode(features);
        concepts.push_back(c);
        nameIndex[name] = idx;
        return idx;
    }

    void associate(const std::string& from, const std::string& to,
                   float strength = 1.0f) {
        auto itF = nameIndex.find(from);
        auto itT = nameIndex.find(to);
        if (itF != nameIndex.end() && itT != nameIndex.end())
            associations.push_back({itF->second, itT->second, strength});
    }

    // ── Recognition: given features, what concept is this? ──
    struct Match {
        int idx;
        float similarity;
    };

    std::vector<Match> recognize(const SDR& input, int topK = 5) {
        std::vector<Match> matches;
        for (int i = 0; i < (int)concepts.size(); i++) {
            float sim = sdr_similarity(input, concepts[i].sdr);
            matches.push_back({i, sim});
        }
        std::sort(matches.begin(), matches.end(),
                  [](auto& a, auto& b) { return a.similarity > b.similarity; });
        if ((int)matches.size() > topK) matches.resize(topK);
        return matches;
    }

    // ── Thinking: follow associations from a concept ────────
    struct Thought {
        int conceptIdx;
        float activation;
        std::string reason;
    };

    std::vector<Thought> think(int startIdx, int steps = MAX_THINK_STEPS) {
        std::vector<Thought> stream;
        std::unordered_set<int> visited;

        int current = startIdx;
        stream.push_back({current, 1.0f, "initial thought"});
        visited.insert(current);

        for (int step = 0; step < steps; step++) {
            // gather all associations from current concept
            std::vector<std::pair<int, float>> candidates;
            for (auto& a : associations) {
                if (a.from == current && visited.find(a.to) == visited.end()) {
                    candidates.push_back({a.to, a.strength});
                }
            }

            // also consider SDR-overlap based associations (emergent!)
            // any concept with high neural overlap gets some activation
            for (int i = 0; i < (int)concepts.size(); i++) {
                if (i == current || visited.count(i)) continue;
                float overlap = sdr_similarity(concepts[current].sdr, concepts[i].sdr);
                if (overlap > 0.15f) {
                    // check if already in candidates
                    bool found = false;
                    for (auto& c : candidates)
                        if (c.first == i) { c.second += overlap * 0.5f; found = true; break; }
                    if (!found)
                        candidates.push_back({i, overlap * 0.5f});
                }
            }

            if (candidates.empty()) break;

            // softmax-ish selection with some randomness
            float total = 0;
            for (auto& c : candidates) total += c.second;

            // pick weighted random
            float roll = (float)(g_rng() % 1000) / 1000.0f * total;
            float cumul = 0;
            int next = candidates[0].first;
            float nextStr = candidates[0].second;
            for (auto& c : candidates) {
                cumul += c.second;
                if (cumul >= roll) { next = c.first; nextStr = c.second; break; }
            }

            // was this an explicit association or an emergent one?
            bool explicit_ = false;
            for (auto& a : associations)
                if (a.from == current && a.to == next) { explicit_ = true; break; }

            std::string reason = explicit_
                ? "association"
                : "neural overlap (emergent)";

            stream.push_back({next, nextStr / total, reason});
            visited.insert(next);
            current = next;
        }

        return stream;
    }
};

// ── Pretty Printing ────────────────────────────────────────

void print_bar(float value, int width = 30, const char* color = "\033[36m") {
    int filled = (int)(value * width);
    printf("%s", color);
    for (int i = 0; i < filled; i++) printf("█");
    printf("\033[90m");
    for (int i = filled; i < width; i++) printf("░");
    printf("\033[0m");
}

void print_sdr_visual(const SDR& sdr, int width = 80) {
    // show which regions of the neuron space are active
    std::vector<int> bins(width, 0);
    for (int n : sdr) bins[n * width / N]++;
    int maxBin = *std::max_element(bins.begin(), bins.end());
    if (maxBin == 0) maxBin = 1;

    printf("  \033[90m│\033[0m");
    for (int i = 0; i < width; i++) {
        float intensity = (float)bins[i] / maxBin;
        if (intensity > 0.7f)      printf("\033[97m█\033[0m");
        else if (intensity > 0.4f) printf("\033[37m▓\033[0m");
        else if (intensity > 0.1f) printf("\033[90m▒\033[0m");
        else                       printf("\033[38;5;236m░\033[0m");
    }
    printf("\033[90m│\033[0m");
    printf(" %d active\n", (int)sdr.size());
}

// ════════════════════════════════════════════════════════════
int main() {
    printf("\033[1;35m");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║   SPARSE DISTRIBUTED MEMORY + ASSOCIATIVE THINKING      ║\n");
    printf("║                                                          ║\n");
    printf("║   10,000 neurons · ideas as sparse activation clouds     ║\n");
    printf("║   recognition · generalization · stream of consciousness ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("\033[0m\n");

    Brain brain;

    // ── Define the world of concepts ────────────────────────
    // Each concept has features. Features -> neurons. Similar features -> shared neurons.
    // This is where generalization comes from.

    // ANIMALS
    brain.addConcept("dog",    "animal", {"alive","animal","mammal","four_legs","furry","domestic","loyal","barks","pet","social","warm_blooded"});
    brain.addConcept("cat",    "animal", {"alive","animal","mammal","four_legs","furry","domestic","independent","purrs","pet","agile","warm_blooded"});
    brain.addConcept("eagle",  "animal", {"alive","animal","bird","wings","feathered","wild","predator","flies","sharp_eyes","warm_blooded"});
    brain.addConcept("fish",   "animal", {"alive","animal","aquatic","fins","scales","cold_blooded","swims","silent","wild"});
    brain.addConcept("snake",  "animal", {"alive","animal","reptile","no_legs","scales","cold_blooded","wild","predator","silent"});
    brain.addConcept("horse",  "animal", {"alive","animal","mammal","four_legs","large","fast","domestic","strong","warm_blooded","herbivore"});
    brain.addConcept("wolf",   "animal", {"alive","animal","mammal","four_legs","furry","wild","predator","howls","social","warm_blooded","pack"});
    brain.addConcept("dolphin","animal", {"alive","animal","mammal","aquatic","intelligent","social","swims","warm_blooded","playful"});

    // PLACES
    brain.addConcept("forest", "place", {"place","natural","trees","green","wild","peaceful","animals","shade","ecosystem"});
    brain.addConcept("ocean",  "place", {"place","natural","water","vast","deep","blue","aquatic","waves","ecosystem","salty"});
    brain.addConcept("city",   "place", {"place","urban","buildings","crowded","noisy","lights","roads","technology","social"});
    brain.addConcept("mountain","place",{"place","natural","high","rocky","cold","vast","scenic","challenging","snow"});
    brain.addConcept("home",   "place", {"place","shelter","warm","safe","domestic","comfortable","family","social"});
    brain.addConcept("park",   "place", {"place","green","trees","peaceful","social","dogs","children","sunshine","urban_nature"});

    // FOOD
    brain.addConcept("apple", "food", {"food","fruit","sweet","healthy","natural","plant","red","crunchy"});
    brain.addConcept("steak", "food", {"food","meat","savory","cooked","protein","animal","rich","warm"});
    brain.addConcept("bread", "food", {"food","grain","baked","warm","staple","carbs","cooked","comforting"});
    brain.addConcept("sushi", "food", {"food","fish","japanese","raw","delicate","aquatic","artful","rice"});

    // ABSTRACT IDEAS
    brain.addConcept("freedom",   "abstract", {"abstract","positive","vast","movement","independence","spirit","wild","open"});
    brain.addConcept("love",      "abstract", {"abstract","positive","warm","connection","family","social","deep","powerful"});
    brain.addConcept("fear",      "abstract", {"abstract","negative","cold","danger","predator","dark","primal","survival"});
    brain.addConcept("curiosity", "abstract", {"abstract","positive","exploration","learning","intelligent","discovery","open","playful"});
    brain.addConcept("loneliness","abstract", {"abstract","negative","cold","silent","isolation","empty","dark"});

    // ACTIVITIES
    brain.addConcept("running",  "activity", {"activity","physical","fast","movement","exercise","legs","outdoor","breathing"});
    brain.addConcept("cooking",  "activity", {"activity","creative","food","warm","home","comforting","skill","social"});
    brain.addConcept("reading",  "activity", {"activity","quiet","learning","imagination","indoor","peaceful","intelligent"});
    brain.addConcept("swimming", "activity", {"activity","physical","aquatic","water","movement","exercise","breathing"});
    brain.addConcept("music",    "activity", {"activity","creative","sound","emotional","social","art","powerful","expression"});

    // ── Define associations (directed links between ideas) ──
    // These are learned temporal correlations: "when I think of X, I often think of Y"

    brain.associate("dog",   "park",      0.9f);
    brain.associate("dog",   "home",      0.7f);
    brain.associate("dog",   "loyal",     0.5f);  // won't match (no concept)
    brain.associate("dog",   "wolf",      0.6f);
    brain.associate("cat",   "home",      0.8f);
    brain.associate("cat",   "curiosity", 0.5f);
    brain.associate("eagle", "freedom",   0.9f);
    brain.associate("eagle", "mountain",  0.7f);
    brain.associate("fish",  "ocean",     0.9f);
    brain.associate("fish",  "sushi",     0.6f);
    brain.associate("wolf",  "forest",    0.9f);
    brain.associate("wolf",  "fear",      0.5f);
    brain.associate("wolf",  "pack",      0.3f);
    brain.associate("horse", "freedom",   0.7f);
    brain.associate("horse", "running",   0.8f);
    brain.associate("dolphin","ocean",    0.9f);
    brain.associate("dolphin","curiosity",0.6f);
    brain.associate("dolphin","playful",  0.3f);

    brain.associate("forest","wolf",      0.6f);
    brain.associate("forest","peace",     0.3f);
    brain.associate("ocean", "fish",      0.7f);
    brain.associate("ocean", "dolphin",   0.8f);
    brain.associate("ocean", "swimming",  0.5f);
    brain.associate("ocean", "freedom",   0.5f);
    brain.associate("city",  "loneliness",0.4f);
    brain.associate("city",  "music",     0.5f);
    brain.associate("home",  "cooking",   0.8f);
    brain.associate("home",  "love",      0.7f);
    brain.associate("home",  "reading",   0.5f);
    brain.associate("park",  "running",   0.7f);
    brain.associate("park",  "dog",       0.6f);

    brain.associate("freedom", "eagle",   0.6f);
    brain.associate("freedom", "running", 0.4f);
    brain.associate("love",    "home",    0.8f);
    brain.associate("love",    "music",   0.5f);
    brain.associate("fear",    "snake",   0.6f);
    brain.associate("fear",    "loneliness",0.4f);
    brain.associate("curiosity","reading",0.7f);
    brain.associate("curiosity","dolphin",0.4f);

    brain.associate("cooking", "steak",   0.6f);
    brain.associate("cooking", "bread",   0.5f);
    brain.associate("cooking", "love",    0.4f);
    brain.associate("running", "park",    0.7f);
    brain.associate("running", "horse",   0.4f);
    brain.associate("swimming","ocean",   0.7f);
    brain.associate("swimming","fish",    0.3f);
    brain.associate("music",   "love",    0.6f);
    brain.associate("reading", "curiosity",0.7f);

    // ── Report ──
    int totalFeatures = (int)g_featureMap.size();
    printf("  Loaded \033[1m%d concepts\033[0m using \033[1m%d features\033[0m ",
           (int)brain.concepts.size(), totalFeatures);
    printf("across \033[1m%d neurons\033[0m\n", N);
    printf("  %d explicit associations defined\n\n", (int)brain.associations.size());

    // show SDR stats
    printf("  SDR statistics:\n");
    int totalActive = 0;
    for (auto& c : brain.concepts) totalActive += c.sdr.size();
    printf("    avg neurons per concept: %.0f / %d (%.1f%% sparsity)\n",
           (float)totalActive / brain.concepts.size(), N,
           100.0f * totalActive / brain.concepts.size() / N);
    printf("    neurons per feature: %d\n\n", NEURONS_PER_FEATURE);

    // ════════════════════════════════════════════════════════
    // DEMO 1: What does an idea look like?
    // ════════════════════════════════════════════════════════
    printf("\033[1;33m══ WHAT DOES AN IDEA LOOK LIKE? ══════════════════════\033[0m\n\n");
    printf("  Each idea is a sparse cloud of active neurons.\n");
    printf("  Here's the neural fingerprint of a few ideas:\n\n");

    const char* showConcepts[] = {"dog", "cat", "wolf", "eagle", "ocean", "freedom"};
    for (auto name : showConcepts) {
        auto& c = brain.concepts[brain.nameIndex[name]];
        printf("  \033[1m%-12s\033[0m ", c.name.c_str());
        print_sdr_visual(c.sdr, 60);

        // show features
        printf("  \033[90m             ");
        for (auto& f : c.features) printf("%s ", f.c_str());
        printf("\033[0m\n\n");
    }

    // ════════════════════════════════════════════════════════
    // DEMO 2: Similar ideas share neurons
    // ════════════════════════════════════════════════════════
    printf("\033[1;33m══ SIMILAR IDEAS SHARE NEURONS ═══════════════════════\033[0m\n\n");
    printf("  This IS generalization. The network doesn't need to be told\n");
    printf("  that dogs and cats are similar — they just share neural substrate.\n\n");

    struct Pair { const char* a; const char* b; };
    Pair pairs[] = {
        {"dog","cat"}, {"dog","wolf"}, {"dog","fish"},
        {"dog","eagle"}, {"cat","curiosity"}, {"ocean","fish"},
        {"ocean","swimming"}, {"forest","wolf"}, {"love","home"},
        {"fear","snake"}, {"freedom","eagle"}, {"dog","freedom"},
    };

    for (auto& p : pairs) {
        auto& ca = brain.concepts[brain.nameIndex[p.a]];
        auto& cb = brain.concepts[brain.nameIndex[p.b]];
        float sim = sdr_similarity(ca.sdr, cb.sdr);
        int shared = sdr_overlap(ca.sdr, cb.sdr);

        printf("  %-12s ↔ %-12s  ", p.a, p.b);
        print_bar(sim, 25, sim > 0.3f ? "\033[32m" : sim > 0.15f ? "\033[33m" : "\033[90m");
        printf("  %.1f%% (%d neurons shared)\n", sim * 100, shared);
    }

    // ════════════════════════════════════════════════════════
    // DEMO 3: Recognition — what is this?
    // ════════════════════════════════════════════════════════
    printf("\n\033[1;33m══ RECOGNITION ═══════════════════════════════════════\033[0m\n\n");
    printf("  Give the brain some features. It tells you what idea matches.\n\n");

    struct Query {
        const char* label;
        std::vector<std::string> features;
    };

    Query queries[] = {
        {"\"furry four-legged pet that barks\"",
         {"furry", "four_legs", "pet", "barks"}},

        {"\"something that flies with feathers\"",
         {"flies", "feathered", "wings"}},

        {"\"cold, dark, dangerous\"",
         {"cold", "dark", "danger", "primal"}},

        {"\"warm social place with family\"",
         {"warm", "social", "family", "shelter"}},

        {"\"physical movement in water\"",
         {"physical", "aquatic", "water", "movement"}},
    };

    for (auto& q : queries) {
        printf("  Input: \033[1m%s\033[0m\n", q.label);
        SDR input = encode(q.features);
        printf("  ");
        print_sdr_visual(input, 50);

        auto matches = brain.recognize(input, 5);
        printf("  Results:\n");
        for (int i = 0; i < (int)matches.size() && matches[i].similarity > 0.01f; i++) {
            auto& c = brain.concepts[matches[i].idx];
            printf("    %s%-20s\033[0m ",
                   i == 0 ? "\033[1;32m" : "\033[90m", c.name.c_str());
            print_bar(matches[i].similarity, 20,
                      i == 0 ? "\033[32m" : "\033[90m");
            printf("  %.1f%%\n", matches[i].similarity * 100);
        }
        printf("\n");
    }

    // ════════════════════════════════════════════════════════
    // DEMO 4: Generalization — handle the unknown
    // ════════════════════════════════════════════════════════
    printf("\033[1;33m══ GENERALIZATION ════════════════════════════════════\033[0m\n\n");
    printf("  Feed in something the brain has NEVER seen.\n");
    printf("  It still finds the closest match — this is generalization.\n\n");

    Query novel[] = {
        {"\"a large wild furry predator that howls\" (≈ wolf, never explicitly trained)",
         {"large", "wild", "furry", "predator", "howls"}},

        {"\"a tiny cold-blooded creature with scales\" (≈ snake/fish hybrid)",
         {"cold_blooded", "scales", "small", "silent"}},

        {"\"a creative warm social expression\" (≈ music? cooking? love?)",
         {"creative", "warm", "social", "expression"}},

        {"\"vast natural cold rocky ecosystem\" (a place we haven't seen)",
         {"vast", "natural", "cold", "rocky", "ecosystem"}},
    };

    for (auto& q : novel) {
        printf("  Novel input: \033[1m%s\033[0m\n", q.label);
        SDR input = encode(q.features);
        auto matches = brain.recognize(input, 4);
        printf("  Brain's best guesses:\n");
        for (int i = 0; i < (int)matches.size() && matches[i].similarity > 0.01f; i++) {
            auto& c = brain.concepts[matches[i].idx];
            printf("    %s%-20s\033[0m (%s) ",
                   i == 0 ? "\033[1;32m" : "\033[90m",
                   c.name.c_str(), c.category.c_str());
            print_bar(matches[i].similarity, 20,
                      i == 0 ? "\033[32m" : "\033[90m");
            printf("  %.1f%%\n", matches[i].similarity * 100);
        }
        printf("\n");
    }

    // ════════════════════════════════════════════════════════
    // DEMO 5: Stream of Consciousness
    // ════════════════════════════════════════════════════════
    printf("\033[1;33m══ STREAM OF CONSCIOUSNESS ═══════════════════════════\033[0m\n\n");
    printf("  Start with an idea. The brain follows associations.\n");
    printf("  Some links are explicit, some EMERGE from neural overlap.\n");
    printf("  This is a toy model of how one thought leads to another.\n\n");

    const char* startPoints[] = {"dog", "ocean", "fear", "cooking"};

    for (auto start : startPoints) {
        int startIdx = brain.nameIndex[start];
        auto stream = brain.think(startIdx, 8);

        printf("  \033[1;36m💭 Starting thought: \"%s\"\033[0m\n", start);
        printf("  ");
        for (int i = 0; i < (int)stream.size(); i++) {
            auto& t = stream[i];
            auto& c = brain.concepts[t.conceptIdx];

            if (i == 0) {
                printf("\033[1m%s\033[0m", c.name.c_str());
            } else {
                bool emergent = (t.reason.find("emergent") != std::string::npos);
                if (emergent)
                    printf(" \033[33m~>\033[0m \033[33m%s\033[0m", c.name.c_str());
                else
                    printf(" \033[32m→\033[0m \033[1m%s\033[0m", c.name.c_str());
            }
        }
        printf("\n");

        // detailed trace
        for (int i = 1; i < (int)stream.size(); i++) {
            auto& t = stream[i];
            auto& c = brain.concepts[t.conceptIdx];
            auto& prev = brain.concepts[stream[i-1].conceptIdx];

            float sim = sdr_similarity(prev.sdr, c.sdr);
            int shared = sdr_overlap(prev.sdr, c.sdr);

            printf("    %d. \033[90m%s → %s\033[0m  (reason: %s",
                   i, prev.name.c_str(), c.name.c_str(), t.reason.c_str());
            if (shared > 0)
                printf(", %d neurons shared", shared);
            printf(")\n");
        }
        printf("\n");
    }

    // ════════════════════════════════════════════════════════
    // DEMO 6: Category Discovery
    // ════════════════════════════════════════════════════════
    printf("\033[1;33m══ EMERGENT CATEGORIES ═══════════════════════════════\033[0m\n\n");
    printf("  We never told the brain what an \"animal\" is.\n");
    printf("  But all animal concepts share neurons — the category\n");
    printf("  EXISTS in the neural overlap. Let's find it.\n\n");

    // compute average intra-category overlap
    std::map<std::string, std::vector<int>> categories;
    for (int i = 0; i < (int)brain.concepts.size(); i++)
        categories[brain.concepts[i].category].push_back(i);

    for (auto& [cat, members] : categories) {
        // average pairwise overlap within category
        float avgIntra = 0;
        int pairs = 0;
        for (int i = 0; i < (int)members.size(); i++)
            for (int j = i+1; j < (int)members.size(); j++) {
                avgIntra += sdr_similarity(brain.concepts[members[i]].sdr,
                                           brain.concepts[members[j]].sdr);
                pairs++;
            }
        if (pairs > 0) avgIntra /= pairs;

        // average overlap with everything NOT in this category
        float avgInter = 0;
        int interPairs = 0;
        for (int i : members)
            for (int j = 0; j < (int)brain.concepts.size(); j++) {
                if (brain.concepts[j].category == cat) continue;
                avgInter += sdr_similarity(brain.concepts[i].sdr,
                                           brain.concepts[j].sdr);
                interPairs++;
            }
        if (interPairs > 0) avgInter /= interPairs;

        printf("  \033[1m%-12s\033[0m (%d concepts)  ", cat.c_str(), (int)members.size());
        printf("intra: ");
        print_bar(avgIntra, 15, "\033[32m");
        printf(" %.1f%%   inter: ", avgIntra * 100);
        print_bar(avgInter, 15, "\033[31m");
        printf(" %.1f%%", avgInter * 100);
        float ratio = (avgInter > 0) ? avgIntra / avgInter : 0;
        printf("   \033[90mratio: %.1fx\033[0m\n", ratio);
    }

    printf("\n  \033[90m(higher intra/inter ratio = stronger emergent category)\033[0m\n");

    // ════════════════════════════════════════════════════════
    // Closing thought
    // ════════════════════════════════════════════════════════
    printf("\n\033[1;35m══════════════════════════════════════════════════════\033[0m\n");
    printf("  What we just built:\n");
    printf("    • Ideas as sparse neuron clouds (SDRs)\n");
    printf("    • Recognition from partial information\n");
    printf("    • Generalization to novel inputs\n");
    printf("    • Associative thought chains\n");
    printf("    • Emergent categories from shared neural substrate\n\n");
    printf("  What's missing for 'real' intelligence:\n");
    printf("    • Learning from experience (we hardcoded everything)\n");
    printf("    • Temporal dynamics (spiking, sequences, prediction)\n");
    printf("    • Hierarchy (features → concepts → abstractions → meta)\n");
    printf("    • Self-modification (the network rewiring itself)\n");
    printf("    • Grounding (connecting to actual sensory data)\n");
    printf("    • Whatever consciousness is\n\n");
    printf("  But the foundation — ideas as distributed patterns,\n");
    printf("  generalization from overlap, thinking as association —\n");
    printf("  that's not wrong. That's actually how brains work.\n");
    printf("\033[1;35m══════════════════════════════════════════════════════\033[0m\n\n");

    return 0;
}
