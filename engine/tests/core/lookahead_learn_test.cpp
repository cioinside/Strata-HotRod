// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Strata Contributors
//
// tests/core/lookahead_learn_test.cpp - the learned look-ahead (STRATA_LA_LEARN): the cadence rule (the fork's
// la_cadence.hpp arithmetic), the leader contest (the plain router keeps the lead until the learned correction wins
// it by a Newcombe interval, and takes it back when it wins two recent blocks in a row), the correction's effect on
// the top-k, and the state file (a round trip and the refusal of another model's file).

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "strata/core/lookahead_learn.hpp"

using strata::core::LookaheadLearner;
using namespace strata::core::la_cadence;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::printf("lookahead_learn_test FAIL at line %d: %s\n", __LINE__, #cond); \
        std::exit(1); \
    } \
} while (0)

namespace {

// the top-k of a row of logits (the caller of the learner sorts; the test does the same)
void top_k(const std::vector<float>& logits, int n_expert, int n_tok, int k, std::vector<int32_t>& out) {
    out.assign((size_t) n_tok * (size_t) k, 0);
    for (int t = 0; t < n_tok; ++t) {
        const float* lt = logits.data() + (size_t) t * (size_t) n_expert;
        std::vector<int32_t> order((size_t) n_expert);
        for (int e = 0; e < n_expert; ++e) order[(size_t) e] = e;
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int32_t a, int32_t b) { return lt[a] > lt[b]; });
        for (int j = 0; j < k; ++j) out[(size_t) t * (size_t) k + (size_t) j] = order[(size_t) j];
    }
}

// the cadence rule: the doubling, the cap and the drop (the fork's la_cadence_test.cpp arithmetic)
void test_cadence() {
    CHECK(frozen_period(0, 0) == 4);
    CHECK(frozen_period(1999, 0) == 4);
    CHECK(frozen_period(2000, 0) == 8);
    CHECK(frozen_period(3999, 0) == 8);
    CHECK(frozen_period(4000, 0) == 16);
    CHECK(frozen_period(100000, 0) == 16);
    // the drop test: a recent block well below the long-run rate
    CHECK(recent_drop(900, 1000, 100, 500) == true);
    CHECK(recent_drop(900, 1000, 800, 500) == false);
    CHECK(recent_drop(0, 0, 10, 500) == false);   // no long-run rate yet
    CHECK(recent_drop(900, 1000, 10, 50) == false);   // the recent block is too small
    // a live slot trains every event, a frozen one every 4 -> 8 -> 16
    State st;
    set_live(st, 10);
    CHECK(st.per == 1 && st.due == 11);
    set_frozen(st, 10, 2000);
    CHECK(st.per == 4 && st.due == 14 && st.base == 2000);
    CHECK(trains_at(st.due, 13) == false);
    CHECK(trains_at(st.due, 14) == true);
    after_event(st, 14, true, 4000, 100, 500, 50);
    CHECK(st.per == 8 && st.due == 22);
    after_event(st, 22, true, 6000, 100, 500, 50);
    CHECK(st.per == 16 && st.due == 38);
    after_event(st, 38, true, 8000, 100, 500, 50);
    CHECK(st.per == 16 && st.due == 54);
    // a drop sends the period back to 4 at once
    after_event(st, 54, true, 9000, 100, 500, 1);
    CHECK(st.per == 4 && st.due <= 58);
}

// the leader contest and the training: the plain router keeps the lead until the learned correction wins it, and the
// correction wins it only if it really predicts the routed experts the plain router misses.
void test_leader_and_training() {
    constexpr int kLayers = 2, kEmb = 8, kExpert = 4, kRank = 2, kK = 2, kTok = 4;
    LookaheadLearner lr;
    std::string err;
    CHECK(lr.start(kLayers, kEmb, kExpert, kRank, kK, err));
    // the plain router's logits: nearly flat, so a small correction moves an expert into the top k
    std::vector<float> plain((size_t) kTok * kExpert, 0.0f);
    for (int t = 0; t < kTok; ++t)
        for (int e = 0; e < kExpert; ++e) plain[(size_t) t * kExpert + e] = 0.03f * (float) e;
    std::vector<float> x((size_t) kTok * kEmb, 0.0f);
    for (int t = 0; t < kTok; ++t) x[(size_t) t * kEmb] = 1.0f;
    // the layer routes experts 0 and 1; expert 1 is in the GPU cache, so only expert 0 is worth predicting
    std::vector<int32_t> ids((size_t) kTok * kK, 0);
    // the residency table of the whole model (layer * n_expert + expert), as the dispatch hands it over
    std::vector<int32_t> res((size_t) kLayers * kExpert, -1);
    res[1 * kExpert + 1] = 0;   // expert 1 of layer 1 is resident
    std::vector<int32_t> tp, tl;
    std::vector<float> out((size_t) kTok * kExpert, 0.0f);
    int leads = 0;
    for (int ev = 0; ev < 400; ++ev) {
        const uint64_t stamp = (uint64_t) (ev + 1);   // the window's stamp, like the engine's gen + 1
        for (int t = 0; t < kTok; ++t) ids[(size_t) t * kK] = 0, ids[(size_t) t * kK + 1] = 1;
        lr.correct(1, 1, x.data(), kTok, stamp, plain.data(), out.data());
        top_k(plain, kExpert, kTok, kK, tp);
        top_k(out, kExpert, kTok, kK, tl);
        lr.set_top(1, 1, tp.data(), tl.data(), kTok);
        lr.set_label(1, stamp, ids.data(), kTok, kK, res.data());
        lr.train_pending();
        if (lr.leads(1, 1) && leads == 0) leads = ev;
    }
    // the correction learned the routed expert the plain router misses, so it took the lead
    CHECK(leads > 0);
    CHECK(leads < 400);
    // the plain router's top k does not hold the routed expert; the correction's does
    bool in_plain = false, in_learn = false;
    for (int z = 0; z < kK; ++z) {
        if (tp[z] == 0) in_plain = true;
        if (tl[z] == 0) in_learn = true;
    }
    CHECK(in_plain == false);
    CHECK(in_learn == true);
}

// a label of another window does not train the slot
void test_stamp_mismatch() {
    constexpr int kLayers = 2, kEmb = 8, kExpert = 4, kRank = 2, kK = 2, kTok = 2;
    LookaheadLearner lr;
    std::string err;
    CHECK(lr.start(kLayers, kEmb, kExpert, kRank, kK, err));
    std::vector<float> plain((size_t) kTok * kExpert, 0.0f), x((size_t) kTok * kEmb, 0.0f), out(plain.size(), 0.0f);
    std::vector<int32_t> tp((size_t) kTok * kK, 0), tl((size_t) kTok * kK, 0), ids((size_t) kTok * kK, 0);
    std::vector<int32_t> res((size_t) kLayers * kExpert, -1);
    lr.correct(1, 1, x.data(), kTok, 100, plain.data(), out.data());
    lr.set_top(1, 1, tp.data(), tl.data(), kTok);
    // the label of another window: the slot is not trained on it
    lr.set_label(1, 101, ids.data(), kTok, kK, res.data());
    lr.train_pending();
    CHECK(lr.trained_count() == 0);
    // the label of its own window trains it
    lr.set_label(1, 100, ids.data(), kTok, kK, res.data());
    lr.train_pending();
    CHECK(lr.trained_count() == 1);
}

// the plain router takes the lead back when it wins two recent blocks in a row
void test_lead_back() {
    constexpr int kLayers = 2, kEmb = 8, kExpert = 4, kRank = 2, kK = 2, kTok = 4;
    LookaheadLearner lr;
    std::string err;
    CHECK(lr.start(kLayers, kEmb, kExpert, kRank, kK, err));
    std::vector<float> plain((size_t) kTok * kExpert, 0.0f);
    for (int t = 0; t < kTok; ++t)
        for (int e = 0; e < kExpert; ++e) plain[(size_t) t * kExpert + e] = 0.03f * (float) e;
    std::vector<float> x((size_t) kTok * kEmb, 0.0f);
    for (int t = 0; t < kTok; ++t) x[(size_t) t * kEmb] = 1.0f;
    std::vector<int32_t> ids((size_t) kTok * kK, 0);
    std::vector<int32_t> res((size_t) kLayers * kExpert, -1);
    res[1 * kExpert + 1] = 0;   // expert 1 of layer 1 is resident: only the other routed expert is worth predicting
    std::vector<int32_t> tp, tl;
    std::vector<float> out((size_t) kTok * kExpert, 0.0f);
    bool led = false, back = false;
    for (int ev = 0; ev < 1200 && !back; ++ev) {
        const uint64_t stamp = (uint64_t) (ev + 1);
        for (int t = 0; t < kTok; ++t) ids[(size_t) t * kK] = 0, ids[(size_t) t * kK + 1] = 1;
        if (led) {
            // the routing moves to the expert the plain router holds and the correction does not: from now on the
            // plain router wins the recent blocks
            bool plain_has = false, learn_has = false;
            for (int z = 0; z < kK; ++z) {
                if (tp[z] == 3) plain_has = true;
                if (tl[z] == 3) learn_has = true;
            }
            if (plain_has && !learn_has)
                for (int t = 0; t < kTok; ++t) ids[(size_t) t * kK] = 3, ids[(size_t) t * kK + 1] = 1;
        }
        lr.correct(1, 1, x.data(), kTok, stamp, plain.data(), out.data());
        top_k(plain, kExpert, kTok, kK, tp);
        top_k(out, kExpert, kTok, kK, tl);
        lr.set_top(1, 1, tp.data(), tl.data(), kTok);
        lr.set_label(1, stamp, ids.data(), kTok, kK, res.data());
        lr.train_pending();
        if (!led && lr.leads(1, 1)) led = true;
        if (led && !lr.leads(1, 1)) back = true;
    }
    CHECK(led);      // the correction won the lead
    CHECK(back);     // and the plain router took it back
}

// a frozen slot whose leader is the plain router needs no correction (the cadence back-off)
void test_frozen_plain_slot() {
    constexpr int kLayers = 2, kEmb = 8, kExpert = 4, kRank = 2, kK = 2, kTok = 4;
    LookaheadLearner lr;
    std::string err;
    CHECK(lr.start(kLayers, kEmb, kExpert, kRank, kK, err));
    // the plain router holds the routed experts, so the correction never wins the lead
    std::vector<float> plain((size_t) kTok * kExpert, 0.0f);
    for (int t = 0; t < kTok; ++t)
        for (int e = 0; e < kExpert; ++e) plain[(size_t) t * kExpert + e] = 0.03f * (float) e;
    std::vector<float> x((size_t) kTok * kEmb, 0.0f);
    for (int t = 0; t < kTok; ++t) x[(size_t) t * kEmb] = 1.0f;
    std::vector<int32_t> ids((size_t) kTok * kK, 0);
    std::vector<int32_t> res((size_t) kLayers * kExpert, -1);
    std::vector<int32_t> tp, tl;
    std::vector<float> out((size_t) kTok * kExpert, 0.0f);
    for (int ev = 0; ev < 700; ++ev) {
        const uint64_t stamp = (uint64_t) (ev + 1);
        for (int t = 0; t < kTok; ++t) ids[(size_t) t * kK] = 3, ids[(size_t) t * kK + 1] = 2;
        lr.correct(1, 1, x.data(), kTok, stamp, plain.data(), out.data());
        top_k(plain, kExpert, kTok, kK, tp);
        top_k(out, kExpert, kTok, kK, tl);
        lr.set_top(1, 1, tp.data(), tl.data(), kTok);
        lr.set_label(1, stamp, ids.data(), kTok, kK, res.data());
        lr.train_pending();
        if (ev == 100) CHECK(lr.needs_corr(1, 1));   // not frozen yet
    }
    CHECK(lr.leads(1, 1) == false);        // the plain router kept the lead
    CHECK(lr.needs_corr(1, 1) == false);   // the slot froze: no correction and no work
}

// the state file: a round trip, and a file of another model's geometry is refused
void test_state_file() {
    constexpr int kLayers = 3, kEmb = 8, kExpert = 4, kRank = 2, kK = 2, kTok = 2;
    LookaheadLearner lr;
    std::string err;
    CHECK(lr.start(kLayers, kEmb, kExpert, kRank, kK, err));
    std::vector<float> plain((size_t) kTok * kExpert, 0.0f), x((size_t) kTok * kEmb, 0.0f), out(plain.size(), 0.0f);
    std::vector<int32_t> tp((size_t) kTok * kK, 0), tl((size_t) kTok * kK, 0), ids((size_t) kTok * kK, 0);
    std::vector<int32_t> res((size_t) kLayers * kExpert, -1);   // the table of the whole model
    lr.correct(2, 1, x.data(), kTok, 1, plain.data(), out.data());
    lr.set_top(2, 1, tp.data(), tl.data(), kTok);
    lr.set_label(2, 1, ids.data(), kTok, kK, res.data());
    lr.train_pending();
    const std::string path = "/tmp/lookahead_learn_test.sal";
    std::remove(path.c_str());
    std::string msg;
    CHECK(lr.save(path, msg));
    CHECK(lr.load(path, msg));
    // another geometry: the file is refused, not read as ours
    LookaheadLearner other;
    CHECK(other.start(kLayers + 1, kEmb, kExpert, kRank, kK, err));
    CHECK(other.load(path, msg) == false);
    // the file of the same geometry loads
    LookaheadLearner same;
    CHECK(same.start(kLayers, kEmb, kExpert, kRank, kK, err));
    CHECK(same.load(path, msg) == true);
    std::remove(path.c_str());
}

}  // namespace

int main() {
    test_cadence();
    test_leader_and_training();
    test_stamp_mismatch();
    test_lead_back();
    test_frozen_plain_slot();
    test_state_file();
    std::printf("lookahead_learn_test: ok\n");
    return 0;
}
