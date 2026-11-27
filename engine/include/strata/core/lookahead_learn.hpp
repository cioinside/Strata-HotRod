// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Strata Contributors
//
// include/strata/core/lookahead_learn.hpp - the learned look-ahead (STRATA_LA_LEARN=1, opt-in): our CPU-side
// look-ahead predicts the next layer's experts by applying that layer's router to this layer's activation; the
// learner corrects those logits with a learned low-rank term and learns it from the routing the layer really made.
//
// The port of the fork's la_learn_* (sangharshadhyeta/Strata-HotRod, src/kernels/cuda/native_router.cu): the fork
// computes the correction on the GPU and learns it against the target layer's true router logits; ours runs on the
// look-ahead's own thread (our router is a BF16 GEMV + top-k on the GPU, our predictor a CPU thread) and learns
// against the routing (the ids the dispatch hands over).  What is kept of the fork: the plain router's score plus
// the learned correction, the leader that falls back to the plain router when it does not win (the Newcombe test on
// the hit rates), the state in a file bound to the model, and the cadence back-off (the fork's
// docs/SPEC-adaptive-la.md, change 2): a (layer, distance) whose leader has not changed for a long time trains every
// 4 -> 8 -> 16 events, and one that is not due does no training work at all.
//
// The correction of a (target layer t, distance d): corr(e) = sum_q B[t,d][e][q] * u[q], u = A[t,d] * x_{t-d}
// (A: rank x n_embd, B: n_expert x rank).  The loss: the cross entropy of the corrected logits over the plain
// router's top k (plus the routed experts that are not in it) against the experts the layer really routed.  AdamW
// updates A and B.

#pragma once

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "strata/kernels/f16_bits.hpp"

namespace strata::core {

/// the pid of this process (the state file's temporary name must not be shared by two processes)
inline unsigned long la_own_pid() {
#ifdef _WIN32
    return (unsigned long) _getpid();
#else
    return (unsigned long) getpid();
#endif
}

/// The training cadence (the fork's docs/SPEC-adaptive-la.md, change 2; its include/strata/kernels/la_cadence.hpp,
/// ported as it is): a live (layer, distance) trains every event, a frozen one every 4 -> 8 -> 16 events (the
/// doubling after each 2,000 picks without a change), and a recent hit rate below the long-run one sends the period
/// back to 4 at once.  The events are the scored labels: one per layer the learner saw.
namespace la_cadence {
inline constexpr int kFreezePicks = 2000;   ///< picks without a change before the back-off starts (and its doubling step)
inline constexpr int kBasePeriod = 4;       ///< a frozen leader's events between trainings to start with
inline constexpr int kMaxPeriod = 16;       ///< its cap
inline constexpr int kRecentMin = 100;      ///< the recent block must hold this many picks before the drop test runs
inline constexpr float kZ90 = 1.6448536f;   ///< the 90% two-sided normal quantile

/// the 90% Wilson interval of a proportion p over n picks
inline void wilson(float p, float n, float& lo, float& hi) {
    const float z2 = kZ90 * kZ90, den = 1.0f + z2 / n;
    const float c = (p + z2 / (2.0f * n)) / den;
    const float h = kZ90 * std::sqrt(p * (1.0f - p) / n + z2 / (4.0f * n * n)) / den;
    lo = c - h;
    hi = c + h;
}

/// Newcombe's lower bound for p1 - p2 (90%): above 0 when p1 is above p2
inline float diff_lo(float p1, float n1, float p2, float n2) {
    float l1, u1, l2, u2;
    wilson(p1, n1, l1, u1);
    wilson(p2, n2, l2, u2);
    return (p1 - p2) - std::sqrt((p1 - l1) * (p1 - l1) + (u2 - p2) * (u2 - p2));
}

/// the frozen leader's events between trainings, given the picks since its cadence base
inline int frozen_period(unsigned long long picks, unsigned long long base) {
    const unsigned long long k = picks > base ? (picks - base) / (unsigned long long) kFreezePicks : 0ull;
    if (k == 0) return kBasePeriod;
    return k == 1 ? 2 * kBasePeriod : kMaxPeriod;
}

/// the drop test: the recent block's hit rate is below the long-run rate
inline bool recent_drop(unsigned long long hits_long, unsigned long long picks_long, unsigned long long hits_rec,
                        unsigned long long picks_rec) {
    if (picks_long == 0 || picks_rec < (unsigned long long) kRecentMin) return false;
    const float pl = (float) hits_long / (float) picks_long, pr = (float) hits_rec / (float) picks_rec;
    return diff_lo(pl, (float) picks_long, pr, (float) picks_rec) > 0.0f;
}

/// one (layer, distance)'s cadence state
struct State {
    unsigned long long base = 0;   ///< the picks at the start of the current period step (the freeze, or the last drop)
    int due = 0;                   ///< the event at which the frozen leader trains next
    int per = 1;                   ///< the current period (1 live, 4..16 frozen)
};

/// a live (layer, distance) trains every event
inline void set_live(State& st, int w) {
    st.base = 0;
    st.due = w + 1;
    st.per = 1;
}
/// the (layer, distance) freezes at event w with `picks` picks since its last change
inline void set_frozen(State& st, int w, unsigned long long picks) {
    st.base = picks;
    st.due = w + kBasePeriod;
    st.per = kBasePeriod;
}
/// the cadence step of one event (every event, trained or not: the drop test sees every event's counters)
inline void after_event(State& st, int w, bool trained, unsigned long long picks, unsigned long long hits_long,
                        unsigned long long picks_rec, unsigned long long hits_rec) {
    if (trained) {
        st.per = frozen_period(picks, st.base);
        st.due = w + st.per;
    }
    if (recent_drop(hits_long, picks, hits_rec, picks_rec)) {
        st.base = picks;
        st.per = kBasePeriod;
        if (st.due > w + kBasePeriod) st.due = w + kBasePeriod;
    }
}
/// does the frozen leader train at event w (its state's due event has come)
inline bool trains_at(int due, int w) { return due <= w; }
}  // namespace la_cadence

/// The learned look-ahead: one low-rank correction per (target layer, distance), the leader that says whether the
/// correction or the plain router is used, and the counters the leader contest runs on.
class LookaheadLearner {
public:
    static constexpr int kMaxDist = 4;        ///< the distances the look-ahead may predict (1..4 layers ahead)
    static constexpr int kMaxTok = 8;         ///< the tokens a window's prediction holds
    static constexpr int kMinPicks = 200;     ///< the picks before the leader contest runs (the fork's kLaMinPicks)
    static constexpr int kFreezePicks = 2000; ///< the picks without a change before a slot freezes
    static constexpr int kRecPicks = 500;     ///< the recent block's size (the fork's kLaRecPicks)
    static constexpr float kLr = 1e-3f, kB1 = 0.9f, kB2 = 0.999f, kEps = 1e-8f, kWd = 1e-3f;

    /// the geometry of the model, the rank of the correction and the experts per token
    bool start(int64_t n_layers, int64_t n_embd, int64_t n_expert, int rank, int k, std::string& err,
                    uint64_t model_key = 0) {
        if (n_layers < 1 || n_embd < 1 || n_expert < 2 || rank < 1 || rank > 64 || k < 1 || k > n_expert) {
            err = "look-ahead learner: the geometry, the rank or k is out of range";
            return false;
        }
        n_layers_ = n_layers;
        n_embd_ = n_embd;
        n_expert_ = n_expert;
        rank_ = rank;
        k_ = k;
        model_key_ = model_key;
        const size_t n = (size_t) (n_layers * kMaxDist);
        A_.assign(n * (size_t) rank_ * (size_t) n_embd_, 0.0f);
        B_.assign(n * (size_t) n_expert_ * (size_t) rank_, 0.0f);
        mA_.assign(A_.size(), 0.0f);
        vA_.assign(A_.size(), 0.0f);
        mB_.assign(B_.size(), 0.0f);
        vB_.assign(B_.size(), 0.0f);
        slot_.assign(n, Slot());
        xsrc_.assign((size_t) n_layers * (size_t) kMaxTok * (size_t) n_embd_, 0.0f);
        xsrc_tok_.assign((size_t) n_layers, 0);
        label_.assign((size_t) n_layers, Label());
        for (size_t i = 0; i < n; ++i) reset_slot(i);   // A random, B zero: nothing is learned yet
        on_ = true;
        return true;
    }

    bool on() const { return on_; }
    int64_t n_expert() const { return n_expert_; }
    int rank() const { return rank_; }

    /// The forward: the logits the look-ahead uses.  `plain` is the plain router's logits (n_tok x n_expert) and
    /// `out` the corrected ones; the returned pointer is the one to use (the plain ones until the learner leads).
    /// The prediction (the plain logits, the source activation and the stamp) is kept for the label that follows.
    /// The corrected logits are always computed: the learner scores its own slot against the plain router's, which
    /// is how it can take the lead later (the fork's forward scores every slot too).
    const float* correct(int64_t target, int dist, const float* x, int n_tok, uint64_t stamp, const float* plain,
                          float* out) {
        if (!on_ || dist < 1 || dist > kMaxDist || target < 0 || target >= n_layers_ || n_tok < 1 || n_tok > kMaxTok)
            return plain;
        const size_t sl = slot_of(target, dist);
        Slot& s = slot_[sl];
        s.plain.assign(plain, plain + (size_t) n_tok * (size_t) n_expert_);
        s.n_tok = n_tok;
        s.stamp = stamp;
        s.src = target - dist;
        if (s.src >= 0 && s.src < n_layers_) {
            float* dst = xsrc_.data() + (size_t) s.src * (size_t) kMaxTok * (size_t) n_embd_;
            std::memcpy(dst, x, (size_t) n_tok * (size_t) n_embd_ * sizeof(float));
            xsrc_tok_[(size_t) s.src] = n_tok;
        }
        s.u.assign((size_t) n_tok * (size_t) rank_, 0.0f);
        project(sl, x, n_tok, s.u.data());
        const float* u = s.u.data();
        const size_t nb = (size_t) n_expert_ * (size_t) rank_;
        const float* b = B_.data() + sl * nb;
        for (int t = 0; t < n_tok; ++t) {
            const float* p = plain + (size_t) t * (size_t) n_expert_;
            float* o = out + (size_t) t * (size_t) n_expert_;
            const float* uu = u + (size_t) t * (size_t) rank_;
            for (int64_t e = 0; e < n_expert_; ++e) {
                const float* be = b + (size_t) e * (size_t) rank_;
                float c = 0.0f;
                for (int q = 0; q < rank_; ++q) c += be[q] * uu[q];
                o[e] = p[e] + c;
            }
        }
        return s.lead != 0 ? out : plain;
    }

    /// Does the learned correction lead at (target, dist)?  Then the look-ahead uses the corrected logits.
    bool leads(int64_t target, int dist) const {
        if (!on_ || dist < 1 || dist > kMaxDist || target < 0 || target >= n_layers_) return false;
        return slot_[slot_of(target, dist)].lead != 0;
    }

    /// Does this (target, distance) still need the correction?  A frozen slot whose leader is the plain router can
    /// never change (the contest and the drop test are off for it), so the correction is not built for it: the
    /// cadence back-off of the fork ("no work where there is nothing to learn").
    bool needs_corr(int64_t target, int dist) const {
        if (!on_ || dist < 1 || dist > kMaxDist || target < 0 || target >= n_layers_) return false;
        const Slot& s = slot_[slot_of(target, dist)];
        return !(s.frozen && s.lead == 0);
    }

    /// the number of the training steps (the test watches it)
    unsigned long long trained_count() const { return trained_; }

    /// The top-k of the plain router and of the learned correction (per token), the caller has them: the hit rates
    /// the leader contest runs on.
    void set_top(int64_t target, int dist, const int32_t* top_plain, const int32_t* top_learn, int n_tok) {
        if (!on_ || dist < 1 || dist > kMaxDist || target < 0 || target >= n_layers_) return;
        Slot& s = slot_[slot_of(target, dist)];
        s.top_plain.assign(top_plain, top_plain + (size_t) n_tok * (size_t) k_);
        s.top_learn.assign(top_learn, top_learn + (size_t) n_tok * (size_t) k_);
        s.ok = true;
    }

    /// The label: the target layer's real routing (the dispatch thread).  The ids are the experts the layer routed
    /// (n_tok x k); `pick` marks the ones the GPU cache does not hold - the ones a prediction is worth.
    void set_label(int64_t target, uint64_t stamp, const int32_t* ids, int n_tok, int k, const int32_t* host_res) {
        if (!on_ || target < 0 || target >= n_layers_ || n_tok < 1) return;
        std::lock_guard<std::mutex> lk(mu_);
        if (host_res != nullptr) has_res_ = true;   // the labels carry the cache's residency: the learner scores the misses
        Label& l = label_[(size_t) target];
        l.ids.assign(ids, ids + (size_t) n_tok * (size_t) k);
        l.pick.assign((size_t) n_tok * (size_t) k, 0);
        for (int t = 0; t < n_tok; ++t)
            for (int j = 0; j < k; ++j) {
                const int32_t e = ids[(size_t) t * (size_t) k + (size_t) j];
                l.pick[(size_t) t * (size_t) k + (size_t) j] =
                    (e < 0 || e >= n_expert_ || host_res == nullptr ||
                     host_res[(size_t) (target * n_expert_ + e)] < 0) ? 1 : 0;
            }
        l.n_tok = n_tok;
        l.k = k;
        l.stamp = stamp;
        l.valid = true;
    }

    /// The scoring, the leader contest and the training of every slot whose label arrived (the look-ahead thread).
    void train_pending() {
        if (!on_) return;
        for (int64_t t = 0; t < n_layers_; ++t) {
            for (int d = 1; d <= kMaxDist; ++d) {
                const size_t sl = slot_of(t, d);
                Slot& s = slot_[sl];
                if (!s.ok || s.stamp == 0) continue;
                Label l;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    const Label& src = label_[(size_t) t];
                    if (!src.valid || src.stamp != s.stamp || s.trained_stamp == s.stamp) continue;
                    s.trained_stamp = s.stamp;
                    l = src;
                }
                score_and_train(sl, t, d, s, l);
            }
        }
    }

    /// the stats line: the hit rates of the plain router and of the leader, per distance
    std::string report() const {
        std::string s = "look-ahead learner: ";
        for (int d = 1; d <= kMaxDist; ++d) {
            unsigned long long n = 0, hp = 0, hl = 0;
            for (int64_t l = 0; l < n_layers_; ++l) {
                const Slot& s = slot_[slot_of(l, d)];
                n += s.picks_all;
                hp += s.hits_plain_all;
                hl += s.hits_learn_all;
            }
            if (n == 0) continue;
            char buf[160];
            std::snprintf(buf, sizeof buf, "d=%d: %.3f plain vs %.3f learned (%llu picks); ", d,
                          (double) hp / (double) n, (double) hl / (double) n, (unsigned long long) n);
            s += buf;
        }
        char buf[160];
        std::snprintf(buf, sizeof buf, "%llu events, %llu trained, %llu leader changes, %llu resets", (unsigned long long) events_,
                      (unsigned long long) trained_, (unsigned long long) changes_, (unsigned long long) resets_);
        s += buf;
        // the objective: with the cache's residency the learner scores the misses, without it the whole routing
        s += has_res_ ? "; mode: the misses' recall" : "; mode: the routing recall (no residency table)";
        return s;
    }

    /// the state file: "SAL1", version 1, the geometry, the rank, k; then per slot the leader and its counters; then
    /// per slot the correction (A and B) in fp16.  A file of another geometry is refused.
    bool load(const std::string& path, std::string& msg) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            msg = "no file";
            return false;
        }
        FileHeader hdr{};
        bool ok = std::fread(&hdr, sizeof hdr, 1, f) == 1 && hdr.magic == 0x314C414Cu && hdr.version == 1u &&
                  hdr.model_key == model_key_ && (int64_t) hdr.n_layers == n_layers_ &&
                  (int64_t) hdr.n_embd == n_embd_ && (int64_t) hdr.n_expert == n_expert_ && (int) hdr.rank == rank_ &&
                  (int) hdr.k == k_ && hdr.n_dist == (uint32_t) kMaxDist;
        const size_t n = (size_t) (n_layers_ * kMaxDist);
        const size_t na = (size_t) rank_ * (size_t) n_embd_, nb = (size_t) n_expert_ * (size_t) rank_;
        // the headers go into temporaries: the slots change only after the whole file is read (a truncated file must
        // not leave a half-loaded state behind)
        std::vector<Meta> meta(n);
        if (ok) ok = std::fread(meta.data(), sizeof(Meta), meta.size(), f) == meta.size();
        size_t leads = 0;
        for (size_t i = 0; ok && i < n; ++i)
            if (meta[i].lead != 0) ++leads;
        std::vector<uint16_t> all;
        if (ok) {
            all.assign(leads * (na + nb), 0);
            ok = std::fread(all.data(), sizeof(uint16_t), all.size(), f) == all.size();
        }
        std::fclose(f);
        if (!ok) {
            msg = "not a SAL1 file of this model";
            return false;
        }
        // the file is whole: the slots take the state now
        size_t off = 0;
        for (size_t i = 0; i < n; ++i) {
            Slot& s = slot_[i];
            s.lead = (int) meta[i].lead;
            s.picks = meta[i].picks;
            s.hits_plain = meta[i].hits_plain;
            s.hits_learn = meta[i].hits_learn;
            s.step = (int) meta[i].step;
            if (s.lead == 0) continue;   // the plain router: no correction to load
            for (size_t j = 0; j < na; ++j) A_[i * na + j] = strata::kernels::f32_from_f16(all[off + j]);
            for (size_t j = 0; j < nb; ++j) B_[i * nb + j] = strata::kernels::f32_from_f16(all[off + na + j]);
            off += na + nb;
        }
        msg = "loaded";
        return true;
    }

    bool save(const std::string& path, std::string& msg) {
        const std::string tmp = path + ".tmp" + std::to_string(la_own_pid());
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (f == nullptr) {
            msg = "cannot write " + tmp;
            return false;
        }
        const FileHeader hdr = {0x314C414Cu, 1u, (uint32_t) n_layers_, (uint32_t) n_embd_, (uint32_t) n_expert_,
                                (uint32_t) rank_, (uint32_t) k_, (uint32_t) kMaxDist, model_key_};
        bool ok = std::fwrite(&hdr, sizeof hdr, 1, f) == 1;
        const size_t n = (size_t) (n_layers_ * kMaxDist);
        std::vector<Meta> meta(n);
        for (size_t i = 0; i < n; ++i) {
            const Slot& s = slot_[i];
            meta[i].lead = (uint32_t) s.lead;
            meta[i].step = (uint32_t) s.step;
            meta[i].picks = s.picks;
            meta[i].hits_plain = s.hits_plain;
            meta[i].hits_learn = s.hits_learn;
        }
        ok = ok && std::fwrite(meta.data(), sizeof(Meta), meta.size(), f) == meta.size();
        std::vector<uint16_t> buf;
        const size_t na = (size_t) rank_ * (size_t) n_embd_, nb = (size_t) n_expert_ * (size_t) rank_;
        for (size_t i = 0; ok && i < n; ++i) {
            if (slot_[i].lead == 0) continue;
            buf.assign(na + nb, 0);
            for (size_t j = 0; j < na; ++j) buf[j] = strata::kernels::f16_from_f32(A_[i * na + j]);
            for (size_t j = 0; j < nb; ++j) buf[na + j] = strata::kernels::f16_from_f32(B_[i * nb + j]);
            ok = std::fwrite(buf.data(), sizeof(uint16_t), buf.size(), f) == buf.size();
        }
        const bool closed = std::fclose(f) == 0;
        if (!ok || !closed) {
            std::remove(tmp.c_str());
            msg = "write failed";
            return false;
        }
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            std::remove(tmp.c_str());
            msg = "rename failed";
            return false;
        }
        msg = "saved";
        return true;
    }

private:
    struct Slot {
        int lead = 0;                  ///< 0: the plain router, 1: the learned correction
        int n_tok = 0;
        int64_t src = -1;
        uint64_t stamp = 0;            ///< the window the prediction was made in
        uint64_t trained_stamp = 0;    ///< the window the slot was trained for
        bool ok = false;               ///< the prediction is there (the top-k came with it)
        bool frozen = false;           ///< 2000 picks without a change: the cadence takes over
        std::vector<float> plain;      ///< the plain router's logits (n_tok x n_expert)
        std::vector<float> u;          ///< the projection of the source activation (n_tok x rank)
        std::vector<int32_t> top_plain, top_learn;
        unsigned long long picks = 0, hits_plain = 0, hits_learn = 0;   ///< since the last change
        unsigned long long rpicks = 0, rhits_plain = 0, rhits_learn = 0; ///< the recent block (kRecPicks)
        unsigned long long picks_all = 0, hits_plain_all = 0, hits_learn_all = 0;
        unsigned long long bad = 0;     ///< the recent blocks in a row where the plain router beat the leader
        unsigned long long ev = 0;      ///< the events (the scored labels)
        int step = 0;                  ///< the Adam's step counter
        la_cadence::State cad;
    };
    struct Label {
        bool valid = false;
        uint64_t stamp = 0;
        int n_tok = 0, k = 0;
        std::vector<int32_t> ids, pick;
    };
    /// the state file's header: the magic, the version, the model's key and the geometry
    struct FileHeader {
        uint32_t magic, version, n_layers, n_embd, n_expert, rank, k, n_dist;
        uint64_t model_key;
    };
    /// the state file's per-slot header
    struct Meta {
        uint32_t lead, step;
        uint64_t picks, hits_plain, hits_learn;
    };

    size_t slot_of(int64_t target, int dist) const { return (size_t) (target * kMaxDist + (dist - 1)); }

    /// the projection of the source activation: u = A * x (the look-ahead thread)
    void project(size_t sl, const float* x, int n_tok, float* u) const {
        const size_t na = (size_t) rank_ * (size_t) n_embd_;
        const float* a = A_.data() + sl * na;
        for (int t = 0; t < n_tok; ++t) {
            const float* xt = x + (size_t) t * (size_t) n_embd_;
            for (int q = 0; q < rank_; ++q) {
                const float* row = a + (size_t) q * (size_t) n_embd_;
                float sum = 0.0f;
                for (int64_t h = 0; h < n_embd_; ++h) sum += row[h] * xt[h];
                u[(size_t) t * (size_t) rank_ + (size_t) q] = sum;
            }
        }
    }

    /// the AdamW step (the fork's kLaLr, kLaB1, kLaB2, kLaEps, kLaWd)
    static void adam(float& p, float& m, float& v, float g, int t) {
        m = kB1 * m + (1.0f - kB1) * g;
        v = kB2 * v + (1.0f - kB2) * g * g;
        const float c1 = 1.0f - std::pow(kB1, (float) t), c2 = 1.0f - std::pow(kB2, (float) t);
        p -= kLr * ((m / c1) / (std::sqrt(v / c2) + kEps) + kWd * p);
    }

    /// uniform in [-sqrt(3)e-3, sqrt(3)e-3]: the initial draws of A (the fork's la_rand_pm)
    static float rand_pm(uint64_t& key) {
        key += 0x9E3779B97F4A7C15ull;
        uint64_t z = key;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        z = z ^ (z >> 31);
        const float u = (float) (z >> 40) * (1.0f / 16777216.0f);
        return (2.0f * u - 1.0f) * 1.7320508e-3f;
    }

    /// a reset of the learned slot: A random, B and the moments zero (the fork's la_b_reinit_k)
    void reset_slot(size_t sl) {
        const size_t na = (size_t) rank_ * (size_t) n_embd_, nb = (size_t) n_expert_ * (size_t) rank_;
        // the counter of the resets is in the key: a rollback does not re-use the projector of the slot it
        // replaced (the fork counts the resets too)
        uint64_t key = (uint64_t) sl * 0x100000001b3ull + (uint64_t) resets_ * 0x9E3779B97F4A7C15ull +
                      0x9E3779B97F4A7C15ull;
        for (size_t j = 0; j < na; ++j) A_[sl * na + j] = rand_pm(key);
        std::memset(B_.data() + sl * nb, 0, nb * sizeof(float));
        std::memset(mA_.data() + sl * na, 0, na * sizeof(float));
        std::memset(vA_.data() + sl * na, 0, na * sizeof(float));
        std::memset(mB_.data() + sl * nb, 0, nb * sizeof(float));
        std::memset(vB_.data() + sl * nb, 0, nb * sizeof(float));
    }

    /// the hits, the leader contest, the cadence and the training step of one event (the look-ahead thread)
    void score_and_train(size_t sl, int64_t target, int dist, Slot& s, const Label& l) {
        (void) target;
        (void) dist;
        s.ev++;
        events_++;
        const int k = l.k, nt = std::min(s.n_tok, l.n_tok);
        // 1. the picks: the routed experts the GPU cache did not hold
        for (int t = 0; t < nt; ++t) {
            for (int j = 0; j < k; ++j) {
                if (l.pick[(size_t) t * (size_t) k + (size_t) j] == 0) continue;
                const int32_t e = l.ids[(size_t) t * (size_t) k + (size_t) j];
                s.picks++;
                s.rpicks++;
                s.picks_all++;
                const int32_t* tp = s.top_plain.data() + (size_t) t * (size_t) k_;
                const int32_t* tu = s.top_learn.data() + (size_t) t * (size_t) k_;
                for (int z = 0; z < k_; ++z) {
                    if (tp[z] == e) {
                        s.hits_plain++;
                        s.rhits_plain++;
                        s.hits_plain_all++;
                        break;
                    }
                }
                for (int z = 0; z < k_; ++z) {
                    if (tu[z] == e) {
                        s.hits_learn++;
                        s.rhits_learn++;
                        s.hits_learn_all++;
                        break;
                    }
                }
            }
        }
        // 2. the leader contest (the fork's la_decide_one): the learned correction takes the lead when it beats the
        //    plain router by a Newcombe 90% interval over the picks since the last change
        const bool was_frozen = s.frozen;
        const int w = (int) s.ev;
        bool trained = false;   // does this event train the learned slot
        if (!was_frozen) {
            trained = true;      // a live slot trains every event (the fork trains every live candidate)
        } else if (s.lead != 0) {
            trained = la_cadence::trains_at(s.cad.due, w);
        }
        if (!s.frozen && s.picks >= (unsigned long long) kMinPicks) {
            const float n = (float) s.picks;
            const float pp = (float) s.hits_plain / n, pl = (float) s.hits_learn / n;
            if (la_cadence::diff_lo(pl, n, pp, n) > 0.0f) {
                s.lead = 1;
                s.picks = s.hits_plain = s.hits_learn = 0;
                s.rpicks = s.rhits_plain = s.rhits_learn = 0;
                s.bad = 0;
                changes_++;
            }
        }
        if (!s.frozen && s.picks >= (unsigned long long) kFreezePicks) {   // 2000 picks without a change: the slot freezes
            s.frozen = true;
        }
        if (s.rpicks >= (unsigned long long) kRecPicks) {
            if (s.lead != 0) {
                // the plain router beats the leader over the recent picks: back to it (two blocks in a row, the fork)
                const float n = (float) s.rpicks;
                const float rp = (float) s.rhits_plain / n, rl = (float) s.rhits_learn / n;
                s.bad = la_cadence::diff_lo(rp, n, rl, n) > 0.0f ? s.bad + 1 : 0;
                if (s.bad >= 2) {
                    s.lead = 0;
                    s.frozen = false;
                    s.bad = 0;
                    s.picks = s.hits_plain = s.hits_learn = 0;
                    s.step = 0;
                    resets_++;
                    reset_slot(sl);
                }
            }
            s.rpicks = s.rhits_plain = s.rhits_learn = 0;
        }
        // 3. the cadence: a frozen slot trains every 4 -> 8 -> 16 events, a live one every event, and a frozen slot
        //    whose leader is the plain router does no training work at all
        if (!s.frozen) {
            la_cadence::set_live(s.cad, w);
        } else if (!was_frozen) {
            la_cadence::set_frozen(s.cad, w, s.picks);
            if (s.lead == 0) s.cad.due = INT_MAX;   // a plain leader while frozen: no set trains
        } else if (s.lead != 0) {
            la_cadence::after_event(s.cad, w, trained, s.picks, s.hits_learn, s.rpicks, s.rhits_learn);
        }
        // 4. the training: the learned correction is trained whenever the cadence says so
        if (trained) {
            train_step(sl, s, l, nt);
            trained_++;
        }
    }

    /// the gradient step: the cross entropy of the corrected logits over the plain router's top k (plus the routed
    /// experts that are not in it) against the routed experts
    void train_step(size_t sl, Slot& s, const Label& l, int nt) {
        const size_t na = (size_t) rank_ * (size_t) n_embd_, nb = (size_t) n_expert_ * (size_t) rank_;
        float* a = A_.data() + sl * na;
        float* b = B_.data() + sl * nb;
        float* ma = mA_.data() + sl * na;
        float* va = vA_.data() + sl * na;
        float* mb = mB_.data() + sl * nb;
        float* vb = vB_.data() + sl * nb;
        const float* x = xsrc_.data() + (size_t) s.src * (size_t) kMaxTok * (size_t) n_embd_;
        const int xt = xsrc_tok_[(size_t) s.src];
        if (xt <= 0) return;
        // the projection with the current A: the chain rule of the gradient of A needs u = A x as it is now
        std::vector<float> u((size_t) nt * (size_t) rank_, 0.0f);
        project(sl, x, nt, u.data());
        std::vector<float> gb(nb, 0.0f), ga(na, 0.0f), cand_logit, probs;
        std::vector<int32_t> cand, truth;
        int used = 0;
        for (int t = 0; t < nt; ++t) {
            // the routed experts of the token: the loss's truth
            truth.clear();
            for (int j = 0; j < l.k; ++j) {
                const int32_t e = l.ids[(size_t) t * (size_t) l.k + (size_t) j];
                if (e < 0 || e >= n_expert_) continue;
                truth.push_back(e);
            }
            if (truth.empty()) continue;
            used++;
            // the candidates: the plain router's top k, plus the routed experts that are not in it
            const float* pl = s.plain.data() + (size_t) t * (size_t) n_expert_;
            cand.clear();
            const int32_t* tp = s.top_plain.data() + (size_t) t * (size_t) k_;
            for (int z = 0; z < k_; ++z)
                if (std::find(cand.begin(), cand.end(), tp[z]) == cand.end()) cand.push_back(tp[z]);
            for (const int32_t e : truth)
                if (std::find(cand.begin(), cand.end(), e) == cand.end()) cand.push_back(e);
            // the corrected logits over the candidates
            const float* uu = u.data() + (size_t) t * (size_t) rank_;
            cand_logit.resize(cand.size());
            float mx = -1e30f;
            for (size_t i = 0; i < cand.size(); ++i) {
                const int32_t e = cand[i];
                const float* be = b + (size_t) e * (size_t) rank_;
                float c = 0.0f;
                for (int q = 0; q < rank_; ++q) c += be[q] * uu[q];
                cand_logit[i] = pl[e] + c;
                if (cand_logit[i] > mx) mx = cand_logit[i];
            }
            double sum = 0.0;
            for (size_t i = 0; i < cand.size(); ++i) {
                cand_logit[i] = std::exp(cand_logit[i] - mx);
                sum += cand_logit[i];
            }
            // the gradient of the cross entropy: p - y over the candidates
            probs.assign(cand.size(), 0.0f);
            for (size_t i = 0; i < cand.size(); ++i) probs[i] = (float) (cand_logit[i] / sum);
            const float y = 1.0f / (float) truth.size();
            for (size_t i = 0; i < cand.size(); ++i) {
                const int32_t e = cand[i];
                const bool is_truth = std::find(truth.begin(), truth.end(), e) != truth.end();
                const float g = probs[i] - (is_truth ? y : 0.0f);
                if (g == 0.0f) continue;
                for (int q = 0; q < rank_; ++q) gb[(size_t) e * (size_t) rank_ + (size_t) q] += g * uu[q];
            }
            // the gradient of A: (sum over the candidates of g * B[e][q]) * x[h]
            for (int q = 0; q < rank_; ++q) {
                float sum_gb = 0.0f;
                for (size_t i = 0; i < cand.size(); ++i) {
                    const int32_t e = cand[i];
                    const bool is_truth = std::find(truth.begin(), truth.end(), e) != truth.end();
                    const float g = probs[i] - (is_truth ? y : 0.0f);
                    sum_gb += g * b[(size_t) e * (size_t) rank_ + (size_t) q];
                }
                for (int64_t h = 0; h < n_embd_; ++h)
                    ga[(size_t) q * (size_t) n_embd_ + (size_t) h] += sum_gb * x[(size_t) t * (size_t) n_embd_ + (size_t) h];
            }
        }
        if (used == 0) return;
        const float scale = 1.0f / (float) used;
        for (size_t i = 0; i < gb.size(); ++i) gb[i] *= scale;
        for (size_t i = 0; i < ga.size(); ++i) ga[i] *= scale;
        const int step = ++s.step;
        for (size_t i = 0; i < gb.size(); ++i) adam(b[i], mb[i], vb[i], gb[i], step);
        for (size_t i = 0; i < ga.size(); ++i) adam(a[i], ma[i], va[i], ga[i], step);
    }

    bool on_ = false;
    std::atomic<bool> has_res_{false};   // the labels carried the cache's residency at least once
    int64_t n_layers_ = 0, n_embd_ = 0, n_expert_ = 0;
    int rank_ = 8, k_ = 10;
    uint64_t model_key_ = 0;
    std::vector<float> A_, B_, mA_, vA_, mB_, vB_;
    std::vector<Slot> slot_;
    std::vector<float> xsrc_;
    std::vector<int> xsrc_tok_;
    std::vector<Label> label_;
    std::mutex mu_;
    unsigned long long events_ = 0, trained_ = 0, changes_ = 0, resets_ = 0;
};

}  // namespace strata::core
