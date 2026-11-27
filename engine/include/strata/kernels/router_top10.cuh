// include/strata/kernels/router_top10.cuh - the 512-expert top-10 pick, ONE copy.
//
// The router's selection is needed in two places: the standalone kernel `native_router_top10_multi` and the
// fused router node (STRATA_ROUTER_FUSE=1: the BF16 GEMV and the selection in ONE launch, so the window's
// router is one node instead of two).  For the fused node to be bit-for-bit the two kernels, the selection must
// be ONE piece of code, not two copies that drift - the same rule `swiglu.cuh` follows for the SwiGLU
// expressions.
//
// ONE WARP PER TOKEN: the warp holds the token's 512 logits (16 per lane), the softmax is over the whole row,
// and the ten picks are ten rounds of the warp-wide max with the smaller expert index breaking a tie.  The
// caller passes the token's row; the lane is the thread's own lane in the warp.
#pragma once

#include <cfloat>
#include <cstdint>

namespace strata::kernels {

__device__ __forceinline__ float router_top10_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float router_top10_warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}

/// `logits_row`: the token's 512 logits.  `ids_row` / `weights_row`: its ten picks and their softmax weights.
/// `lane`: the thread's lane (0..31); the whole warp must call this.
///
/// `FRESH` reads the logits past the L1 (the fused node's tail reads rows another block wrote a moment ago:
/// the fence before the counter says they are visible, the L1 must not answer with an older line).
template <bool FRESH = false>
__device__ __forceinline__ void router_top10_row(const float* logits_row, int32_t* ids_row, float* weights_row,
                                                  int lane) {
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if constexpr (FRESH) values[i] = *reinterpret_cast<const volatile float*>(logits_row + lane + i * 32);
        else values[i] = logits_row[lane + i * 32];
    }
    __syncwarp();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = router_top10_warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / router_top10_warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i)
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids_row[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(router_top10_warp_sum(selected_sum), 6.103515625e-05f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights_row[lane] = selected * inverse_selected_sum;
}

}  // namespace strata::kernels
