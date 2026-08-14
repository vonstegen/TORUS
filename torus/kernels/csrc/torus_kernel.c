/*
 * TORUS ternary GEMM kernels (C reference + SIMD dispatch).
 *
 * Implements the contract documented in `docs/KERNELS.md`:
 *   y[b, r] = sum_k x[b, k] * T[r, k] * s[r, k / group_size]
 *
 * The packed layout matches `torus/quant/packing.py`: 4 codes per
 * byte, the right-side padding is zero-coded and never consumed.
 *
 * Inner loop (per output row r, per batch b):
 *   partials = [0] * n_groups
 *   for each packed byte b at columns [4i, 4i+4):
 *       for each slot s in {0..3} with k = 4i + s:
 *           c = unpack(b, s)
 *           if c ==  0: skip
 *           if c == +1: partials[k / group_size] += x[b, k]; adds++
 *           if c == -1: partials[k / group_size] -= x[b, k]; subs++
 *   y[b, r] = sum_g partials[g] * s[r, g]
 *
 * Note: padding slots (k >= in_features) break out of the inner loop
 * without incrementing any counter.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t adds;
    int64_t subs;
    int64_t skips;
    int64_t n_rows;
    int64_t n_cols;
    int64_t elems_loaded;
} torus_op_count;

static inline int32_t ternary_unpack(uint8_t byte, int slot) {
    uint8_t pair = (byte >> (slot * 2)) & 0x3;
    if (pair == 0x1) return 1;
    if (pair == 0x2) return -1;
    return 0;
}

void ternary_gemm_portable(
    const float* x,
    const uint8_t* packed,
    const float* scales,
    float* y,
    int batch,
    int in_features,
    int in_packed,
    int n_groups,
    int group_size,
    int activate_residual,
    torus_op_count* ops)
{
    int64_t adds = 0, subs = 0, skips = 0;
    const int64_t n_rows = ops->n_rows;

    /* Allocate one partials buffer per call. n_groups is small and
     * we are inside an outer test/spec verifier; a SIMD kernel would
     * keep these in registers. */
    float* partials = (float*)calloc((size_t)n_groups, sizeof(float));
    if (partials == NULL) {
        ops->adds = ops->subs = ops->skips = 0;
        ops->elems_loaded = 0;
        (void)activate_residual;
        return;
    }

    for (int r = 0; r < n_rows; ++r) {
        const uint8_t* row_packed = packed + (int64_t)r * in_packed;
        const float* row_scales = scales + (int64_t)r * n_groups;

        for (int b = 0; b < batch; ++b) {
            const float* x_row = x + (int64_t)b * in_features;
            memset(partials, 0, (size_t)n_groups * sizeof(float));

            for (int p = 0; p < in_packed; ++p) {
                uint8_t byte = row_packed[p];
                int k0 = p * 4;

                for (int slot = 0; slot < 4; ++slot) {
                    int k = k0 + slot;
                    if (k >= in_features) break;
                    int32_t c = ternary_unpack(byte, slot);
                    if (c == 0) { skips++; continue; }
                    float v = x_row[k];
                    int g = k / group_size;
                    if (c == 1) { partials[g] += v; adds++; }
                    else        { partials[g] -= v; subs++; }
                }
            }

            float acc = 0.0f;
            for (int g = 0; g < n_groups; ++g) {
                acc += partials[g] * row_scales[g];
            }
            y[(int64_t)b * n_rows + r] = acc;
        }
    }

    free(partials);

    ops->adds = adds;
    ops->subs = subs;
    ops->skips = skips;
    ops->elems_loaded = (int64_t)n_rows * in_features;
    (void)activate_residual;
}

#if defined(__x86_64__) && defined(__AVX512F__)
void ternary_gemm_avx512(
    const float* x, const uint8_t* packed, const float* scales,
    float* y, int batch, int in_features, int in_packed,
    int n_groups, int group_size, int activate_residual,
    torus_op_count* ops)
{
    /* Forward to the portable reference. Wiring real AVX-512 vector
     * loops for the predicate ladder + scale fusion is a follow-on
     * optimization; the contract is satisfied by the reference. */
    ternary_gemm_portable(
        x, packed, scales, y, batch, in_features, in_packed,
        n_groups, group_size, activate_residual, ops);
}
#endif

#if defined(__x86_64__) && defined(__AVX2__) && !defined(__AVX512F__)
void ternary_gemm_avx2(
    const float* x, const uint8_t* packed, const float* scales,
    float* y, int batch, int in_features, int in_packed,
    int n_groups, int group_size, int activate_residual,
    torus_op_count* ops)
{
    ternary_gemm_portable(
        x, packed, scales, y, batch, in_features, in_packed,
        n_groups, group_size, activate_residual, ops);
}
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
void ternary_gemm_sve(
    const float* x, const uint8_t* packed, const float* scales,
    float* y, int batch, int in_features, int in_packed,
    int n_groups, int group_size, int activate_residual,
    torus_op_count* ops)
{
    ternary_gemm_portable(
        x, packed, scales, y, batch, in_features, in_packed,
        n_groups, group_size, activate_residual, ops);
}
#endif

/* === Public dispatch ===================================================== */

void ternary_gemm(
    const float* x, const uint8_t* packed, const float* scales,
    float* y, int batch, int in_features, int in_packed,
    int n_groups, int group_size, int activate_residual,
    torus_op_count* ops)
{
#if defined(__x86_64__) && defined(__AVX512F__)
    ternary_gemm_avx512(
        x, packed, scales, y, batch, in_features, in_packed,
        n_groups, group_size, activate_residual, ops);
#elif defined(__x86_64__) && defined(__AVX2__)
    ternary_gemm_avx2(
        x, packed, scales, y, batch, in_features, in_packed,
        n_groups, group_size, activate_residual, ops);
#elif defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
    ternary_gemm_sve(
        x, packed, scales, y, batch, in_features, in_packed,
        n_groups, group_size, activate_residual, ops);
#else
    ternary_gemm_portable(
        x, packed, scales, y, batch, in_features, in_packed,
        n_groups, group_size, activate_residual, ops);
#endif
}
