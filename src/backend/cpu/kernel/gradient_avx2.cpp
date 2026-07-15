/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/gradient_avx2.hpp>

#if (defined(__x86_64__) || defined(__amd64__) || defined(_M_X64) || \
     defined(__i386__) || defined(_M_IX86)) &&                       \
    (defined(__GNUC__) || defined(__clang__) ||                      \
     (defined(_MSC_VER) && defined(AF_CPU_COMPILE_AVX2)))
#define AF_CPU_HAS_AVX2_INTRINSICS
#include <immintrin.h>
#endif

#if defined(AF_CPU_HAS_AVX2_INTRINSICS) && \
    (defined(__GNUC__) || defined(__clang__))
#define AF_CPU_AVX2_TARGET __attribute__((target("avx2")))
#else
#define AF_CPU_AVX2_TARGET
#endif

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace detail {

#if defined(AF_CPU_HAS_AVX2_INTRINSICS)

bool isGradientAVX2Compiled() noexcept { return true; }

AF_CPU_AVX2_TARGET void gradientRowAVX2(float *grad0, float *grad1,
                                        const float *current,
                                        const float *previous,
                                        const float *next, dim_t width,
                                        float vertical_scale) noexcept {
    if (width == 1) {
        grad0[0] = current[0] - current[0];
        grad1[0] = vertical_scale * (next[0] - previous[0]);
        return;
    }

    grad0[0] = current[1] - current[0];
    grad1[0] = vertical_scale * (next[0] - previous[0]);

    const __m256 half    = _mm256_set1_ps(0.5F);
    const __m256 y_scale = _mm256_set1_ps(vertical_scale);
    dim_t x              = 1;
    for (; x + 8 <= width - 1; x += 8) {
        const __m256 right = _mm256_loadu_ps(current + x + 1);
        const __m256 left  = _mm256_loadu_ps(current + x - 1);
        const __m256 upper = _mm256_loadu_ps(next + x);
        const __m256 lower = _mm256_loadu_ps(previous + x);

        _mm256_storeu_ps(grad0 + x,
                         _mm256_mul_ps(half, _mm256_sub_ps(right, left)));
        _mm256_storeu_ps(grad1 + x,
                         _mm256_mul_ps(y_scale, _mm256_sub_ps(upper, lower)));
    }
    for (; x < width - 1; ++x) {
        grad0[x] = 0.5F * (current[x + 1] - current[x - 1]);
        grad1[x] = vertical_scale * (next[x] - previous[x]);
    }

    grad0[width - 1] = current[width - 1] - current[width - 2];
    grad1[width - 1] = vertical_scale * (next[width - 1] - previous[width - 1]);
}

AF_CPU_AVX2_TARGET void gradientRowAVX2(double *grad0, double *grad1,
                                        const double *current,
                                        const double *previous,
                                        const double *next, dim_t width,
                                        double vertical_scale) noexcept {
    if (width == 1) {
        grad0[0] = current[0] - current[0];
        grad1[0] = vertical_scale * (next[0] - previous[0]);
        return;
    }

    grad0[0] = current[1] - current[0];
    grad1[0] = vertical_scale * (next[0] - previous[0]);

    const __m256d half    = _mm256_set1_pd(0.5);
    const __m256d y_scale = _mm256_set1_pd(vertical_scale);
    dim_t x               = 1;
    for (; x + 4 <= width - 1; x += 4) {
        const __m256d right = _mm256_loadu_pd(current + x + 1);
        const __m256d left  = _mm256_loadu_pd(current + x - 1);
        const __m256d upper = _mm256_loadu_pd(next + x);
        const __m256d lower = _mm256_loadu_pd(previous + x);

        _mm256_storeu_pd(grad0 + x,
                         _mm256_mul_pd(half, _mm256_sub_pd(right, left)));
        _mm256_storeu_pd(grad1 + x,
                         _mm256_mul_pd(y_scale, _mm256_sub_pd(upper, lower)));
    }
    for (; x < width - 1; ++x) {
        grad0[x] = 0.5 * (current[x + 1] - current[x - 1]);
        grad1[x] = vertical_scale * (next[x] - previous[x]);
    }

    grad0[width - 1] = current[width - 1] - current[width - 2];
    grad1[width - 1] = vertical_scale * (next[width - 1] - previous[width - 1]);
}

#else

bool isGradientAVX2Compiled() noexcept { return false; }

void gradientRowAVX2(float *, float *, const float *, const float *,
                     const float *, dim_t, float) noexcept {}

void gradientRowAVX2(double *, double *, const double *, const double *,
                     const double *, dim_t, double) noexcept {}

#endif

}  // namespace detail
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire

#undef AF_CPU_AVX2_TARGET
#undef AF_CPU_HAS_AVX2_INTRINSICS
