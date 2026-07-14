/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <algorithm>
#include <cstddef>

namespace arrayfire {
namespace cpu {
namespace detail {

constexpr size_t FFTW_POINTS_PER_PLAN_THREAD = 65536;
constexpr size_t FFTW_MAX_PLAN_THREADS       = 4;
constexpr size_t FFTW_LARGE_ONLY_MIN_THREADED_POINTS =
    FFTW_POINTS_PER_PLAN_THREAD * FFTW_MAX_PLAN_THREADS;

enum class FFTPlanThreadPolicy { LARGE_ONLY, GRADUAL };

inline size_t selectFFTPlanThreadCount(const size_t logicalPoints,
                                       const size_t configuredThreads,
                                       const FFTPlanThreadPolicy policy) {
    if (policy == FFTPlanThreadPolicy::LARGE_ONLY &&
        logicalPoints < FFTW_LARGE_ONLY_MIN_THREADED_POINTS) {
        return 1;
    }
    const size_t workThreads =
        std::max<size_t>(1, logicalPoints / FFTW_POINTS_PER_PLAN_THREAD);
    return std::min(
        std::min(std::max<size_t>(1, configuredThreads), FFTW_MAX_PLAN_THREADS),
        workThreads);
}

}  // namespace detail
}  // namespace cpu
}  // namespace arrayfire
