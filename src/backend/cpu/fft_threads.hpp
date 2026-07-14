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

constexpr size_t FFTW_MIN_THREADED_POINTS = 262144;
constexpr size_t FFTW_MAX_PLAN_THREADS    = 4;

inline size_t selectFFTPlanThreadCount(const size_t logicalPoints,
                                       const size_t configuredThreads) {
    if (logicalPoints < FFTW_MIN_THREADED_POINTS) { return 1; }
    return std::min(std::max<size_t>(1, configuredThreads),
                    FFTW_MAX_PLAN_THREADS);
}

}  // namespace detail
}  // namespace cpu
}  // namespace arrayfire
