/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/dim4.hpp>

#include <limits>

namespace arrayfire {
namespace cuda {
namespace kernel {

inline bool useSegmentedSort(const af::dim4 &dims, unsigned dim,
                             bool hasValues) {
    if (dim >= 4) { return false; }

    const dim_t segmentLength = dims[dim];
    const dim_t elements      = dims.elements();
    if (segmentLength <= 1 || elements <= 0 ||
        elements > std::numeric_limits<int>::max()) {
        return false;
    }

    const dim_t segments = elements / segmentLength;
    if (segments > std::numeric_limits<int>::max()) { return false; }

    // CUB is faster inside this envelope on the CUDA devices measured so far.
    // Outside it, scheduling overhead for many short segments and full radix
    // passes for long segments can exceed the legacy batched-sort cost.
    constexpr dim_t maxSegmentLength = 4096;
    constexpr dim_t maxSegments      = 128;
    if (segmentLength > maxSegmentLength || segments > maxSegments) {
        return false;
    }

    return hasValues ? segments > 4 : segments > 10;
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
