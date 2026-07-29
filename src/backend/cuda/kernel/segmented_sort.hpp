/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <debug_cuda.hpp>
#include <platform.hpp>
#include <types.hpp>
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

    // Preserve the existing batched-sort gates until native measurements
    // justify widening the segmented path.
    return hasValues ? segments > 4 && segmentLength < 100000 : segments > 10;
}

inline af::dim4 sortPreorder(unsigned dim) {
    af::dim4 order(0, 1, 2, 3);
    order[0] = dim;
    for (unsigned i = 0; i < dim; ++i) { order[i + 1] = i; }
    return order;
}

inline af::dim4 sortPostorder(unsigned dim) {
    af::dim4 order(0, 1, 2, 3);
    for (unsigned i = 0; i < dim; ++i) { order[i] = i + 1; }
    order[dim] = 0;
    return order;
}

template<typename T>
void segmentedSortKeys(T *out, const T *in, int elements, int segmentLength,
                       bool isAscending);

template<typename T>
void segmentedSortPairs(T *outKeys, uint *outIndices, const T *inKeys,
                        int elements, int segmentLength, bool isAscending);

template<typename T>
__global__ void gatherSegmentedKernel(T *out, const T *in,
                                      const uint *sortedIndices, int elements,
                                      int segmentLength) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= elements) { return; }

    const int segmentStart = (id / segmentLength) * segmentLength;
    out[id]                = in[segmentStart + sortedIndices[id]];
}

template<typename T>
void gatherSegmented(T *out, const T *in, const uint *sortedIndices,
                     int elements, int segmentLength) {
    constexpr int threads = 256;
    const int blocks      = (elements - 1) / threads + 1;
    CUDA_LAUNCH(gatherSegmentedKernel<T>, blocks, threads, out, in,
                sortedIndices, elements, segmentLength);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
