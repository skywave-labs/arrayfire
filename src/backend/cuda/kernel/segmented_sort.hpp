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
#include <kernel/segmented_sort_dispatch.hpp>
#include <platform.hpp>
#include <types.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

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
