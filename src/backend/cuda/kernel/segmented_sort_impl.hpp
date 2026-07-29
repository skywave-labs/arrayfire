/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <kernel/segmented_sort.hpp>

#include <err_cuda.hpp>
#include <memory.hpp>
#include <platform.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <type_traits>

namespace arrayfire {
namespace cuda {
namespace kernel {
namespace {

struct SegmentOffset {
    int segmentLength;

    __host__ __device__ __forceinline__ int operator()(int segment) const {
        return segment * segmentLength;
    }
};

inline auto makeSegmentOffsets(int segmentLength) {
    return thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                           SegmentOffset{segmentLength});
}

template<typename T>
struct RadixKey {
    using type                            = T;
    static constexpr bool requiresPacking = false;
};

template<>
struct RadixKey<float> {
    using type                            = uint;
    static constexpr bool requiresPacking = true;
};

template<>
struct RadixKey<double> {
    using type                            = uintl;
    static constexpr bool requiresPacking = true;
};

template<typename T>
struct ToRadixKey;

template<>
struct ToRadixKey<float> {
    __device__ __forceinline__ uint operator()(float value) const {
        constexpr uint sign = uint{1} << 31;
        uint bits           = __float_as_uint(value);

        // C++ comparison considers both zero encodings equivalent. Give them
        // one radix key so the stable segmented sort preserves their order.
        if ((bits & ~sign) == 0) { bits = 0; }

        return (bits & sign) ? ~bits : (bits ^ sign);
    }
};

template<>
struct ToRadixKey<double> {
    __device__ __forceinline__ uintl operator()(double value) const {
        constexpr uintl sign = uintl{1} << 63;
        uintl bits           = static_cast<uintl>(__double_as_longlong(value));

        if ((bits & ~sign) == 0) { bits = 0; }

        return (bits & sign) ? ~bits : (bits ^ sign);
    }
};

__global__ void initSegmentIndices(uint *indices, int elements,
                                   int segmentLength) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < elements) { indices[id] = static_cast<uint>(id % segmentLength); }
}

template<typename T>
__global__ void prepareRadixPairs(typename RadixKey<T>::type *keys,
                                  uint *indices, const T *input, int elements,
                                  int segmentLength) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < elements) {
        keys[id]    = ToRadixKey<T>()(input[id]);
        indices[id] = static_cast<uint>(id % segmentLength);
    }
}

template<typename T>
__global__ void prepareRadixKeys(typename RadixKey<T>::type *keys,
                                 const T *input, int elements) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < elements) { keys[id] = ToRadixKey<T>()(input[id]); }
}

inline void launchSegmentIndices(uint *indices, int elements,
                                 int segmentLength) {
    constexpr int threads = 256;
    const int blocks      = (elements - 1) / threads + 1;
    CUDA_LAUNCH(initSegmentIndices, blocks, threads, indices, elements,
                segmentLength);
}

template<typename T>
void launchRadixKeys(typename RadixKey<T>::type *keys, const T *input,
                     int elements) {
    constexpr int threads = 256;
    const int blocks      = (elements - 1) / threads + 1;
    CUDA_LAUNCH(prepareRadixKeys<T>, blocks, threads, keys, input, elements);
}

template<typename T>
void launchRadixPairs(typename RadixKey<T>::type *keys, uint *indices,
                      const T *input, int elements, int segmentLength) {
    constexpr int threads = 256;
    const int blocks      = (elements - 1) / threads + 1;
    CUDA_LAUNCH(prepareRadixPairs<T>, blocks, threads, keys, indices, input,
                elements, segmentLength);
}

template<typename KeyT>
void cubSortKeys(KeyT *out, const KeyT *in, int elements, int segmentLength,
                 bool isAscending) {
    const int segments  = elements / segmentLength;
    auto offsets        = makeSegmentOffsets(segmentLength);
    size_t tempBytes    = 0;
    cudaStream_t stream = getActiveStream();

    if (isAscending) {
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, tempBytes, in, out, elements, segments, offsets,
            offsets + 1, 0, sizeof(KeyT) * 8, stream));
    } else {
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeysDescending(
            nullptr, tempBytes, in, out, elements, segments, offsets,
            offsets + 1, 0, sizeof(KeyT) * 8, stream));
    }

    // The memory manager may reclaim this allocation as soon as the call
    // returns. Reuse remains safe because every CUDA backend operation for a
    // device is ordered on the same stream.
    auto temp = memAlloc<char>(tempBytes);
    if (isAscending) {
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
            temp.get(), tempBytes, in, out, elements, segments, offsets,
            offsets + 1, 0, sizeof(KeyT) * 8, stream));
    } else {
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeysDescending(
            temp.get(), tempBytes, in, out, elements, segments, offsets,
            offsets + 1, 0, sizeof(KeyT) * 8, stream));
    }
}

template<typename KeyT, typename ValueT>
void cubSortPairs(KeyT *outKeys, ValueT *outValues, const KeyT *inKeys,
                  const ValueT *inValues, int elements, int segmentLength,
                  bool isAscending) {
    const int segments  = elements / segmentLength;
    auto offsets        = makeSegmentOffsets(segmentLength);
    size_t tempBytes    = 0;
    cudaStream_t stream = getActiveStream();

    if (isAscending) {
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, tempBytes, inKeys, outKeys, inValues, outValues, elements,
            segments, offsets, offsets + 1, 0, sizeof(KeyT) * 8, stream));
    } else {
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
            nullptr, tempBytes, inKeys, outKeys, inValues, outValues, elements,
            segments, offsets, offsets + 1, 0, sizeof(KeyT) * 8, stream));
    }

    auto temp = memAlloc<char>(tempBytes);
    if (isAscending) {
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
            temp.get(), tempBytes, inKeys, outKeys, inValues, outValues,
            elements, segments, offsets, offsets + 1, 0, sizeof(KeyT) * 8,
            stream));
    } else {
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp.get(), tempBytes, inKeys, outKeys, inValues, outValues,
            elements, segments, offsets, offsets + 1, 0, sizeof(KeyT) * 8,
            stream));
    }
}

template<typename T>
void segmentedSortKeysImpl(T *out, const T *in, int elements, int segmentLength,
                           bool isAscending, std::false_type) {
    cubSortKeys(out, in, elements, segmentLength, isAscending);
}

template<typename T>
void segmentedSortKeysImpl(T *out, const T *in, int elements, int segmentLength,
                           bool isAscending, std::true_type) {
    using KeyT      = typename RadixKey<T>::type;
    auto inputKeys  = memAlloc<KeyT>(elements);
    auto outputKeys = memAlloc<KeyT>(elements);

    launchRadixKeys<T>(inputKeys.get(), in, elements);
    cubSortPairs(outputKeys.get(), out, inputKeys.get(), in, elements,
                 segmentLength, isAscending);
}

template<typename T>
void segmentedSortPairsImpl(T *outKeys, uint *outIndices, const T *inKeys,
                            int elements, int segmentLength, bool isAscending,
                            std::false_type) {
    auto inputIndices = memAlloc<uint>(elements);
    launchSegmentIndices(inputIndices.get(), elements, segmentLength);
    cubSortPairs(outKeys, outIndices, inKeys, inputIndices.get(), elements,
                 segmentLength, isAscending);
}

template<typename T>
void segmentedSortPairsImpl(T *outKeys, uint *outIndices, const T *inKeys,
                            int elements, int segmentLength, bool isAscending,
                            std::true_type) {
    using KeyT = typename RadixKey<T>::type;
    static_assert(sizeof(KeyT) == sizeof(T) && alignof(KeyT) == alignof(T),
                  "Packed radix keys must match the source representation");

    auto inputKeys    = memAlloc<KeyT>(elements);
    auto inputIndices = memAlloc<uint>(elements);

    launchRadixPairs<T>(inputKeys.get(), inputIndices.get(), inKeys, elements,
                        segmentLength);
    cubSortPairs(reinterpret_cast<KeyT *>(outKeys), outIndices, inputKeys.get(),
                 inputIndices.get(), elements, segmentLength, isAscending);
    gatherSegmented(outKeys, inKeys, outIndices, elements, segmentLength);
}

}  // namespace

template<typename T>
void segmentedSortKeys(T *out, const T *in, int elements, int segmentLength,
                       bool isAscending) {
    using RequiresPacking =
        std::integral_constant<bool, RadixKey<T>::requiresPacking>;
    segmentedSortKeysImpl(out, in, elements, segmentLength, isAscending,
                          RequiresPacking());
    POST_LAUNCH_CHECK();
}

template<typename T>
void segmentedSortPairs(T *outKeys, uint *outIndices, const T *inKeys,
                        int elements, int segmentLength, bool isAscending) {
    using RequiresPacking =
        std::integral_constant<bool, RadixKey<T>::requiresPacking>;
    segmentedSortPairsImpl(outKeys, outIndices, inKeys, elements, segmentLength,
                           isAscending, RequiresPacking());
    POST_LAUNCH_CHECK();
}

#define INSTANTIATE_SEGMENTED_SORT(T)                                   \
    template void segmentedSortKeys<T>(T *, const T *, int, int, bool); \
    template void segmentedSortPairs<T>(T *, uint *, const T *, int, int, bool);

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
