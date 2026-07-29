/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <Param.hpp>
#include <common/Source.hpp>
#include <common/deterministicHash.hpp>
#include <common/kernel_cache.hpp>
#include <common/util.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <jit/Consumer.hpp>
#include <kernel/reduce.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <types.hpp>
#include "config.hpp"

#include <af/dim4.hpp>
#include <af/traits.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace arrayfire {
namespace cuda {
namespace kernel {
namespace jit_reduce_detail {

constexpr dim_t minimum_elements         = 1 << 16;
constexpr size_t maximum_parameter_bytes = 4096 - 256;

// Start with whole-array floating-point sums whose producers can be indexed
// linearly. Other operations, types, dimensional reductions, and expression
// layouts keep the established materialize-then-reduce path until native
// measurements justify broadening this dispatch.
template<typename Ti, typename To, af_op_t op>
struct is_supported
    : std::integral_constant<bool, op == af_add_t &&
                                       std::is_same<Ti, To>::value &&
                                       (std::is_same<Ti, float>::value ||
                                        std::is_same<Ti, double>::value)> {};

template<typename Ti, typename To>
std::string makeSource(const jit::Consumer &consumer,
                       const std::string &function_name,
                       const std::string &final_function_name) {
    std::stringstream source;
    source << "typedef unsigned int uint;\ntypedef " << getFullName<dim_t>()
           << " dim_t;\n"
           << jit::getKernelPreamble() << R"JIT(

template<typename T>
struct Param {
    dim_t dims[4];
    dim_t strides[4];
    T *ptr;
};

using InputT = )JIT"
           << getFullName<Ti>() << R"JIT(;
using OutputT = )JIT"
           << getFullName<To>() << R"JIT(;
using ComputeT = OutputT;

extern "C" __global__ void )JIT"
           << function_name << "(\n"
           << consumer.parameterDeclarations() << R"JIT(
    Param<InputT> materialized, Param<OutputT> partial,
    int dim0, int dim1, int dim2, int dim3, uint blocks_x, uint blocks_y,
    uint repeat, bool change_nan, OutputT nanval) {
    const uint tidx = threadIdx.x;
    const uint tidy = threadIdx.y;
    const uint tid = tidy * blockDim.x + tidx;

    const uint zid = blockIdx.x / blocks_x;
    const uint block_idx_x = blockIdx.x - blocks_x * zid;
    const uint wid = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
    const uint block_idx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - blocks_y * wid;
    const uint yid = block_idx_y * blockDim.y + tidy;
    const uint xid = block_idx_x * blockDim.x * repeat + tidx;

    const bool valid = yid < static_cast<uint>(dim1) &&
                       zid < static_cast<uint>(dim2) &&
                       wid < static_cast<uint>(dim3);
    const unsigned long long endpoint =
        static_cast<unsigned long long>(xid) +
        static_cast<unsigned long long>(repeat) * blockDim.x;
    const unsigned long long limit =
        endpoint < static_cast<unsigned long long>(dim0)
            ? endpoint
            : static_cast<unsigned long long>(dim0);

    ComputeT reduced = static_cast<ComputeT>(0);

    for (unsigned long long id = xid; valid && id < limit;
         id += blockDim.x) {
        const int idx =
            static_cast<int>(id) +
            dim0 * (static_cast<int>(yid) +
                    dim1 * (static_cast<int>(zid) +
                            dim2 * static_cast<int>(wid)));
)JIT" << consumer.offsetsAndOperations()
           << R"JIT(
        InputT produced = static_cast<InputT>(val)JIT"
           << consumer.outputId() << R"JIT();
        materialized.ptr[idx] = produced;
        ComputeT value = static_cast<ComputeT>(produced);
        if (change_nan) {
            value = value == value ? value : static_cast<ComputeT>(nanval);
        }
        reduced = value + reduced;
    }

    __shared__ ComputeT values[)JIT"
           << THREADS_PER_BLOCK << R"JIT(];
    const uint lane_id = tid & 31;
    const uint warp_id = tid / 32;
    ComputeT warp_value = reduced;
    for (uint offset = 16; offset > 0; offset /= 2) {
        warp_value =
            warp_value +
            __shfl_down_sync(0xffffffffu, warp_value, offset);
    }

    if (lane_id == 0) { values[warp_id] = warp_value; }
    __syncthreads();

    if (tid < 32) {
        ComputeT block_value =
            tid < blockDim.x / 32
                ? values[tid]
                : static_cast<ComputeT>(0);
        for (uint offset = 16; offset > 0; offset /= 2) {
            block_value =
                block_value +
                __shfl_down_sync(0xffffffffu, block_value, offset);
        }
        if (tid == 0) { partial.ptr[block_idx_x] = block_value; }
    }
}

extern "C" __global__ void )JIT"
           << final_function_name << R"JIT((
    Param<OutputT> out, Param<OutputT> partial, uint partial_count) {
    const uint tid = threadIdx.x;
    const uint lane_id = tid & 31;
    const uint warp_id = tid / 32;

    ComputeT reduced = static_cast<ComputeT>(0);
    for (uint id = tid; id < partial_count; id += blockDim.x) {
        reduced = static_cast<ComputeT>(partial.ptr[id]) + reduced;
    }

    for (uint offset = 16; offset > 0; offset /= 2) {
        reduced =
            reduced + __shfl_down_sync(0xffffffffu, reduced, offset);
    }

    __shared__ ComputeT values[8];
    if (lane_id == 0) { values[warp_id] = reduced; }
    __syncthreads();

    if (tid < 32) {
        ComputeT block_value =
            tid < blockDim.x / 32 ? values[tid] : static_cast<ComputeT>(0);
        for (uint offset = 16; offset > 0; offset /= 2) {
            block_value =
                block_value +
                __shfl_down_sync(0xffffffffu, block_value, offset);
        }
        if (tid == 0) { out.ptr[0] = static_cast<OutputT>(block_value); }
    }
}
)JIT";
    return source.str();
}

template<typename Ti, typename To, af_op_t op>
typename std::enable_if<!is_supported<Ti, To, op>::value, bool>::type launch(
    Param<To>, const Array<Ti> &, bool, double) {
    return false;
}

template<typename Ti, typename To, af_op_t op>
typename std::enable_if<is_supported<Ti, To, op>::value, bool>::type launch(
    Param<To> out, const Array<Ti> &in, bool change_nan, double nanval) {
    if (in.isReady() || in.elements() < minimum_elements ||
        in.elements() > INT_MAX) {
        return false;
    }

    af::dim4 input_dims(in.elements());
    for (int dim = 0; dim < 4; ++dim) {
        if (input_dims[dim] > INT_MAX) { return false; }
    }

    jit::Consumer consumer(in.getNode(), in.dims());
    constexpr size_t reduction_parameter_bytes =
        sizeof(Param<Ti>) + sizeof(Param<To>) + 128;
    if (!consumer.isLinear() ||
        consumer.outputNode().getType() !=
            static_cast<af::dtype>(af::dtype_traits<Ti>::af_type)) {
        return false;
    }
    const size_t consumer_parameter_bytes = consumer.parameterBytes();
    if (consumer_parameter_bytes + reduction_parameter_bytes >=
        maximum_parameter_bytes) {
        return false;
    }

    int extent0 = static_cast<int>(input_dims[0]);
    int extent1 = static_cast<int>(input_dims[1]);
    int extent2 = static_cast<int>(input_dims[2]);
    int extent3 = static_cast<int>(input_dims[3]);

    uint threads_x       = nextpow2(std::max(32u, static_cast<uint>(extent0)));
    threads_x            = std::min(threads_x, THREADS_PER_BLOCK);
    const uint threads_y = THREADS_PER_BLOCK / threads_x;
    uint blocks_x        = divup(extent0, threads_x * REPEAT);
    uint blocks_y        = divup(extent1, threads_y);
    uint repeat          = divup(extent0, blocks_x * threads_x);

    const uint64_t logical_blocks =
        static_cast<uint64_t>(blocks_x) * blocks_y * extent2 * extent3;
    const uint64_t target_blocks =
        4ULL * getMultiProcessorCount(getActiveDeviceId());
    if (consumer.hasExpensiveOperations() &&
        logical_blocks < target_blocks) {
        return false;
    }
    // Even cheap f64 trees lose to the established path at very small grids.
    constexpr uint64_t minimum_double_blocks = 32;
    if (std::is_same<Ti, double>::value &&
        logical_blocks < minimum_double_blocks) {
        return false;
    }

    dim3 threads(threads_x, threads_y);
    dim3 blocks(blocks_x * extent2, blocks_y * extent3);
    const int max_blocks_y = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z               = divup(blocks.y, max_blocks_y);
    blocks.y               = divup(blocks.y, blocks.z);

    dim_t partial_dims[]    = {static_cast<dim_t>(blocks_x), input_dims[1],
                               input_dims[2], input_dims[3]};
    dim_t partial_strides[] = {
        1, partial_dims[0], partial_dims[0] * partial_dims[1],
        partial_dims[0] * partial_dims[1] * partial_dims[2]};

    Array<Ti> materialized_array = createEmptyArray<Ti>(in.dims());
    Param<Ti> materialized       = materialized_array;

    uptr<To> partial_allocation;
    Param<To> partial = out;
    if (blocks_x > 1) {
        partial_allocation = memAlloc<To>(partial_strides[3] *
                                          static_cast<size_t>(partial_dims[3]));
        partial =
            Param<To>(partial_allocation.get(), partial_dims, partial_strides);
    }

    const std::string function_name = consumer.kernelName(
        "_reduce_all_materialize_" +
        std::to_string(static_cast<int>(op)) + "_" +
        std::to_string(static_cast<int>(af::dtype_traits<To>::af_type)));
    const std::string final_function_name = function_name + "_final";
    CUfunction fused_reduce               = nullptr;
    CUmodule fused_module                 = nullptr;
    const auto cached_module = common::findModule(
        getActiveDeviceId(), deterministicHash(function_name));
    if (cached_module) {
        fused_module = cached_module.get();
        fused_reduce =
            common::getKernel(cached_module, function_name, true).get();
    } else {
        const std::string source =
            makeSource<Ti, To>(consumer, function_name, final_function_name);
        common::saveKernel(function_name, source, ".cu");
        const common::Source kernel_source{source.c_str(), source.size(),
                                           deterministicHash(source)};
        auto kernel =
            common::getKernel(function_name, {{kernel_source}}, {}, {}, true);
        fused_module = kernel.getModuleHandle();
        fused_reduce = kernel.get();
    }

    bool replace_nan = change_nan;
    To replacement   = scalar<To>(nanval);
    std::vector<void *> arguments;
    arguments.reserve(consumer_parameter_bytes / sizeof(void *) + 12);
    consumer.appendArguments(arguments);
    arguments.push_back(&materialized);
    arguments.push_back(&partial);
    arguments.push_back(&extent0);
    arguments.push_back(&extent1);
    arguments.push_back(&extent2);
    arguments.push_back(&extent3);
    arguments.push_back(&blocks_x);
    arguments.push_back(&blocks_y);
    arguments.push_back(&repeat);
    arguments.push_back(&replace_nan);
    arguments.push_back(&replacement);

    CU_CHECK(cuLaunchKernel(fused_reduce, blocks.x, blocks.y, blocks.z,
                            threads.x, threads.y, threads.z, 0,
                            getActiveStream(), arguments.data(), nullptr));
    POST_LAUNCH_CHECK();

    if (blocks_x > 1) {
        CUfunction final_reduce = nullptr;
        CU_CHECK(cuModuleGetFunction(&final_reduce, fused_module,
                                     final_function_name.c_str()));
        uint partial_count      = blocks_x;
        void *final_arguments[] = {&out, &partial, &partial_count};
        CU_CHECK(cuLaunchKernel(final_reduce, 1, 1, 1, THREADS_PER_BLOCK, 1, 1,
                                0, getActiveStream(), final_arguments,
                                nullptr));
        POST_LAUNCH_CHECK();
    }
    const_cast<Array<Ti> &>(in).swap(materialized_array);
    return true;
}

}  // namespace jit_reduce_detail

template<typename Ti, typename To, af_op_t op>
bool jitReduceAll(Param<To> out, const Array<Ti> &in, bool change_nan,
                  double nanval) {
    return jit_reduce_detail::launch<Ti, To, op>(out, in, change_nan, nanval);
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
