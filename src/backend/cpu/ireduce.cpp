/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <ireduce.hpp>
#include <kernel/ireduce.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <parallel.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace cpu {
namespace {

template<typename T>
struct IndexedPartial {
    T value{};
    double key{0};
    uint index{0};
    bool valid{false};
};

template<af_op_t op, typename T>
void updatePartial(IndexedPartial<T> &partial, const T value,
                   const uint index) {
    const double key = kernel::cabs(value);
    if (std::isnan(key)) { return; }

    if (!partial.valid) {
        partial.value = value;
        partial.key   = key;
        partial.index = index;
        partial.valid = true;
        return;
    }

    kernel::MinMaxOp<op, T>::update(partial.value, partial.key, partial.index,
                                    value, key, index);
}

template<af_op_t op, typename T>
IndexedPartial<T> ireduceRange(CParam<T> in, const size_t begin,
                               const size_t end, const bool is_linear) {
    const af::dim4 dims       = in.dims();
    const af::dim4 strides    = in.strides();
    const T *const input      = in.get();
    IndexedPartial<T> partial = {};

    if (is_linear) {
        for (size_t item = begin; item < end; ++item) {
            updatePartial<op>(partial, input[item], static_cast<uint>(item));
        }
        return partial;
    }

    size_t linear = begin;
    std::array<dim_t, 4> coord;
    for (int dim = 0; dim < 4; ++dim) {
        coord[dim] =
            static_cast<dim_t>(linear % static_cast<size_t>(dims[dim]));
        linear /= static_cast<size_t>(dims[dim]);
    }

    size_t item = begin;
    while (item < end) {
        // Keep the existing ireduce_all assumption that stride 0 is one.
        const dim_t offset = coord[0] + coord[1] * strides[1] +
                             coord[2] * strides[2] +
                             coord[3] * strides[3];
        const size_t run =
            std::min(end - item, static_cast<size_t>(dims[0] - coord[0]));
        for (size_t x = 0; x < run; ++x) {
            updatePartial<op>(partial, input[offset + static_cast<dim_t>(x)],
                              static_cast<uint>(item + x));
        }
        item += run;

        coord[0] += static_cast<dim_t>(run);
        if (coord[0] < dims[0]) { continue; }
        coord[0] = 0;
        for (int dim = 1; dim < 4; ++dim) {
            if (++coord[dim] < dims[dim]) { break; }
            coord[dim] = 0;
        }
    }
    return partial;
}

}  // namespace

template<af_op_t op, typename T>
using ireduce_dim_func =
    std::function<void(Param<T>, Param<uint>, const dim_t, CParam<T>,
                       const dim_t, const int, CParam<uint>)>;

template<af_op_t op, typename T>
void ireduce(Array<T> &out, Array<uint> &loc, const Array<T> &in,
             const int dim) {
    dim4 odims       = in.dims();
    odims[dim]       = 1;
    Array<uint> rlen = createEmptyArray<uint>(af::dim4(0));
    static const ireduce_dim_func<op, T> ireduce_funcs[] = {
        kernel::ireduce_dim<op, T, 1>(), kernel::ireduce_dim<op, T, 2>(),
        kernel::ireduce_dim<op, T, 3>(), kernel::ireduce_dim<op, T, 4>()};

    getQueue().enqueue(ireduce_funcs[in.ndims() - 1], out, loc, 0, in, 0, dim,
                       rlen);
}

template<af_op_t op, typename T>
void rreduce(Array<T> &out, Array<uint> &loc, const Array<T> &in, const int dim,
             const Array<uint> &rlen) {
    dim4 odims = in.dims();
    odims[dim] = 1;

    static const ireduce_dim_func<op, T> ireduce_funcs[] = {
        kernel::ireduce_dim<op, T, 1>(), kernel::ireduce_dim<op, T, 2>(),
        kernel::ireduce_dim<op, T, 3>(), kernel::ireduce_dim<op, T, 4>()};

    getQueue().enqueue(ireduce_funcs[in.ndims() - 1], out, loc, 0, in, 0, dim,
                       rlen);
}

template<af_op_t op, typename T>
T ireduce_all(unsigned *loc, const Array<T> &in) {
    in.eval();
    getQueue().sync();

    const af::dim4 dims       = in.dims();
    const af::dim4 strides    = in.strides();
    const CParam<T> input     = in;
    const T *const inPtr      = in.get();

    constexpr size_t block_elements        = 1 << 16;
    constexpr size_t min_parallel_elements = 1 << 17;
    constexpr size_t max_blocks            = 1024;
    const size_t elements = static_cast<size_t>(dims.elements());

    if (elements >= min_parallel_elements &&
        elements - 1 <= std::numeric_limits<uint>::max() &&
        getParallelThreadCount() > 1) {
        const size_t requested_blocks =
            1 + (elements - 1) / block_elements;
        const size_t block_count = std::min(max_blocks, requested_blocks);
        std::vector<IndexedPartial<T>> partials(block_count);
        const size_t elements_per_block = elements / block_count;
        const size_t remainder          = elements % block_count;

        dim_t contiguous_elements = 1;
        bool is_linear            = true;
        for (int dim = 0; dim < 4; ++dim) {
            if (dims[dim] > 1 && strides[dim] != contiguous_elements) {
                is_linear = false;
                break;
            }
            contiguous_elements *= dims[dim];
        }

        parallelForRange(
            block_count, 1,
            [&](const size_t block_begin, const size_t block_end) {
                for (size_t block = block_begin; block < block_end; ++block) {
                    const size_t begin = block * elements_per_block +
                                         std::min(block, remainder);
                    const size_t end = begin + elements_per_block +
                                       (block < remainder ? 1 : 0);
                    partials[block] =
                        ireduceRange<op, T>(input, begin, end, is_linear);
                }
            });

        // Preserve the serial path's special initialization from logical
        // element zero. In particular, an initial NaN retains index zero and
        // an operation identity until a comparable candidate can replace it.
        kernel::MinMaxOp<op, T> Op(inPtr[0], 0);
        for (const IndexedPartial<T> &partial : partials) {
            if (partial.valid) {
                Op.consider(partial.value, partial.key, partial.index);
            }
        }

        *loc = Op.m_idx;
        return Op.m_val;
    }

    kernel::MinMaxOp<op, T> Op(inPtr[0], 0);
    dim_t idx = 0;

    for (dim_t l = 0; l < dims[3]; l++) {
        dim_t off3 = l * strides[3];

        for (dim_t k = 0; k < dims[2]; k++) {
            dim_t off2 = k * strides[2];

            for (dim_t j = 0; j < dims[1]; j++) {
                dim_t off1 = j * strides[1];

                for (dim_t i = 0; i < dims[0]; i++) {
                    dim_t d_idx = i + off1 + off2 + off3;
                    Op(inPtr[d_idx], idx++);
                }
            }
        }
    }

    *loc = Op.m_idx;
    return Op.m_val;
}

#define INSTANTIATE(ROp, T)                                           \
    template void ireduce<ROp, T>(Array<T> & out, Array<uint> & loc,  \
                                  const Array<T> &in, const int dim); \
    template void rreduce<ROp, T>(Array<T> & out, Array<uint> & loc,  \
                                  const Array<T> &in, const int dim,  \
                                  const Array<uint> &rlen);           \
    template T ireduce_all<ROp, T>(unsigned *loc, const Array<T> &in);

// min
INSTANTIATE(af_min_t, float)
INSTANTIATE(af_min_t, double)
INSTANTIATE(af_min_t, cfloat)
INSTANTIATE(af_min_t, cdouble)
INSTANTIATE(af_min_t, int)
INSTANTIATE(af_min_t, uint)
INSTANTIATE(af_min_t, intl)
INSTANTIATE(af_min_t, uintl)
INSTANTIATE(af_min_t, char)
INSTANTIATE(af_min_t, schar)
INSTANTIATE(af_min_t, uchar)
INSTANTIATE(af_min_t, short)
INSTANTIATE(af_min_t, ushort)
INSTANTIATE(af_min_t, half)

// max
INSTANTIATE(af_max_t, float)
INSTANTIATE(af_max_t, double)
INSTANTIATE(af_max_t, cfloat)
INSTANTIATE(af_max_t, cdouble)
INSTANTIATE(af_max_t, int)
INSTANTIATE(af_max_t, uint)
INSTANTIATE(af_max_t, intl)
INSTANTIATE(af_max_t, uintl)
INSTANTIATE(af_max_t, char)
INSTANTIATE(af_max_t, schar)
INSTANTIATE(af_max_t, uchar)
INSTANTIATE(af_max_t, short)
INSTANTIATE(af_max_t, ushort)
INSTANTIATE(af_max_t, half)

}  // namespace cpu
}  // namespace arrayfire
