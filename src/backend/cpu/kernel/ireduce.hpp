/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <common/Binary.hpp>
#include <common/half.hpp>
#include <parallel.hpp>
#include <algorithm>
#include <cmath>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
double cabs(const T in) {
    return (double)in;
}
static double cabs(const char in) { return (double)(in > 0); }
static double cabs(const cfloat &in) { return (double)abs(in); }
static double cabs(const cdouble &in) { return (double)abs(in); }

template<af_op_t op, typename T>
struct MinMaxOp {
    T m_val;
    double m_key;
    uint m_idx;
    MinMaxOp(T val, uint idx) : m_val(val), m_key(0), m_idx(idx) {
        using arrayfire::cpu::is_nan;
        if (is_nan(val)) { m_val = common::Binary<T, op>::init(); }
        m_key = cabs(m_val);
    }

    static void update(T &current_val, double &current_key,
                       uint &current_idx, T val, const double key,
                       const uint idx) {
        if (key < current_key ||
            (key == current_key && idx > current_idx)) {
            current_val = val;
            current_key = key;
            current_idx = idx;
        }
    }

    void consider(T val, const double key, const uint idx) {
        update(m_val, m_key, m_idx, val, key, idx);
    }

    void operator()(T val, uint idx) { consider(val, cabs(val), idx); }
};

template<typename T>
struct MinMaxOp<af_max_t, T> {
    T m_val;
    double m_key;
    uint m_idx;
    MinMaxOp(T val, uint idx) : m_val(val), m_key(0), m_idx(idx) {
        using arrayfire::cpu::is_nan;
        if (is_nan(val)) { m_val = common::Binary<T, af_max_t>::init(); }
        m_key = cabs(m_val);
    }

    static void update(T &current_val, double &current_key,
                       uint &current_idx, T val, const double key,
                       const uint idx) {
        if (key > current_key ||
            (key == current_key && idx <= current_idx)) {
            current_val = val;
            current_key = key;
            current_idx = idx;
        }
    }

    void consider(T val, const double key, const uint idx) {
        update(m_val, m_key, m_idx, val, key, idx);
    }

    void operator()(T val, uint idx) { consider(val, cabs(val), idx); }
};

template<af_op_t op, typename T, int D>
struct ireduce_dim {
    void operator()(Param<T> output, Param<uint> locParam,
                    const dim_t outOffset, CParam<T> input,
                    const dim_t inOffset, const int dim, CParam<uint> rlen) {
        const af::dim4 odims    = output.dims();
        const af::dim4 ostrides = output.strides();
        const af::dim4 istrides = input.strides();
        const int D1            = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            ireduce_dim<op, T, D1>()(output, locParam,
                                     outOffset + i * ostrides[D1], input,
                                     inOffset + i * istrides[D1], dim, rlen);
        }
    }
};

template<af_op_t op, typename T>
struct ireduce_dim<op, T, 0> {
    void operator()(Param<T> output, Param<uint> locParam,
                    const dim_t outOffset, CParam<T> input,
                    const dim_t inOffset, const int dim, CParam<uint> rlen) {
        const af::dim4 idims    = input.dims();
        const af::dim4 istrides = input.strides();

        T const *const in   = input.get();
        T *out              = output.get();
        uint *loc           = locParam.get();
        const uint *rlenptr = (rlen.get()) ? rlen.get() + outOffset : nullptr;

        dim_t stride = istrides[dim];
        MinMaxOp<op, T> Op(in[inOffset], 0);
        int lim =
            (rlenptr) ? std::min(idims[dim], (dim_t)*rlenptr) : idims[dim];
        for (dim_t i = 0; i < lim; i++) { Op(in[inOffset + i * stride], i); }

        out[outOffset] = Op.m_val;
        loc[outOffset] = Op.m_idx;
    }
};

template<af_op_t op, typename T>
void ireduce_dim_parallel(Param<T> output, Param<uint> locParam,
                          CParam<T> input, const int dim,
                          CParam<uint> rlen) {
    const af::dim4 odims    = output.dims();
    const af::dim4 ostrides = output.strides();
    const af::dim4 idims    = input.dims();
    const af::dim4 istrides = input.strides();
    const size_t reduced_elements =
        std::max<size_t>(1, static_cast<size_t>(idims[dim]));

    constexpr size_t min_input_elements_per_task = 1 << 16;
    const size_t min_lines_per_task =
        std::max<size_t>(1, min_input_elements_per_task / reduced_elements);
    // Keep each line's scalar fold intact so value/index tie rules are exact.
    parallelForEach(
        odims, min_lines_per_task, [=](dim_t x, dim_t y, dim_t z, dim_t w) {
            const dim_t out_offset = x * ostrides[0] + y * ostrides[1] +
                                     z * ostrides[2] + w * ostrides[3];
            const dim_t in_offset = x * istrides[0] + y * istrides[1] +
                                    z * istrides[2] + w * istrides[3];
            ireduce_dim<op, T, 0> reduce_line;
            reduce_line(output, locParam, out_offset, input, in_offset, dim,
                        rlen);
        });
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
