/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <cpu_features.hpp>
#include <kernel/gradient_avx2.hpp>
#include <math.hpp>
#include <parallel.hpp>

#include <cstddef>
#include <type_traits>

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace detail {

template<typename T>
void gradientRowScalar(T *grad0, T *grad1, const T *current, const T *previous,
                       const T *next, dim_t width, dim_t input_stride,
                       dim_t grad0_stride, dim_t grad1_stride,
                       T vertical_scale) {
    const T half = scalar<T>(0.5);
    const T one  = scalar<T>(1.0);

    if (width == 1) {
        grad0[0] = one * (current[0] - current[0]);
        grad1[0] = vertical_scale * (next[0] - previous[0]);
        return;
    }

    grad0[0] = one * (current[input_stride] - current[0]);
    grad1[0] = vertical_scale * (next[0] - previous[0]);

    for (dim_t x = 1; x < width - 1; ++x) {
        const dim_t input_offset = x * input_stride;
        grad0[x * grad0_stride] = half * (current[input_offset + input_stride] -
                                          current[input_offset - input_stride]);
        grad1[x * grad1_stride] =
            vertical_scale * (next[input_offset] - previous[input_offset]);
    }

    const dim_t last_input = (width - 1) * input_stride;
    grad0[(width - 1) * grad0_stride] =
        one * (current[last_input] - current[last_input - input_stride]);
    grad1[(width - 1) * grad1_stride] =
        vertical_scale * (next[last_input] - previous[last_input]);
}

}  // namespace detail

template<typename T>
void gradient(Param<T> grad0, Param<T> grad1, CParam<T> in) {
    const af::dim4 dims = in.dims();
    if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0 || dims[3] == 0) {
        return;
    }

    T *d_grad0    = grad0.get();
    T *d_grad1    = grad1.get();
    const T *d_in = in.get();

    const af::dim4 inst = in.strides();
    const af::dim4 g0st = grad0.strides();
    const af::dim4 g1st = grad1.strides();

    const dim_t width  = dims[0];
    const dim_t height = dims[1];
    const dim_t depth  = dims[2];

    const dim_t input_x_stride = inst[0];
    const dim_t input_y_stride = inst[1];
    const dim_t input_z_stride = inst[2];
    const dim_t input_w_stride = inst[3];
    const dim_t grad0_x_stride = g0st[0];
    const dim_t grad0_y_stride = g0st[1];
    const dim_t grad0_z_stride = g0st[2];
    const dim_t grad0_w_stride = g0st[3];
    const dim_t grad1_x_stride = g1st[0];
    const dim_t grad1_y_stride = g1st[1];
    const dim_t grad1_z_stride = g1st[2];
    const dim_t grad1_w_stride = g1st[3];

    const T half = scalar<T>(0.5);
    const T one  = scalar<T>(1.0);

    constexpr size_t target_elements_per_task = 1U << 16U;
    const size_t width_elements               = static_cast<size_t>(width);
    const size_t min_rows_per_task =
        width_elements >= target_elements_per_task
            ? 1
            : (target_elements_per_task + width_elements - 1) / width_elements;
    const size_t row_count = static_cast<size_t>(height) *
                             static_cast<size_t>(depth) *
                             static_cast<size_t>(dims[3]);

    bool use_avx2 = false;
    if constexpr (std::is_same<T, float>::value ||
                  std::is_same<T, double>::value) {
        use_avx2 = input_x_stride == 1 && grad0_x_stride == 1 &&
                   grad1_x_stride == 1 &&
                   arrayfire::cpu::detail::isAVX2Supported() &&
                   detail::isGradientAVX2Compiled();
    }

    parallelForRange(
        row_count, min_rows_per_task, [=](size_t begin, size_t end) {
            size_t linear = begin;
            dim_t y = static_cast<dim_t>(linear % static_cast<size_t>(height));
            linear /= static_cast<size_t>(height);
            dim_t z = static_cast<dim_t>(linear % static_cast<size_t>(depth));
            dim_t w = static_cast<dim_t>(linear / static_cast<size_t>(depth));

            for (size_t row = begin; row < end; ++row) {
                const dim_t input_offset = w * input_w_stride +
                                           z * input_z_stride +
                                           y * input_y_stride;
                const dim_t grad0_offset = w * grad0_w_stride +
                                           z * grad0_z_stride +
                                           y * grad0_y_stride;
                const dim_t grad1_offset = w * grad1_w_stride +
                                           z * grad1_z_stride +
                                           y * grad1_y_stride;

                const T *current  = d_in + input_offset;
                const T *previous = current - (y == 0 ? 0 : input_y_stride);
                const T *next =
                    current + (y == height - 1 ? 0 : input_y_stride);
                T *grad0_row           = d_grad0 + grad0_offset;
                T *grad1_row           = d_grad1 + grad1_offset;
                const T vertical_scale = y == 0 || y == height - 1 ? one : half;

                if constexpr (std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) {
                    if (use_avx2) {
                        detail::gradientRowAVX2(grad0_row, grad1_row, current,
                                                previous, next, width,
                                                vertical_scale);
                    } else {
                        detail::gradientRowScalar(
                            grad0_row, grad1_row, current, previous, next,
                            width, input_x_stride, grad0_x_stride,
                            grad1_x_stride, vertical_scale);
                    }
                } else {
                    detail::gradientRowScalar(grad0_row, grad1_row, current,
                                              previous, next, width,
                                              input_x_stride, grad0_x_stride,
                                              grad1_x_stride, vertical_scale);
                }

                if (++y == height) {
                    y = 0;
                    if (++z == depth) {
                        z = 0;
                        ++w;
                    }
                }
            }
        });
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
