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
#include <parallel.hpp>

#include <algorithm>
#include <cstring>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void tile(Param<T> out, CParam<T> in) {
    T* outPtr      = out.get();
    const T* inPtr = in.get();

    const af::dim4 iDims = in.dims();
    const af::dim4 oDims = out.dims();
    const af::dim4 ist   = in.strides();
    const af::dim4 ost   = out.strides();

    constexpr size_t min_elements_per_task = 1 << 18;
    if (iDims.elements() == 1) {
        const T value = inPtr[0];
        parallelForRange(static_cast<size_t>(oDims.elements()),
                         min_elements_per_task, [=](size_t begin, size_t end) {
                             std::fill(outPtr + begin, outPtr + end, value);
                         });
        return;
    }

    const size_t row_bytes = static_cast<size_t>(iDims[0]) * sizeof(T);
    if (row_bytes >= 64) {
        const size_t x_repeats = static_cast<size_t>(oDims[0] / iDims[0]);
        const size_t row_count = static_cast<size_t>(oDims[1]) *
                                 static_cast<size_t>(oDims[2]) *
                                 static_cast<size_t>(oDims[3]);
        const size_t copy_count          = row_count * x_repeats;
        const size_t min_copies_per_task = std::max<size_t>(
            1, min_elements_per_task / static_cast<size_t>(iDims[0]));
        parallelForRange(
            copy_count, min_copies_per_task, [=](size_t begin, size_t end) {
                for (size_t copy = begin; copy < end; ++copy) {
                    const size_t x_repeat = copy % x_repeats;
                    size_t row            = copy / x_repeats;
                    const dim_t oy        = static_cast<dim_t>(row % oDims[1]);
                    row /= static_cast<size_t>(oDims[1]);
                    const dim_t oz = static_cast<dim_t>(row % oDims[2]);
                    const dim_t ow = static_cast<dim_t>(row / oDims[2]);
                    const dim_t iy = oy % iDims[1];
                    const dim_t iz = oz % iDims[2];
                    const dim_t iw = ow % iDims[3];
                    const dim_t input_offset =
                        iy * ist[1] + iz * ist[2] + iw * ist[3];
                    const dim_t output_offset =
                        oy * ost[1] + oz * ost[2] + ow * ost[3] +
                        static_cast<dim_t>(x_repeat) * iDims[0];
                    std::memcpy(outPtr + output_offset, inPtr + input_offset,
                                row_bytes);
                }
            });
        return;
    }

    parallelForRange(
        static_cast<size_t>(oDims.elements()), min_elements_per_task,
        [=](size_t begin, size_t end) {
            size_t linear = begin;
            dim_t ox      = static_cast<dim_t>(linear % oDims[0]);
            linear /= static_cast<size_t>(oDims[0]);
            dim_t oy = static_cast<dim_t>(linear % oDims[1]);
            linear /= static_cast<size_t>(oDims[1]);
            dim_t oz = static_cast<dim_t>(linear % oDims[2]);
            dim_t ow = static_cast<dim_t>(linear / oDims[2]);

            while (begin < end) {
                const size_t run = std::min<size_t>(
                    end - begin, static_cast<size_t>(oDims[0] - ox));
                const dim_t input_row = (oy % iDims[1]) * ist[1] +
                                        (oz % iDims[2]) * ist[2] +
                                        (ow % iDims[3]) * ist[3];
                dim_t output_idx =
                    ox * ost[0] + oy * ost[1] + oz * ost[2] + ow * ost[3];
                for (size_t element = 0; element < run; ++element) {
                    const dim_t ix =
                        (ox + static_cast<dim_t>(element)) % iDims[0];
                    outPtr[output_idx] = inPtr[input_row + ix];
                    output_idx += ost[0];
                }

                begin += run;
                ox += static_cast<dim_t>(run);
                if (ox == oDims[0]) {
                    ox = 0;
                    if (++oy == oDims[1]) {
                        oy = 0;
                        if (++oz == oDims[2]) {
                            oz = 0;
                            ++ow;
                        }
                    }
                }
            }
        });
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
