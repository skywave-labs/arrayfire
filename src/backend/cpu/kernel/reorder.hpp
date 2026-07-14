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

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void reorder(Param<T> out, CParam<T> in, const af::dim4 oDims,
             const af::dim4 rdims) {
    T* outPtr      = out.get();
    const T* inPtr = in.get();

    const af::dim4 ist = in.strides();
    const af::dim4 ost = out.strides();

    constexpr size_t min_elements_per_task = 1 << 18;
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
                dim_t ids[4]  = {0};
                ids[rdims[0]] = ox;
                ids[rdims[1]] = oy;
                ids[rdims[2]] = oz;
                ids[rdims[3]] = ow;
                dim_t in_idx  = ids[0] * ist[0] + ids[1] * ist[1] +
                               ids[2] * ist[2] + ids[3] * ist[3];
                dim_t out_idx =
                    ox * ost[0] + oy * ost[1] + oz * ost[2] + ow * ost[3];
                const dim_t in_x_stride = ist[rdims[0]];
                for (size_t element = 0; element < run; ++element) {
                    outPtr[out_idx] = inPtr[in_idx];
                    out_idx += ost[0];
                    in_idx += in_x_stride;
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
