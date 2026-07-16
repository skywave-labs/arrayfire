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

#include <array>
#include <cstddef>

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace detail {

constexpr size_t sort_target_elements_per_task = 1U << 16U;

inline size_t sortLineCount(const af::dim4 &dims, const unsigned dim) {
    size_t line_count = 1;
    for (unsigned current_dim = 0; current_dim < 4; ++current_dim) {
        if (current_dim != dim) {
            line_count *= static_cast<size_t>(dims[current_dim]);
        }
    }
    return line_count;
}

inline size_t sortMinLinesPerTask(const dim_t line_length) {
    if (line_length <= 0) { return 1; }

    const size_t elements = static_cast<size_t>(line_length);
    return elements >= sort_target_elements_per_task
               ? 1
               : (sort_target_elements_per_task + elements - 1) / elements;
}

class SortLineIndexer {
   public:
    SortLineIndexer(size_t line, const af::dim4 &dims,
                    const af::dim4 &strides, const unsigned sort_dim)
        : dims_(dims), strides_(strides), sort_dim_(sort_dim) {
        for (unsigned dim = 0; dim < 4; ++dim) {
            if (dim == sort_dim_) { continue; }

            const size_t extent = static_cast<size_t>(dims_[dim]);
            coords_[dim]        = static_cast<dim_t>(line % extent);
            line /= extent;
            offset_ += coords_[dim] * strides_[dim];
        }
    }

    dim_t offset() const { return offset_; }

    void next() {
        for (unsigned dim = 0; dim < 4; ++dim) {
            if (dim == sort_dim_) { continue; }

            ++coords_[dim];
            offset_ += strides_[dim];
            if (coords_[dim] < dims_[dim]) { return; }

            offset_ -= coords_[dim] * strides_[dim];
            coords_[dim] = 0;
        }
    }

   private:
    const af::dim4 dims_;
    const af::dim4 strides_;
    const unsigned sort_dim_;
    std::array<dim_t, 4> coords_{{0, 0, 0, 0}};
    dim_t offset_{0};
};

}  // namespace detail
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
