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
#include <kernel/sort_utils.hpp>
#include <parallel.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <vector>

namespace arrayfire {
namespace cpu {
namespace kernel {

namespace detail {

template<typename Iterator>
void sortRange(Iterator begin, Iterator end, const bool isAscending,
               const bool isStable) {
    using Value = typename std::iterator_traits<Iterator>::value_type;
    if (isAscending) {
        if (isStable) {
            std::stable_sort(begin, end, std::less<Value>());
        } else {
            std::sort(begin, end, std::less<Value>());
        }
    } else {
        if (isStable) {
            std::stable_sort(begin, end, std::greater<Value>());
        } else {
            std::sort(begin, end, std::greater<Value>());
        }
    }
}

}  // namespace detail

template<typename T>
void sortDim(Param<T> val, const unsigned dim, const bool isAscending) {
    const af::dim4 dims       = val.dims();
    const dim_t line_length   = dims[dim];
    const size_t line_count   = detail::sortLineCount(dims, dim);
    if (line_length <= 1 || line_count == 0) { return; }

    T *const values             = val.get();
    const af::dim4 strides      = val.strides();
    const dim_t element_stride  = strides[dim];
    const size_t length         = static_cast<size_t>(line_length);
    const size_t min_lines      = detail::sortMinLinesPerTask(line_length);
    // Preserve the legacy value-sort policy: small dim-0 batches used
    // std::sort, while every former global-batched path was stable.
    const bool preserve_stable_order = dim != 0 || line_count > 10;

    parallelForRange(
        line_count, min_lines, [=](const size_t begin, const size_t end) {
            std::vector<T> scratch(element_stride == 1 ? 0 : length);
            detail::SortLineIndexer line_indexer(begin, dims, strides, dim);

            for (size_t line = begin; line < end; ++line) {
                T *const line_values = values + line_indexer.offset();

                if (element_stride == 1) {
                    detail::sortRange(line_values, line_values + line_length,
                                      isAscending, preserve_stable_order);
                } else {
                    for (size_t element = 0; element < length; ++element) {
                        scratch[element] =
                            line_values[static_cast<dim_t>(element) *
                                        element_stride];
                    }
                    detail::sortRange(scratch.begin(), scratch.end(),
                                      isAscending, preserve_stable_order);
                    for (size_t element = 0; element < length; ++element) {
                        line_values[static_cast<dim_t>(element) *
                                    element_stride] = scratch[element];
                    }
                }
                line_indexer.next();
            }
        });
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
