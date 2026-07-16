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
#include <err_cpu.hpp>
#include <kernel/sort_by_key.hpp>
#include <kernel/sort_helper.hpp>
#include <kernel/sort_utils.hpp>
#include <parallel.hpp>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename Tk, typename Tv>
void sortByKeyBatched(Param<Tk> okey, Param<Tv> oval, const unsigned dim,
                      bool isAscending) {
    const af::dim4 dims       = okey.dims();
    const dim_t line_length   = dims[dim];
    const size_t line_count   = detail::sortLineCount(dims, dim);
    if (line_length <= 1 || line_count == 0) { return; }

    Tk *const keys               = okey.get();
    Tv *const values             = oval.get();
    const af::dim4 key_strides   = okey.strides();
    const af::dim4 value_strides = oval.strides();
    const dim_t key_stride       = key_strides[dim];
    const dim_t value_stride     = value_strides[dim];
    const size_t length          = static_cast<size_t>(line_length);
    const size_t min_lines       = detail::sortMinLinesPerTask(line_length);

    using CurrentPair = IndexPair<Tk, Tv>;
    parallelForRange(
        line_count, min_lines, [=](const size_t begin, const size_t end) {
            std::vector<CurrentPair> pairs(length);
            detail::SortLineIndexer key_indexer(begin, dims, key_strides, dim);
            detail::SortLineIndexer value_indexer(begin, dims, value_strides,
                                                   dim);

            for (size_t line = begin; line < end; ++line) {
                const dim_t key_offset   = key_indexer.offset();
                const dim_t value_offset = value_indexer.offset();

                for (size_t element = 0; element < length; ++element) {
                    const dim_t key_index =
                        key_offset + static_cast<dim_t>(element) * key_stride;
                    const dim_t value_index = value_offset +
                                              static_cast<dim_t>(element) *
                                                  value_stride;
                    pairs[element] =
                        std::make_tuple(keys[key_index], values[value_index]);
                }

                if (isAscending) {
                    std::stable_sort(pairs.begin(), pairs.end(),
                                     IPCompare<Tk, Tv, true>());
                } else {
                    std::stable_sort(pairs.begin(), pairs.end(),
                                     IPCompare<Tk, Tv, false>());
                }

                for (size_t element = 0; element < length; ++element) {
                    const dim_t key_index =
                        key_offset + static_cast<dim_t>(element) * key_stride;
                    const dim_t value_index = value_offset +
                                              static_cast<dim_t>(element) *
                                                  value_stride;
                    keys[key_index]     = std::get<0>(pairs[element]);
                    values[value_index] = std::get<1>(pairs[element]);
                }
                key_indexer.next();
                value_indexer.next();
            }
        });
}

#define INSTANTIATE(Tk, Tv)                                                \
    template void sortByKeyBatched<Tk, Tv>(Param<Tk> okey, Param<Tv> oval, \
                                           const unsigned dim,             \
                                           bool isAscending);

#define INSTANTIATE1(Tk)     \
    INSTANTIATE(Tk, float)   \
    INSTANTIATE(Tk, double)  \
    INSTANTIATE(Tk, cfloat)  \
    INSTANTIATE(Tk, cdouble) \
    INSTANTIATE(Tk, int)     \
    INSTANTIATE(Tk, uint)    \
    INSTANTIATE(Tk, short)   \
    INSTANTIATE(Tk, ushort)  \
    INSTANTIATE(Tk, char)    \
    INSTANTIATE(Tk, schar)   \
    INSTANTIATE(Tk, uchar)   \
    INSTANTIATE(Tk, intl)    \
    INSTANTIATE(Tk, uintl)

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
