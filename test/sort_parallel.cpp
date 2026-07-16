/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

using af::array;
using af::dim4;
using std::vector;

namespace {

const dim4 parallelDims(257, 17, 7, 20);

size_t linearIndex(const dim4 &dims, const dim_t x, const dim_t y,
                   const dim_t z, const dim_t w) {
    return static_cast<size_t>(
        x + dims[0] * (y + dims[1] * (z + dims[2] * w)));
}

size_t lineElementIndex(const dim4 &dims, const unsigned dim,
                        const size_t line, const dim_t element) {
    dim4 line_dims = dims;
    line_dims[dim] = 1;

    size_t remaining = line;
    std::array<dim_t, 4> coord;
    for (unsigned axis = 0; axis < 4; ++axis) {
        coord[axis] = static_cast<dim_t>(
            remaining % static_cast<size_t>(line_dims[axis]));
        remaining /= static_cast<size_t>(line_dims[axis]);
    }
    coord[dim] = element;
    return linearIndex(dims, coord[0], coord[1], coord[2], coord[3]);
}

template<typename T, typename Compare>
void sortLinesReference(vector<T> &values, const dim4 &dims,
                        const unsigned dim, Compare compare) {
    const dim_t line_length = dims[dim];
    const size_t line_count =
        static_cast<size_t>(dims.elements() / line_length);
    vector<T> line(static_cast<size_t>(line_length));

    for (size_t line_index = 0; line_index < line_count; ++line_index) {
        for (dim_t element = 0; element < line_length; ++element) {
            line[static_cast<size_t>(element)] =
                values[lineElementIndex(dims, dim, line_index, element)];
        }
        std::sort(line.begin(), line.end(), compare);
        for (dim_t element = 0; element < line_length; ++element) {
            values[lineElementIndex(dims, dim, line_index, element)] =
                line[static_cast<size_t>(element)];
        }
    }
}

template<typename Tk, typename Tv, typename Compare>
void stableSortLinesByKeyReference(vector<Tk> &keys, vector<Tv> &values,
                                   const dim4 &dims, const unsigned dim,
                                   Compare compare) {
    const dim_t line_length = dims[dim];
    const size_t line_count =
        static_cast<size_t>(dims.elements() / line_length);
    vector<std::pair<Tk, Tv>> line(static_cast<size_t>(line_length));

    for (size_t line_index = 0; line_index < line_count; ++line_index) {
        for (dim_t element = 0; element < line_length; ++element) {
            const size_t index =
                lineElementIndex(dims, dim, line_index, element);
            line[static_cast<size_t>(element)] =
                std::make_pair(keys[index], values[index]);
        }
        std::stable_sort(
            line.begin(), line.end(),
            [compare](const std::pair<Tk, Tv> &lhs,
                      const std::pair<Tk, Tv> &rhs) {
                return compare(lhs.first, rhs.first);
            });
        for (dim_t element = 0; element < line_length; ++element) {
            const size_t index =
                lineElementIndex(dims, dim, line_index, element);
            keys[index]   = line[static_cast<size_t>(element)].first;
            values[index] = line[static_cast<size_t>(element)].second;
        }
    }
}

}  // namespace

TEST(SortParallel, ValuesAcrossAllDimensions) {
    const size_t elements = static_cast<size_t>(parallelDims.elements());

    for (unsigned dim = 0; dim < 4; ++dim) {
        const bool ascending    = dim % 2 == 0;
        const dim_t line_length = parallelDims[dim];
        const size_t line_count = elements / static_cast<size_t>(line_length);
        vector<int> values(elements);

        for (size_t line = 0; line < line_count; ++line) {
            for (dim_t element = 0; element < line_length; ++element) {
                const size_t index =
                    lineElementIndex(parallelDims, dim, line, element);
                values[index] =
                    static_cast<int>((37 * static_cast<size_t>(element) +
                                      13 * line) %
                                     97) -
                    48;
            }
        }

        vector<int> expected = values;
        if (ascending) {
            sortLinesReference(expected, parallelDims, dim, std::less<int>());
        } else {
            sortLinesReference(expected, parallelDims, dim,
                               std::greater<int>());
        }

        SCOPED_TRACE(::testing::Message()
                     << "dim=" << dim << " ascending=" << ascending);
        const array input(parallelDims, values.data());
        const array output = af::sort(input, dim, ascending);
        ASSERT_VEC_ARRAY_EQ(expected, parallelDims, output);
    }
}

TEST(SortParallel, StableAssociationsAcrossAllDimensions) {
    const size_t elements = static_cast<size_t>(parallelDims.elements());

    for (unsigned dim = 0; dim < 4; ++dim) {
        const bool ascending    = dim % 2 != 0;
        const dim_t line_length = parallelDims[dim];
        const size_t line_count = elements / static_cast<size_t>(line_length);
        vector<int> keys(elements);
        vector<unsigned> indices(elements);
        vector<unsigned> payload(elements);

        for (size_t line = 0; line < line_count; ++line) {
            for (dim_t element = 0; element < line_length; ++element) {
                const size_t index =
                    lineElementIndex(parallelDims, dim, line, element);
                keys[index] =
                    static_cast<int>((2 * static_cast<size_t>(element) + line) %
                                     3) -
                    1;
                indices[index] = static_cast<unsigned>(element);
                payload[index] = static_cast<unsigned>(
                    line * static_cast<size_t>(line_length) +
                    static_cast<size_t>(element));
            }
        }

        vector<int> expected_index_keys = keys;
        vector<unsigned> expected_indices = indices;
        vector<int> expected_payload_keys = keys;
        vector<unsigned> expected_payload = payload;
        if (ascending) {
            stableSortLinesByKeyReference(expected_index_keys,
                                          expected_indices, parallelDims, dim,
                                          std::less<int>());
            stableSortLinesByKeyReference(expected_payload_keys,
                                          expected_payload, parallelDims, dim,
                                          std::less<int>());
        } else {
            stableSortLinesByKeyReference(expected_index_keys,
                                          expected_indices, parallelDims, dim,
                                          std::greater<int>());
            stableSortLinesByKeyReference(expected_payload_keys,
                                          expected_payload, parallelDims, dim,
                                          std::greater<int>());
        }

        const array key_array(parallelDims, keys.data());
        {
            SCOPED_TRACE(::testing::Message()
                         << "sortIndex dim=" << dim
                         << " ascending=" << ascending);
            array sorted_keys;
            array sorted_indices;
            af::sort(sorted_keys, sorted_indices, key_array, dim, ascending);
            ASSERT_VEC_ARRAY_EQ(expected_index_keys, parallelDims,
                                sorted_keys);
            ASSERT_VEC_ARRAY_EQ(expected_indices, parallelDims,
                                sorted_indices);
        }

        const array payload_array(parallelDims, payload.data());
        {
            SCOPED_TRACE(::testing::Message()
                         << "sortByKey dim=" << dim
                         << " ascending=" << ascending);
            array sorted_payload_keys;
            array sorted_payload;
            af::sort(sorted_payload_keys, sorted_payload, key_array,
                     payload_array, dim, ascending);
            ASSERT_VEC_ARRAY_EQ(expected_payload_keys, parallelDims,
                                sorted_payload_keys);
            ASSERT_VEC_ARRAY_EQ(expected_payload, parallelDims,
                                sorted_payload);
        }
    }
}
