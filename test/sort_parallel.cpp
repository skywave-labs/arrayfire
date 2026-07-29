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
#include <cmath>
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
    return static_cast<size_t>(x + dims[0] * (y + dims[1] * (z + dims[2] * w)));
}

size_t lineElementIndex(const dim4 &dims, const unsigned dim, const size_t line,
                        const dim_t element) {
    dim4 line_dims = dims;
    line_dims[dim] = 1;

    size_t remaining = line;
    std::array<dim_t, 4> coord;
    for (unsigned axis = 0; axis < 4; ++axis) {
        coord[axis] = static_cast<dim_t>(remaining %
                                         static_cast<size_t>(line_dims[axis]));
        remaining /= static_cast<size_t>(line_dims[axis]);
    }
    coord[dim] = element;
    return linearIndex(dims, coord[0], coord[1], coord[2], coord[3]);
}

template<typename T, typename Compare>
void sortLinesReference(vector<T> &values, const dim4 &dims, const unsigned dim,
                        Compare compare) {
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
        std::stable_sort(line.begin(), line.end(),
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

template<typename T>
void assertFloatingKeys(const vector<T> &expected, const dim4 &dims,
                        const array &actual_array) {
    vector<T> actual(expected.size());
    actual_array.host(actual.data());

    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(expected[i], actual[i]) << "at: " << i;
        if (expected[i] == T(0)) {
            ASSERT_EQ(std::signbit(expected[i]), std::signbit(actual[i]))
                << "at: " << i;
        }
    }
    ASSERT_EQ(dims, actual_array.dims());
}

template<typename T>
void checkFloatingSignedZeroAssociations() {
    const dim4 dims(64, 257);
    const size_t elements = static_cast<size_t>(dims.elements());
    vector<T> keys(elements);
    vector<unsigned> payload(elements);

    for (dim_t line = 0; line < dims[1]; ++line) {
        for (dim_t element = 0; element < dims[0]; ++element) {
            const size_t index = static_cast<size_t>(line * dims[0] + element);
            switch (element % 8) {
                case 0:
                case 4: keys[index] = T(0); break;
                case 2:
                case 6: keys[index] = -T(0); break;
                default:
                    keys[index] =
                        static_cast<T>(static_cast<int>(element % 5) - 2);
            }
            payload[index] = static_cast<unsigned>(element);
        }
    }

    for (const bool ascending : {true, false}) {
        SCOPED_TRACE(::testing::Message() << "element_size=" << sizeof(T)
                                          << " ascending=" << ascending);
        vector<T> expected_keys           = keys;
        vector<unsigned> expected_payload = payload;
        if (ascending) {
            stableSortLinesByKeyReference(expected_keys, expected_payload, dims,
                                          0, std::less<T>());
        } else {
            stableSortLinesByKeyReference(expected_keys, expected_payload, dims,
                                          0, std::greater<T>());
        }

        const array key_array(dims, keys.data());
        const array payload_array(dims, payload.data());

        const array values_only = af::sort(key_array, 0, ascending);
        assertFloatingKeys(expected_keys, dims, values_only);

        array indexed_keys;
        array sorted_indices;
        af::sort(indexed_keys, sorted_indices, key_array, 0, ascending);
        assertFloatingKeys(expected_keys, dims, indexed_keys);
        ASSERT_VEC_ARRAY_EQ(expected_payload, dims, sorted_indices);

        array sorted_keys;
        array sorted_payload;
        af::sort(sorted_keys, sorted_payload, key_array, payload_array, 0,
                 ascending);
        assertFloatingKeys(expected_keys, dims, sorted_keys);
        ASSERT_VEC_ARRAY_EQ(expected_payload, dims, sorted_payload);
    }
}

void checkDim0ViewSorts(const array &key_view, const array &payload_view,
                        const vector<int> &keys,
                        const vector<unsigned> &payload, const dim4 &dims,
                        const bool ascending) {
    ASSERT_EQ(dims, key_view.dims());
    ASSERT_EQ(dims, payload_view.dims());

    vector<int> expected_values = keys;
    if (ascending) {
        sortLinesReference(expected_values, dims, 0, std::less<int>());
    } else {
        sortLinesReference(expected_values, dims, 0, std::greater<int>());
    }
    const array sorted_values = af::sort(key_view, 0, ascending);
    ASSERT_VEC_ARRAY_EQ(expected_values, dims, sorted_values);

    vector<unsigned> indices(static_cast<size_t>(dims.elements()));
    for (dim_t line = 0; line < dims[1]; ++line) {
        for (dim_t element = 0; element < dims[0]; ++element) {
            indices[static_cast<size_t>(line * dims[0] + element)] =
                static_cast<unsigned>(element);
        }
    }

    vector<int> expected_index_keys   = keys;
    vector<unsigned> expected_indices = indices;
    vector<int> expected_payload_keys = keys;
    vector<unsigned> expected_payload = payload;
    if (ascending) {
        stableSortLinesByKeyReference(expected_index_keys, expected_indices,
                                      dims, 0, std::less<int>());
        stableSortLinesByKeyReference(expected_payload_keys, expected_payload,
                                      dims, 0, std::less<int>());
    } else {
        stableSortLinesByKeyReference(expected_index_keys, expected_indices,
                                      dims, 0, std::greater<int>());
        stableSortLinesByKeyReference(expected_payload_keys, expected_payload,
                                      dims, 0, std::greater<int>());
    }

    array indexed_keys;
    array sorted_indices;
    af::sort(indexed_keys, sorted_indices, key_view, 0, ascending);
    ASSERT_VEC_ARRAY_EQ(expected_index_keys, dims, indexed_keys);
    ASSERT_VEC_ARRAY_EQ(expected_indices, dims, sorted_indices);

    array sorted_keys;
    array sorted_payload;
    af::sort(sorted_keys, sorted_payload, key_view, payload_view, 0, ascending);
    ASSERT_VEC_ARRAY_EQ(expected_payload_keys, dims, sorted_keys);
    ASSERT_VEC_ARRAY_EQ(expected_payload, dims, sorted_payload);
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
                    static_cast<int>(
                        (37 * static_cast<size_t>(element) + 13 * line) % 97) -
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

        vector<int> expected_index_keys   = keys;
        vector<unsigned> expected_indices = indices;
        vector<int> expected_payload_keys = keys;
        vector<unsigned> expected_payload = payload;
        if (ascending) {
            stableSortLinesByKeyReference(expected_index_keys, expected_indices,
                                          parallelDims, dim, std::less<int>());
            stableSortLinesByKeyReference(expected_payload_keys,
                                          expected_payload, parallelDims, dim,
                                          std::less<int>());
        } else {
            stableSortLinesByKeyReference(expected_index_keys, expected_indices,
                                          parallelDims, dim,
                                          std::greater<int>());
            stableSortLinesByKeyReference(expected_payload_keys,
                                          expected_payload, parallelDims, dim,
                                          std::greater<int>());
        }

        const array key_array(parallelDims, keys.data());
        {
            SCOPED_TRACE(::testing::Message() << "sortIndex dim=" << dim
                                              << " ascending=" << ascending);
            array sorted_keys;
            array sorted_indices;
            af::sort(sorted_keys, sorted_indices, key_array, dim, ascending);
            ASSERT_VEC_ARRAY_EQ(expected_index_keys, parallelDims, sorted_keys);
            ASSERT_VEC_ARRAY_EQ(expected_indices, parallelDims, sorted_indices);
        }

        const array payload_array(parallelDims, payload.data());
        {
            SCOPED_TRACE(::testing::Message() << "sortByKey dim=" << dim
                                              << " ascending=" << ascending);
            array sorted_payload_keys;
            array sorted_payload;
            af::sort(sorted_payload_keys, sorted_payload, key_array,
                     payload_array, dim, ascending);
            ASSERT_VEC_ARRAY_EQ(expected_payload_keys, parallelDims,
                                sorted_payload_keys);
            ASSERT_VEC_ARRAY_EQ(expected_payload, parallelDims, sorted_payload);
        }
    }
}

TEST(SortParallel, FloatingSignedZeroAssociationsRemainStable) {
    SKIP_IF_FAST_MATH_ENABLED();
    checkFloatingSignedZeroAssociations<float>();
    checkFloatingSignedZeroAssociations<double>();
}

TEST(SortParallel, QueuedSegmentedSortReusesBuffersSafely) {
    const dim4 dims(32, 1024);
    const size_t elements = static_cast<size_t>(dims.elements());
    vector<int> values(elements);
    for (size_t i = 0; i < elements; ++i) {
        values[i] = static_cast<int>((i * 37 + i / 32) % 101) - 50;
    }

    vector<int> expected = values;
    sortLinesReference(expected, dims, 0, std::less<int>());

    const array input(dims, values.data());
    vector<array> outputs;
    outputs.reserve(16);
    for (int i = 0; i < 16; ++i) {
        outputs.emplace_back(af::sort(input, 0, true));
    }

    for (const array &output : outputs) {
        ASSERT_VEC_ARRAY_EQ(expected, dims, output);
    }
}

TEST(SortParallel, QueuedNoSyncFallbackReusesBuffersSafely) {
    // Four lines stay below both segmented-sort thresholds, exercising the
    // no-sync Thrust value and pair fallbacks.
    const dim4 dims(64, 4);
    const size_t elements = static_cast<size_t>(dims.elements());
    vector<int> keys(elements);
    vector<unsigned> payload(elements);
    for (dim_t line = 0; line < dims[1]; ++line) {
        for (dim_t element = 0; element < dims[0]; ++element) {
            const size_t index = static_cast<size_t>(line * dims[0] + element);
            keys[index] = static_cast<int>((element * 19 + line * 7) % 11) - 5;
            payload[index] = static_cast<unsigned>(line * dims[0] + element);
        }
    }

    vector<int> expected_values = keys;
    sortLinesReference(expected_values, dims, 0, std::less<int>());
    vector<int> expected_keys         = keys;
    vector<unsigned> expected_payload = payload;
    stableSortLinesByKeyReference(expected_keys, expected_payload, dims, 0,
                                  std::less<int>());
    vector<unsigned> expected_indices(elements);
    for (dim_t line = 0; line < dims[1]; ++line) {
        for (dim_t element = 0; element < dims[0]; ++element) {
            expected_indices[static_cast<size_t>(line * dims[0] + element)] =
                static_cast<unsigned>(element);
        }
    }
    vector<int> expected_index_keys = keys;
    stableSortLinesByKeyReference(expected_index_keys, expected_indices, dims,
                                  0, std::less<int>());

    const array key_array(dims, keys.data());
    const array payload_array(dims, payload.data());
    vector<array> value_outputs;
    vector<array> indexed_key_outputs;
    vector<array> index_outputs;
    vector<array> keyed_key_outputs;
    vector<array> keyed_value_outputs;
    constexpr int queued = 12;
    value_outputs.reserve(queued);
    indexed_key_outputs.reserve(queued);
    index_outputs.reserve(queued);
    keyed_key_outputs.reserve(queued);
    keyed_value_outputs.reserve(queued);

    for (int i = 0; i < queued; ++i) {
        value_outputs.emplace_back(af::sort(key_array, 0, true));

        array indexed_keys;
        array sorted_indices;
        af::sort(indexed_keys, sorted_indices, key_array, 0, true);
        indexed_key_outputs.emplace_back(indexed_keys);
        index_outputs.emplace_back(sorted_indices);

        array sorted_keys;
        array sorted_payload;
        af::sort(sorted_keys, sorted_payload, key_array, payload_array, 0,
                 true);
        keyed_key_outputs.emplace_back(sorted_keys);
        keyed_value_outputs.emplace_back(sorted_payload);
    }

    for (int i = 0; i < queued; ++i) {
        ASSERT_VEC_ARRAY_EQ(expected_values, dims, value_outputs[i]);
        ASSERT_VEC_ARRAY_EQ(expected_index_keys, dims, indexed_key_outputs[i]);
        ASSERT_VEC_ARRAY_EQ(expected_indices, dims, index_outputs[i]);
        ASSERT_VEC_ARRAY_EQ(expected_keys, dims, keyed_key_outputs[i]);
        ASSERT_VEC_ARRAY_EQ(expected_payload, dims, keyed_value_outputs[i]);
    }
}

TEST(SortParallel, GappedViewAcrossNonzeroDimension) {
    const dim4 dims(17, 13, 5);
    const dim4 base_dims(dims[0] * 2, dims[1], dims[2]);
    const size_t elements = static_cast<size_t>(dims.elements());
    vector<int> keys(elements);
    vector<unsigned> indices(elements);
    vector<unsigned> payload(elements);
    vector<int> key_storage(static_cast<size_t>(base_dims.elements()), 99);
    vector<unsigned> payload_storage(static_cast<size_t>(base_dims.elements()),
                                     0);

    for (dim_t z = 0; z < dims[2]; ++z) {
        for (dim_t y = 0; y < dims[1]; ++y) {
            for (dim_t x = 0; x < dims[0]; ++x) {
                const size_t index = linearIndex(dims, x, y, z, 0);
                const size_t base_index =
                    linearIndex(base_dims, x * 2, y, z, 0);
                keys[index] = static_cast<int>((y * 7 + x * 3 + z * 5) % 9) - 4;
                indices[index]          = static_cast<unsigned>(y);
                payload[index]          = static_cast<unsigned>(index + 1000);
                key_storage[base_index] = keys[index];
                payload_storage[base_index] = payload[index];
            }
        }
    }

    const array key_base(base_dims, key_storage.data());
    const array payload_base(base_dims, payload_storage.data());
    const af::seq every_other_x(0, base_dims[0] - 2, 2);
    const array key_view =
        key_base(every_other_x, af::span, af::span, af::span);
    const array payload_view =
        payload_base(every_other_x, af::span, af::span, af::span);

    vector<int> expected_values = keys;
    sortLinesReference(expected_values, dims, 1, std::greater<int>());
    const array sorted_values = af::sort(key_view, 1, false);
    ASSERT_VEC_ARRAY_EQ(expected_values, dims, sorted_values);

    vector<int> expected_index_keys   = keys;
    vector<unsigned> expected_indices = indices;
    stableSortLinesByKeyReference(expected_index_keys, expected_indices, dims,
                                  1, std::greater<int>());
    array indexed_keys;
    array sorted_indices;
    af::sort(indexed_keys, sorted_indices, key_view, 1, false);
    ASSERT_VEC_ARRAY_EQ(expected_index_keys, dims, indexed_keys);
    ASSERT_VEC_ARRAY_EQ(expected_indices, dims, sorted_indices);

    vector<int> expected_keys         = keys;
    vector<unsigned> expected_payload = payload;
    stableSortLinesByKeyReference(expected_keys, expected_payload, dims, 1,
                                  std::greater<int>());
    array sorted_keys;
    array sorted_payload;
    af::sort(sorted_keys, sorted_payload, key_view, payload_view, 1, false);
    ASSERT_VEC_ARRAY_EQ(expected_keys, dims, sorted_keys);
    ASSERT_VEC_ARRAY_EQ(expected_payload, dims, sorted_payload);
}

TEST(SortParallel, GappedViewAcrossSegmentedDim0) {
    const dim4 dims(32, 17);
    const dim4 base_dims(dims[0], dims[1] * 2);
    const size_t elements = static_cast<size_t>(dims.elements());
    vector<int> keys(elements);
    vector<unsigned> payload(elements);
    vector<int> key_storage(static_cast<size_t>(base_dims.elements()), 99);
    vector<unsigned> payload_storage(static_cast<size_t>(base_dims.elements()),
                                     0);

    for (dim_t line = 0; line < dims[1]; ++line) {
        for (dim_t element = 0; element < dims[0]; ++element) {
            const size_t index = static_cast<size_t>(line * dims[0] + element);
            const size_t base_index =
                static_cast<size_t>(line * 2 * base_dims[0] + element);
            keys[index] = static_cast<int>((element * 11 + line * 5) % 13) - 6;
            payload[index]              = static_cast<unsigned>(index + 1000);
            key_storage[base_index]     = keys[index];
            payload_storage[base_index] = payload[index];
        }
    }

    const array key_base(base_dims, key_storage.data());
    const array payload_base(base_dims, payload_storage.data());
    const af::seq every_other_line(0, base_dims[1] - 2, 2);
    const array key_view =
        key_base(af::span, every_other_line, af::span, af::span);
    const array payload_view =
        payload_base(af::span, every_other_line, af::span, af::span);

    checkDim0ViewSorts(key_view, payload_view, keys, payload, dims, true);
}

TEST(SortParallel, LinearOffsetViewAcrossSegmentedDim0) {
    const dim4 dims(32, 17);
    const dim4 base_dims(dims[0], dims[1] + 2);
    const size_t elements = static_cast<size_t>(dims.elements());
    vector<int> keys(elements);
    vector<unsigned> payload(elements);
    vector<int> key_storage(static_cast<size_t>(base_dims.elements()), 99);
    vector<unsigned> payload_storage(static_cast<size_t>(base_dims.elements()),
                                     0);

    for (dim_t line = 0; line < dims[1]; ++line) {
        for (dim_t element = 0; element < dims[0]; ++element) {
            const size_t index = static_cast<size_t>(line * dims[0] + element);
            const size_t base_index =
                static_cast<size_t>((line + 1) * base_dims[0] + element);
            keys[index] = static_cast<int>((element * 7 + line * 3) % 11) - 5;
            payload[index]              = static_cast<unsigned>(index + 2000);
            key_storage[base_index]     = keys[index];
            payload_storage[base_index] = payload[index];
        }
    }

    const array key_base(base_dims, key_storage.data());
    const array payload_base(base_dims, payload_storage.data());
    const af::seq interior_lines(1, dims[1]);
    const array key_view =
        key_base(af::span, interior_lines, af::span, af::span);
    const array payload_view =
        payload_base(af::span, interior_lines, af::span, af::span);

    checkDim0ViewSorts(key_view, payload_view, keys, payload, dims, false);
}
