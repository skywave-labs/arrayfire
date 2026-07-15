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
#include <half.hpp>
#include <testHelpers.hpp>
#include <af/internal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

using af::array;
using af::cfloat;
using af::dim4;
using af::seq;
using af::span;
using std::vector;

namespace {

constexpr size_t blockElements    = 1 << 16;
constexpr size_t parallelElements = 9 * blockElements;
// Eight worker-sized input grains regardless of the reduced dimension.
const dim4 dimensionalDims(256, 32, 16, 4);

size_t linearIndex(const dim4 &dims, const dim_t x, const dim_t y,
                   const dim_t z, const dim_t w) {
    return static_cast<size_t>(
        x + dims[0] * (y + dims[1] * (z + dims[2] * w)));
}

size_t lineElementIndex(const dim4 &dims, const int dim, const size_t line,
                        const dim_t reducedIndex) {
    dim4 odims = dims;
    odims[dim] = 1;

    size_t remaining = line;
    std::array<dim_t, 4> coord;
    for (int axis = 0; axis < 4; ++axis) {
        coord[axis] = static_cast<dim_t>(
            remaining % static_cast<size_t>(odims[axis]));
        remaining /= static_cast<size_t>(odims[axis]);
    }
    coord[dim] = reducedIndex;
    return linearIndex(dims, coord[0], coord[1], coord[2], coord[3]);
}

array makeGappedView(const array &dense) {
    const dim4 dims = dense.dims();
    const dim4 parentDims(dims[0] + 2, dims[1] + 2, dims[2] + 2,
                          dims[3] + 2);
    array parent = af::constant(0, parentDims, dense.type());
    const seq x(1, dims[0]);
    const seq y(1, dims[1]);
    const seq z(1, dims[2]);
    const seq w(1, dims[3]);
    parent(x, y, z, w) = dense;
    return parent(x, y, z, w);
}

template<typename T>
vector<T> hostVector(const array &input) {
    vector<T> result(input.elements());
    input.host(result.data());
    return result;
}

void expectComplexEqual(const cfloat expected, const cfloat actual) {
    EXPECT_FLOAT_EQ(af::real(expected), af::real(actual));
    EXPECT_FLOAT_EQ(af::imag(expected), af::imag(actual));
}

void expectRawInfNaN(const cfloat value) {
    EXPECT_TRUE(std::isinf(af::real(value)));
    EXPECT_GT(af::real(value), 0.f);
    EXPECT_TRUE(std::isnan(af::imag(value)));
}

}  // namespace

TEST(IReduceAllParallel, RealTiesPreserveIndexAndSignedZero) {
    SKIP_IF_FAST_MATH_ENABLED();
    const size_t first = blockElements - 1;
    const size_t last  = parallelElements - 1;

    vector<float> minValues(parallelElements, 2.f);
    minValues[first]         = 0.f;
    minValues[blockElements] = 0.f;
    minValues[last]          = -0.f;
    const array minInput(static_cast<dim_t>(parallelElements),
                         minValues.data());

    float minValue;
    unsigned minIndex;
    af::min<float>(&minValue, &minIndex, minInput);
    EXPECT_EQ(last, static_cast<size_t>(minIndex));
    EXPECT_EQ(0.f, minValue);
    EXPECT_TRUE(std::signbit(minValue));

    vector<float> maxValues(parallelElements, -2.f);
    maxValues[first]         = -0.f;
    maxValues[blockElements] = 0.f;
    maxValues[last]          = 0.f;
    const array maxInput(static_cast<dim_t>(parallelElements),
                         maxValues.data());

    float maxValue;
    unsigned maxIndex;
    af::max<float>(&maxValue, &maxIndex, maxInput);
    EXPECT_EQ(first, static_cast<size_t>(maxIndex));
    EXPECT_EQ(0.f, maxValue);
    EXPECT_TRUE(std::signbit(maxValue));
}

TEST(IReduceAllParallel, ComplexMagnitudeTiesPreserveIndexAndPayload) {
    const size_t first = blockElements - 1;
    const size_t last  = parallelElements - 1;

    vector<cfloat> minValues(parallelElements, cfloat(3.f, 0.f));
    minValues[first]         = cfloat(1.f, 0.f);
    minValues[blockElements] = cfloat(0.f, 1.f);
    const cfloat expectedMin(-1.f, 0.f);
    minValues[last] = expectedMin;
    const array minInput(static_cast<dim_t>(parallelElements),
                         minValues.data());

    cfloat minValue;
    unsigned minIndex;
    af::min<cfloat>(&minValue, &minIndex, minInput);
    EXPECT_EQ(last, static_cast<size_t>(minIndex));
    expectComplexEqual(expectedMin, minValue);

    vector<cfloat> maxValues(parallelElements, cfloat(0.25f, 0.f));
    const cfloat expectedMax(4.f, 0.f);
    maxValues[first]         = expectedMax;
    maxValues[blockElements] = cfloat(0.f, 4.f);
    maxValues[last]          = cfloat(-4.f, 0.f);
    const array maxInput(static_cast<dim_t>(parallelElements),
                         maxValues.data());

    cfloat maxValue;
    unsigned maxIndex;
    af::max<cfloat>(&maxValue, &maxIndex, maxInput);
    EXPECT_EQ(first, static_cast<size_t>(maxIndex));
    expectComplexEqual(expectedMax, maxValue);
}

TEST(IReduceAllParallel, NaNIdentitiesAndInvalidBlocksPreserveIndices) {
    SKIP_IF_FAST_MATH_ENABLED();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    vector<float> values(parallelElements, nan);
    array input(static_cast<dim_t>(parallelElements), values.data());

    float minValue;
    float maxValue;
    unsigned minIndex;
    unsigned maxIndex;
    af::min<float>(&minValue, &minIndex, input);
    af::max<float>(&maxValue, &maxIndex, input);
    EXPECT_TRUE(std::isinf(minValue));
    EXPECT_GT(minValue, 0.f);
    EXPECT_EQ(0u, minIndex);
    EXPECT_TRUE(std::isinf(maxValue));
    EXPECT_LT(maxValue, 0.f);
    EXPECT_EQ(0u, maxIndex);

    // A valid value equal to the min identity must not be displaced by the
    // identity summaries of later, entirely-NaN blocks.
    const size_t validIndex = blockElements - 1;
    values[validIndex]      = inf;
    input = array(static_cast<dim_t>(parallelElements), values.data());
    af::min<float>(&minValue, &minIndex, input);
    EXPECT_TRUE(std::isinf(minValue));
    EXPECT_GT(minValue, 0.f);
    EXPECT_EQ(validIndex, static_cast<size_t>(minIndex));

    // When element zero is NaN, a max identity at that index wins ties with
    // later real values equal to the identity.
    std::fill(values.begin(), values.end(), -inf);
    values.front() = nan;
    input          = array(static_cast<dim_t>(parallelElements), values.data());
    af::max<float>(&maxValue, &maxIndex, input);
    EXPECT_TRUE(std::isinf(maxValue));
    EXPECT_LT(maxValue, 0.f);
    EXPECT_EQ(0u, maxIndex);
}

TEST(IReduceAllParallel, ComplexInfNaNEventsKeepRawComparisonBehavior) {
    SKIP_IF_FAST_MATH_ENABLED();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const cfloat ordinaryNaN(nan, 0.f);
    const cfloat rawEvent(inf, nan);

    // The constructor maps a NaN-containing first element to the operation
    // identity. Reconsidering the raw event uses its infinite magnitude:
    // min keeps its identity tie, while max selects the original payload.
    vector<cfloat> values(parallelElements, ordinaryNaN);
    values.front() = rawEvent;
    array input(static_cast<dim_t>(parallelElements), values.data());

    cfloat minValue;
    cfloat maxValue;
    unsigned minIndex;
    unsigned maxIndex;
    af::min<cfloat>(&minValue, &minIndex, input);
    af::max<cfloat>(&maxValue, &maxIndex, input);
    EXPECT_TRUE(std::isinf(af::real(minValue)));
    EXPECT_GT(af::real(minValue), 0.f);
    EXPECT_EQ(0.f, af::imag(minValue));
    EXPECT_EQ(0u, minIndex);
    expectRawInfNaN(maxValue);
    EXPECT_EQ(0u, maxIndex);

    // Outside element zero, the same payload is a valid infinite-magnitude
    // event rather than an ordinary NaN and must survive a cross-block merge.
    const size_t eventIndex = blockElements;
    values.front()          = ordinaryNaN;
    values[eventIndex]      = rawEvent;
    input = array(static_cast<dim_t>(parallelElements), values.data());
    af::min<cfloat>(&minValue, &minIndex, input);
    af::max<cfloat>(&maxValue, &maxIndex, input);
    expectRawInfNaN(minValue);
    EXPECT_EQ(eventIndex, static_cast<size_t>(minIndex));
    expectRawInfNaN(maxValue);
    EXPECT_EQ(eventIndex, static_cast<size_t>(maxIndex));
}

TEST(IReduceAllParallel, GappedFourDimensionalViewUsesLogicalIndices) {
    const dim4 parentDims(259, 69, 5, 13);
    const dim4 viewDims(259, 67, 5, 13);
    vector<float> values(parentDims.elements(), 1.f);

    const dim_t minX0 = 17, minY0 = 2, minZ0 = 1, minW0 = 1;
    const dim_t minX1 = 128, minY1 = 60, minZ1 = 4, minW1 = 12;
    values[linearIndex(parentDims, minX0, minY0, minZ0, minW0)] = -5.f;
    values[linearIndex(parentDims, minX1, minY1, minZ1, minW1)] = -5.f;

    const dim_t maxX0 = 42, maxY0 = 3, maxZ0 = 0, maxW0 = 0;
    const dim_t maxX1 = 200, maxY1 = 50, maxZ1 = 3, maxW1 = 10;
    values[linearIndex(parentDims, maxX0, maxY0, maxZ0, maxW0)] = 9.f;
    values[linearIndex(parentDims, maxX1, maxY1, maxZ1, maxW1)] = 9.f;

    const array parent(parentDims, values.data());
    const array view = parent(span, seq(1, 67), span, span);
    const dim4 viewStrides = af::getStrides(view);
    ASSERT_EQ(1, viewStrides[0]);
    ASSERT_EQ(parentDims[0], viewStrides[1]);
    ASSERT_EQ(parentDims[0] * parentDims[1], viewStrides[2]);
    ASSERT_EQ(parentDims[0] * parentDims[1] * parentDims[2], viewStrides[3]);

    float minValue;
    float maxValue;
    unsigned minIndex;
    unsigned maxIndex;
    af::min<float>(&minValue, &minIndex, view);
    af::max<float>(&maxValue, &maxIndex, view);

    const size_t expectedMin =
        linearIndex(viewDims, minX1, minY1 - 1, minZ1, minW1);
    const size_t expectedMax =
        linearIndex(viewDims, maxX0, maxY0 - 1, maxZ0, maxW0);
    EXPECT_EQ(-5.f, minValue);
    EXPECT_EQ(expectedMin, static_cast<size_t>(minIndex));
    EXPECT_EQ(9.f, maxValue);
    EXPECT_EQ(expectedMax, static_cast<size_t>(maxIndex));
}

TEST(IReduceAllParallel, EvaluatesLargeLazyInputBeforeWorkerReads) {
    const dim_t elements = static_cast<dim_t>(parallelElements);
    const array lazy = af::range(dim4(elements)) * -2.f + 5.f;

    float minValue;
    float maxValue;
    unsigned minIndex;
    unsigned maxIndex;
    af::min<float>(&minValue, &minIndex, lazy);
    af::max<float>(&maxValue, &maxIndex, lazy);

    EXPECT_EQ(5.f - 2.f * static_cast<float>(elements - 1), minValue);
    EXPECT_EQ(parallelElements - 1, static_cast<size_t>(minIndex));
    EXPECT_EQ(5.f, maxValue);
    EXPECT_EQ(0u, maxIndex);
}

TEST(IReduceAllParallel, CApiU64PreservesDoubleKeyCollisions) {
    const unsigned long long twoTo53 = 9007199254740992ULL;
    const size_t first               = blockElements - 1;
    const size_t second              = blockElements;

    vector<unsigned long long> values(parallelElements, twoTo53 + 1024);
    values[first]  = twoTo53;
    values[second] = twoTo53 + 1;
    array input(static_cast<dim_t>(parallelElements), values.data());

    double real = 0;
    double imag = 1;
    unsigned index;
    ASSERT_EQ(AF_SUCCESS, af_imin_all(&real, &imag, &index, input.get()));
    EXPECT_EQ(static_cast<double>(twoTo53), real);
    EXPECT_EQ(0., imag);
    EXPECT_EQ(second, static_cast<size_t>(index));

    std::fill(values.begin(), values.end(), 0ULL);
    values[first]  = twoTo53;
    values[second] = twoTo53 + 1;
    input = array(static_cast<dim_t>(parallelElements), values.data());
    real  = 0;
    imag  = 1;
    ASSERT_EQ(AF_SUCCESS, af_imax_all(&real, &imag, &index, input.get()));
    EXPECT_EQ(static_cast<double>(twoTo53), real);
    EXPECT_EQ(0., imag);
    EXPECT_EQ(first, static_cast<size_t>(index));
}

TEST(IReduceAllParallel, BooleanKeysPreserveRawCharPayloads) {
    vector<char> values(parallelElements, 1);
    values.front() = 2;
    values.back()  = 3;
    const array input(static_cast<dim_t>(parallelElements), values.data());

    char minValue;
    char maxValue;
    unsigned minIndex;
    unsigned maxIndex;
    af::min<char>(&minValue, &minIndex, input);
    af::max<char>(&maxValue, &maxIndex, input);

    EXPECT_EQ(3, minValue);
    EXPECT_EQ(parallelElements - 1, static_cast<size_t>(minIndex));
    EXPECT_EQ(2, maxValue);
    EXPECT_EQ(0u, maxIndex);
}

TEST(IReduceDimParallel, GappedTiesPreserveIndicesAcrossAllDimensions) {
    for (int dim = 0; dim < 4; ++dim) {
        const dim_t reducedElements = dimensionalDims[dim];
        const size_t lineCount = static_cast<size_t>(
            dimensionalDims.elements() / reducedElements);
        vector<float> values(dimensionalDims.elements(), 1.f);

        for (size_t line = 0; line < lineCount; ++line) {
            values[lineElementIndex(dimensionalDims, dim, line, 0)] = 9.f;
            values[lineElementIndex(dimensionalDims, dim, line,
                                    reducedElements - 2)] = 9.f;
            values[lineElementIndex(dimensionalDims, dim, line, 1)] = -7.f;
            values[lineElementIndex(dimensionalDims, dim, line,
                                    reducedElements - 1)] = -7.f;
        }

        const array input = makeGappedView(array(dimensionalDims,
                                                 values.data()));
        const dim4 strides = af::getStrides(input);
        ASSERT_EQ(1, strides[0]);
        ASSERT_EQ(dimensionalDims[0] + 2, strides[1]);

        array minValues;
        array minIndices;
        array maxValues;
        array maxIndices;
        af::min(minValues, minIndices, input, dim);
        af::max(maxValues, maxIndices, input, dim);

        const vector<float> hostMinValues = hostVector<float>(minValues);
        const vector<unsigned> hostMinIndices =
            hostVector<unsigned>(minIndices);
        const vector<float> hostMaxValues = hostVector<float>(maxValues);
        const vector<unsigned> hostMaxIndices =
            hostVector<unsigned>(maxIndices);
        ASSERT_EQ(lineCount, hostMinValues.size());
        ASSERT_EQ(lineCount, hostMaxValues.size());
        for (size_t line = 0; line < lineCount; ++line) {
            EXPECT_EQ(-7.f, hostMinValues[line]);
            EXPECT_EQ(static_cast<unsigned>(reducedElements - 1),
                      hostMinIndices[line]);
            EXPECT_EQ(9.f, hostMaxValues[line]);
            EXPECT_EQ(0u, hostMaxIndices[line]);
        }
    }
}

TEST(IReduceDimParallel, RealSpecialValuesPreserveScalarSemantics) {
    SKIP_IF_FAST_MATH_ENABLED();
    const int dim               = 0;
    const dim_t reducedElements = dimensionalDims[dim];
    const size_t lineCount      = static_cast<size_t>(
        dimensionalDims.elements() / reducedElements);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    vector<float> minInput(dimensionalDims.elements(), 2.f);
    vector<float> maxInput(dimensionalDims.elements(), -2.f);

    for (size_t line = 0; line < lineCount; ++line) {
        if (line % 3 == 0) {
            minInput[lineElementIndex(dimensionalDims, dim, line, 1)] = 0.f;
            minInput[lineElementIndex(dimensionalDims, dim, line,
                                      reducedElements - 1)] = -0.f;
            maxInput[lineElementIndex(dimensionalDims, dim, line, 1)] = -0.f;
            maxInput[lineElementIndex(dimensionalDims, dim, line,
                                      reducedElements - 1)] = 0.f;
        } else if (line % 3 == 1) {
            for (dim_t reduced = 0; reduced < reducedElements; ++reduced) {
                minInput[lineElementIndex(dimensionalDims, dim, line,
                                          reduced)] = nan;
                maxInput[lineElementIndex(dimensionalDims, dim, line,
                                          reduced)] = nan;
            }
        } else {
            minInput[lineElementIndex(dimensionalDims, dim, line, 0)] = nan;
            maxInput[lineElementIndex(dimensionalDims, dim, line, 0)] = nan;
            minInput[lineElementIndex(dimensionalDims, dim, line, 5)] = -1.f;
            maxInput[lineElementIndex(dimensionalDims, dim, line, 5)] = 1.f;
        }
    }

    array minValues;
    array minIndices;
    array maxValues;
    array maxIndices;
    af::min(minValues, minIndices,
            array(dimensionalDims, minInput.data()), dim);
    af::max(maxValues, maxIndices,
            array(dimensionalDims, maxInput.data()), dim);

    const vector<float> hostMinValues = hostVector<float>(minValues);
    const vector<unsigned> hostMinIndices =
        hostVector<unsigned>(minIndices);
    const vector<float> hostMaxValues = hostVector<float>(maxValues);
    const vector<unsigned> hostMaxIndices =
        hostVector<unsigned>(maxIndices);
    for (size_t line = 0; line < lineCount; ++line) {
        if (line % 3 == 0) {
            EXPECT_EQ(0.f, hostMinValues[line]);
            EXPECT_TRUE(std::signbit(hostMinValues[line]));
            EXPECT_EQ(static_cast<unsigned>(reducedElements - 1),
                      hostMinIndices[line]);
            EXPECT_EQ(0.f, hostMaxValues[line]);
            EXPECT_TRUE(std::signbit(hostMaxValues[line]));
            EXPECT_EQ(1u, hostMaxIndices[line]);
        } else if (line % 3 == 1) {
            EXPECT_TRUE(std::isinf(hostMinValues[line]));
            EXPECT_GT(hostMinValues[line], 0.f);
            EXPECT_EQ(0u, hostMinIndices[line]);
            EXPECT_TRUE(std::isinf(hostMaxValues[line]));
            EXPECT_LT(hostMaxValues[line], 0.f);
            EXPECT_EQ(0u, hostMaxIndices[line]);
        } else {
            EXPECT_EQ(-1.f, hostMinValues[line]);
            EXPECT_EQ(5u, hostMinIndices[line]);
            EXPECT_EQ(1.f, hostMaxValues[line]);
            EXPECT_EQ(5u, hostMaxIndices[line]);
        }
    }
}

TEST(IReduceDimParallel, ComplexKeysPreserveTiesAndRawNaNEvents) {
    SKIP_IF_FAST_MATH_ENABLED();
    const int dim               = 0;
    const dim_t reducedElements = dimensionalDims[dim];
    const size_t lineCount      = static_cast<size_t>(
        dimensionalDims.elements() / reducedElements);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const cfloat ordinaryNaN(nan, 0.f);
    const cfloat rawEvent(inf, nan);
    vector<cfloat> minInput(dimensionalDims.elements(), cfloat(3.f, 0.f));
    vector<cfloat> maxInput(dimensionalDims.elements(), cfloat(.25f, 0.f));

    for (size_t line = 0; line < lineCount; ++line) {
        minInput[lineElementIndex(dimensionalDims, dim, line, 1)] =
            cfloat(1.f, 0.f);
        minInput[lineElementIndex(dimensionalDims, dim, line,
                                  reducedElements - 1)] = cfloat(0.f, -1.f);
        maxInput[lineElementIndex(dimensionalDims, dim, line, 1)] =
            cfloat(4.f, 0.f);
        maxInput[lineElementIndex(dimensionalDims, dim, line,
                                  reducedElements - 1)] = cfloat(0.f, 4.f);
    }

    for (size_t line = 0; line < 2; ++line) {
        for (dim_t reduced = 0; reduced < reducedElements; ++reduced) {
            minInput[lineElementIndex(dimensionalDims, dim, line, reduced)] =
                ordinaryNaN;
            maxInput[lineElementIndex(dimensionalDims, dim, line, reduced)] =
                ordinaryNaN;
        }
    }
    minInput[lineElementIndex(dimensionalDims, dim, 0, 0)] = rawEvent;
    maxInput[lineElementIndex(dimensionalDims, dim, 0, 0)] = rawEvent;
    minInput[lineElementIndex(dimensionalDims, dim, 1, 7)] = rawEvent;
    maxInput[lineElementIndex(dimensionalDims, dim, 1, 7)] = rawEvent;

    array minValues;
    array minIndices;
    array maxValues;
    array maxIndices;
    af::min(minValues, minIndices,
            array(dimensionalDims, minInput.data()), dim);
    af::max(maxValues, maxIndices,
            array(dimensionalDims, maxInput.data()), dim);

    const vector<cfloat> hostMinValues = hostVector<cfloat>(minValues);
    const vector<unsigned> hostMinIndices =
        hostVector<unsigned>(minIndices);
    const vector<cfloat> hostMaxValues = hostVector<cfloat>(maxValues);
    const vector<unsigned> hostMaxIndices =
        hostVector<unsigned>(maxIndices);
    EXPECT_TRUE(std::isinf(af::real(hostMinValues[0])));
    EXPECT_GT(af::real(hostMinValues[0]), 0.f);
    EXPECT_EQ(0.f, af::imag(hostMinValues[0]));
    EXPECT_EQ(0u, hostMinIndices[0]);
    expectRawInfNaN(hostMaxValues[0]);
    EXPECT_EQ(0u, hostMaxIndices[0]);
    expectRawInfNaN(hostMinValues[1]);
    EXPECT_EQ(7u, hostMinIndices[1]);
    expectRawInfNaN(hostMaxValues[1]);
    EXPECT_EQ(7u, hostMaxIndices[1]);

    for (size_t line = 2; line < lineCount; ++line) {
        expectComplexEqual(cfloat(0.f, -1.f), hostMinValues[line]);
        EXPECT_EQ(static_cast<unsigned>(reducedElements - 1),
                  hostMinIndices[line]);
        expectComplexEqual(cfloat(4.f, 0.f), hostMaxValues[line]);
        EXPECT_EQ(1u, hostMaxIndices[line]);
    }
}

TEST(IReduceDimParallel, TypeSpecificKeysPreserveRawPayloads) {
    const dim4 dims(256, 512);
    const dim_t reducedElements = dims[0];
    const size_t lineCount = static_cast<size_t>(dims[1]);
    const unsigned long long twoTo53 = 9007199254740992ULL;
    vector<unsigned long long> minU64(dims.elements(), twoTo53 + 1024);
    vector<unsigned long long> maxU64(dims.elements(), 0);
    vector<char> booleanValues(dims.elements(), 1);
    vector<half_float::half> halfValues(dims.elements(),
                                        half_float::half(0.f));

    for (size_t line = 0; line < lineCount; ++line) {
        minU64[lineElementIndex(dims, 0, line, 7)] = twoTo53;
        minU64[lineElementIndex(dims, 0, line,
                                reducedElements - 1)] = twoTo53 + 1;
        maxU64[lineElementIndex(dims, 0, line, 7)] = twoTo53;
        maxU64[lineElementIndex(dims, 0, line,
                                reducedElements - 1)] = twoTo53 + 1;
        booleanValues[lineElementIndex(dims, 0, line, 0)] = 2;
        booleanValues[lineElementIndex(dims, 0, line,
                                       reducedElements - 1)] = 3;
        halfValues[lineElementIndex(dims, 0, line, 0)] =
            half_float::half(9.f);
        halfValues[lineElementIndex(dims, 0, line,
                                    reducedElements - 2)] =
            half_float::half(9.f);
        halfValues[lineElementIndex(dims, 0, line, 1)] =
            half_float::half(-7.f);
        halfValues[lineElementIndex(dims, 0, line,
                                    reducedElements - 1)] =
            half_float::half(-7.f);
    }

    array minValues;
    array minIndices;
    array maxValues;
    array maxIndices;
    af::min(minValues, minIndices, array(dims, minU64.data()), 0);
    af::max(maxValues, maxIndices, array(dims, maxU64.data()), 0);
    const vector<unsigned long long> hostMinU64 =
        hostVector<unsigned long long>(minValues);
    const vector<unsigned> hostMinU64Indices =
        hostVector<unsigned>(minIndices);
    const vector<unsigned long long> hostMaxU64 =
        hostVector<unsigned long long>(maxValues);
    const vector<unsigned> hostMaxU64Indices =
        hostVector<unsigned>(maxIndices);
    for (size_t line = 0; line < lineCount; ++line) {
        EXPECT_EQ(twoTo53 + 1, hostMinU64[line]);
        EXPECT_EQ(static_cast<unsigned>(reducedElements - 1),
                  hostMinU64Indices[line]);
        EXPECT_EQ(twoTo53, hostMaxU64[line]);
        EXPECT_EQ(7u, hostMaxU64Indices[line]);
    }

    af::min(minValues, minIndices, array(dims, booleanValues.data()), 0);
    af::max(maxValues, maxIndices, array(dims, booleanValues.data()), 0);
    const vector<char> hostMinBoolean = hostVector<char>(minValues);
    const vector<unsigned> hostMinBooleanIndices =
        hostVector<unsigned>(minIndices);
    const vector<char> hostMaxBoolean = hostVector<char>(maxValues);
    const vector<unsigned> hostMaxBooleanIndices =
        hostVector<unsigned>(maxIndices);
    for (size_t line = 0; line < lineCount; ++line) {
        EXPECT_EQ(3, hostMinBoolean[line]);
        EXPECT_EQ(static_cast<unsigned>(reducedElements - 1),
                  hostMinBooleanIndices[line]);
        EXPECT_EQ(2, hostMaxBoolean[line]);
        EXPECT_EQ(0u, hostMaxBooleanIndices[line]);
    }

    af::min(minValues, minIndices, array(dims, halfValues.data()), 0);
    af::max(maxValues, maxIndices, array(dims, halfValues.data()), 0);
    const vector<half_float::half> hostMinHalf =
        hostVector<half_float::half>(minValues);
    const vector<unsigned> hostMinHalfIndices =
        hostVector<unsigned>(minIndices);
    const vector<half_float::half> hostMaxHalf =
        hostVector<half_float::half>(maxValues);
    const vector<unsigned> hostMaxHalfIndices =
        hostVector<unsigned>(maxIndices);
    for (size_t line = 0; line < lineCount; ++line) {
        EXPECT_EQ(-7.f, static_cast<float>(hostMinHalf[line]));
        EXPECT_EQ(static_cast<unsigned>(reducedElements - 1),
                  hostMinHalfIndices[line]);
        EXPECT_EQ(9.f, static_cast<float>(hostMaxHalf[line]));
        EXPECT_EQ(0u, hostMaxHalfIndices[line]);
    }
}

TEST(IReduceDimParallel, EvaluatesLargeLazyInputBeforeWorkerReads) {
    const int dim = 2;
    const array lazy = af::range(dimensionalDims, dim) * -2.f + 5.f;
    array minValues;
    array minIndices;
    array maxValues;
    array maxIndices;
    af::min(minValues, minIndices, lazy, dim);
    af::max(maxValues, maxIndices, lazy, dim);

    const vector<float> hostMinValues = hostVector<float>(minValues);
    const vector<unsigned> hostMinIndices =
        hostVector<unsigned>(minIndices);
    const vector<float> hostMaxValues = hostVector<float>(maxValues);
    const vector<unsigned> hostMaxIndices =
        hostVector<unsigned>(maxIndices);
    const float expectedMin =
        5.f - 2.f * static_cast<float>(dimensionalDims[dim] - 1);
    for (size_t line = 0; line < hostMinValues.size(); ++line) {
        EXPECT_EQ(expectedMin, hostMinValues[line]);
        EXPECT_EQ(static_cast<unsigned>(dimensionalDims[dim] - 1),
                  hostMinIndices[line]);
        EXPECT_EQ(5.f, hostMaxValues[line]);
        EXPECT_EQ(0u, hostMaxIndices[line]);
    }
}

TEST(IReduceRaggedParallel, StridedLengthsUseLogicalCoordinates) {
    const float inputValues[] = {1.f, 2.f, 3.f, 4.f, 10.f, 20.f,
                                 30.f, 40.f, 5.f, 6.f, 7.f, 8.f};
    const unsigned parentLengths[] = {1, 99, 2, 99, 3, 99};
    const array input(dim4(4, 3), inputValues);
    const array lengthParent(dim4(2, 3), parentLengths);
    const array lengths = lengthParent(0, span);
    ASSERT_EQ(1, lengths.dims(0));
    ASSERT_EQ(3, lengths.dims(1));
    ASSERT_EQ(2, af::getStrides(lengths)[1]);

    array values;
    array indices;
    af::max(values, indices, input, lengths, 0);

    const vector<float> expectedValues{1.f, 20.f, 7.f};
    const vector<unsigned> expectedIndices{0, 1, 2};
    EXPECT_VEC_ARRAY_EQ(expectedValues, dim4(1, 3), values);
    EXPECT_VEC_ARRAY_EQ(expectedIndices, dim4(1, 3), indices);
}

TEST(IReduceRaggedParallel,
     MixedLengthsAndGappedViewsAcrossAllDimensions) {
    const float negInf = -std::numeric_limits<float>::infinity();
    for (int dim = 0; dim < 4; ++dim) {
        const dim_t reducedElements = dimensionalDims[dim];
        dim4 outputDims             = dimensionalDims;
        outputDims[dim]             = 1;
        const size_t lineCount = static_cast<size_t>(outputDims.elements());
        vector<float> inputValues(dimensionalDims.elements());
        vector<unsigned> lengths(lineCount);

        for (size_t line = 0; line < lineCount; ++line) {
            for (dim_t reduced = 0; reduced < reducedElements; ++reduced) {
                inputValues[lineElementIndex(dimensionalDims, dim, line,
                                             reduced)] =
                    static_cast<float>(reduced);
            }

            switch (line % 5) {
                case 0:
                    lengths[line] = 0;
                    inputValues[lineElementIndex(dimensionalDims, dim, line,
                                                 0)] = negInf;
                    break;
                case 1: lengths[line] = 1; break;
                case 2:
                    lengths[line] =
                        static_cast<unsigned>(reducedElements / 2);
                    break;
                case 3:
                    lengths[line] = static_cast<unsigned>(reducedElements);
                    break;
                default:
                    lengths[line] =
                        static_cast<unsigned>(reducedElements + 17);
                    break;
            }
        }

        const array input = makeGappedView(
            array(dimensionalDims, inputValues.data()));
        const array raggedLengths =
            makeGappedView(array(outputDims, lengths.data()));
        ASSERT_GT(af::getOffset(input), 0);
        ASSERT_GT(af::getOffset(raggedLengths), 0);

        array values;
        array indices;
        af::max(values, indices, input, raggedLengths, dim);
        const vector<float> hostValues = hostVector<float>(values);
        const vector<unsigned> hostIndices = hostVector<unsigned>(indices);

        for (size_t line = 0; line < lineCount; ++line) {
            const dim_t length = std::min(
                reducedElements, static_cast<dim_t>(lengths[line]));
            if (length == 0) {
                EXPECT_EQ(negInf, hostValues[line]);
                EXPECT_EQ(0u, hostIndices[line]);
            } else {
                EXPECT_EQ(static_cast<float>(length - 1), hostValues[line]);
                EXPECT_EQ(static_cast<unsigned>(length - 1),
                          hostIndices[line]);
            }
        }
    }
}

TEST(IReduceRaggedParallel, TruncatedRealPrefixesPreserveScalarSemantics) {
    SKIP_IF_FAST_MATH_ENABLED();
    const dim4 dims(256, 2048);
    const dim_t reducedElements = dims[0];
    const size_t lineCount      = static_cast<size_t>(dims[1]);
    const unsigned prefixLength = 192;
    const float nan             = std::numeric_limits<float>::quiet_NaN();
    const float negInf          = -std::numeric_limits<float>::infinity();
    vector<float> inputValues(dims.elements(), -2.f);
    vector<unsigned> lengths(lineCount, prefixLength);

    for (size_t line = 0; line < lineCount; ++line) {
        inputValues[lineElementIndex(dims, 0, line,
                                     reducedElements - 1)] = 100.f;
        switch (line % 5) {
            case 0:
                inputValues[lineElementIndex(dims, 0, line, 1)] = -0.f;
                inputValues[lineElementIndex(dims, 0, line, 7)] = 0.f;
                break;
            case 1:
                for (unsigned reduced = 0; reduced < prefixLength;
                     ++reduced) {
                    inputValues[lineElementIndex(dims, 0, line, reduced)] =
                        nan;
                }
                break;
            case 2:
                inputValues[lineElementIndex(dims, 0, line, 0)] = nan;
                inputValues[lineElementIndex(dims, 0, line, 3)] = 5.f;
                inputValues[lineElementIndex(dims, 0, line, 10)] = 5.f;
                break;
            case 3:
                inputValues[lineElementIndex(dims, 0, line, 0)] = nan;
                for (unsigned reduced = 1; reduced < prefixLength;
                     ++reduced) {
                    inputValues[lineElementIndex(dims, 0, line, reduced)] =
                        negInf;
                }
                break;
            default:
                lengths[line] = 0;
                inputValues[lineElementIndex(dims, 0, line, 0)] = negInf;
                break;
        }
    }

    array values;
    array indices;
    af::max(values, indices, array(dims, inputValues.data()),
            array(dim4(1, static_cast<dim_t>(lineCount)), lengths.data()), 0);
    const vector<float> hostValues = hostVector<float>(values);
    const vector<unsigned> hostIndices = hostVector<unsigned>(indices);

    for (size_t line = 0; line < lineCount; ++line) {
        switch (line % 5) {
            case 0:
                EXPECT_EQ(0.f, hostValues[line]);
                EXPECT_TRUE(std::signbit(hostValues[line]));
                EXPECT_EQ(1u, hostIndices[line]);
                break;
            case 1:
            case 3:
            case 4:
                EXPECT_EQ(negInf, hostValues[line]);
                EXPECT_EQ(0u, hostIndices[line]);
                break;
            default:
                EXPECT_EQ(5.f, hostValues[line]);
                EXPECT_EQ(3u, hostIndices[line]);
                break;
        }
    }
}

TEST(IReduceRaggedParallel, ComplexPrefixesPreserveMagnitudeAndNaNEvents) {
    SKIP_IF_FAST_MATH_ENABLED();
    const dim4 dims(256, 2048);
    const dim_t reducedElements = dims[0];
    const size_t lineCount      = static_cast<size_t>(dims[1]);
    const unsigned prefixLength = 192;
    const float nan             = std::numeric_limits<float>::quiet_NaN();
    const float inf             = std::numeric_limits<float>::infinity();
    const cfloat ordinaryNaN(nan, 0.f);
    const cfloat rawEvent(inf, nan);
    vector<cfloat> inputValues(dims.elements(), cfloat(.25f, 0.f));
    vector<unsigned> lengths(lineCount, prefixLength);

    for (size_t line = 0; line < lineCount; ++line) {
        inputValues[lineElementIndex(dims, 0, line, 1)] = cfloat(4.f, 0.f);
        inputValues[lineElementIndex(dims, 0, line, 7)] = cfloat(0.f, 4.f);
        inputValues[lineElementIndex(dims, 0, line,
                                     reducedElements - 1)] =
            cfloat(100.f, 0.f);
    }
    for (size_t line = 0; line < 3; ++line) {
        for (unsigned reduced = 0; reduced < prefixLength; ++reduced) {
            inputValues[lineElementIndex(dims, 0, line, reduced)] =
                ordinaryNaN;
        }
    }
    inputValues[lineElementIndex(dims, 0, 0, 0)] = rawEvent;
    inputValues[lineElementIndex(dims, 0, 1, 7)] = rawEvent;

    array values;
    array indices;
    af::max(values, indices, array(dims, inputValues.data()),
            array(dim4(1, static_cast<dim_t>(lineCount)), lengths.data()), 0);
    const vector<cfloat> hostValues = hostVector<cfloat>(values);
    const vector<unsigned> hostIndices = hostVector<unsigned>(indices);

    expectRawInfNaN(hostValues[0]);
    EXPECT_EQ(0u, hostIndices[0]);
    expectRawInfNaN(hostValues[1]);
    EXPECT_EQ(7u, hostIndices[1]);
    expectComplexEqual(cfloat(0.f, 0.f), hostValues[2]);
    EXPECT_EQ(0u, hostIndices[2]);

    for (size_t line = 3; line < lineCount; ++line) {
        expectComplexEqual(cfloat(4.f, 0.f), hostValues[line]);
        EXPECT_EQ(1u, hostIndices[line]);
    }
}

TEST(IReduceRaggedParallel, EvaluatesLazyInputAndVariableLengths) {
    const int dim               = 2;
    const dim_t reducedElements = dimensionalDims[dim];
    dim4 outputDims             = dimensionalDims;
    outputDims[dim]             = 1;
    const array lazyInput = af::range(dimensionalDims, dim) * 2.f - 3.f;
    const array lazyLengths =
        (af::range(outputDims, 0, af::dtype::u32) %
         static_cast<unsigned>(reducedElements)) +
        1;

    array values;
    array indices;
    af::max(values, indices, lazyInput, lazyLengths, dim);
    const vector<float> hostValues = hostVector<float>(values);
    const vector<unsigned> hostIndices = hostVector<unsigned>(indices);

    for (size_t line = 0; line < hostValues.size(); ++line) {
        const unsigned length =
            static_cast<unsigned>(line % outputDims[0]) %
                static_cast<unsigned>(reducedElements) +
            1;
        EXPECT_EQ(2.f * static_cast<float>(length - 1) - 3.f,
                  hostValues[line]);
        EXPECT_EQ(length - 1, hostIndices[line]);
    }
}
