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
#include <af/internal.h>

#include <algorithm>
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

size_t linearIndex(const dim4 &dims, const dim_t x, const dim_t y,
                   const dim_t z, const dim_t w) {
    return static_cast<size_t>(
        x + dims[0] * (y + dims[1] * (z + dims[2] * w)));
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
