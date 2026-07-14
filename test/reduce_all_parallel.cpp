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

constexpr size_t blockElements = 1 << 16;

size_t linearIndex(const dim4 &dims, const dim_t x, const dim_t y,
                   const dim_t z, const dim_t w) {
    return static_cast<size_t>(
        x + dims[0] * (y + dims[1] * (z + dims[2] * w)));
}

void expectComplexEqual(const cfloat expected, const cfloat actual) {
    EXPECT_FLOAT_EQ(real(expected), real(actual));
    EXPECT_FLOAT_EQ(imag(expected), imag(actual));
}

}  // namespace

TEST(ReduceAllParallel, TruthAndCountAtPartitionBoundaries) {
    const size_t sizes[] = {blockElements - 1, blockElements,
                            2 * blockElements - 1, 2 * blockElements,
                            9 * blockElements + 17};

    for (const size_t elements : sizes) {
        SCOPED_TRACE(::testing::Message() << "elements " << elements);
        vector<int> values(elements, 1);
        values[elements / 2] = 0;
        const array input(static_cast<dim_t>(elements), values.data());

        EXPECT_EQ(static_cast<unsigned>(elements - 1),
                  af::count<unsigned>(input));
        EXPECT_TRUE(af::anyTrue<bool>(input));
        EXPECT_FALSE(af::allTrue<bool>(input));
    }

    const size_t elements = 9 * blockElements + 17;
    vector<int> values(elements, 0);
    array input(static_cast<dim_t>(elements), values.data());
    EXPECT_EQ(0u, af::count<unsigned>(input));
    EXPECT_FALSE(af::anyTrue<bool>(input));
    EXPECT_FALSE(af::allTrue<bool>(input));

    values.back() = 1;
    input         = array(static_cast<dim_t>(elements), values.data());
    EXPECT_EQ(1u, af::count<unsigned>(input));
    EXPECT_TRUE(af::anyTrue<bool>(input));
    EXPECT_FALSE(af::allTrue<bool>(input));

    std::fill(values.begin(), values.end(), 1);
    input = array(static_cast<dim_t>(elements), values.data());
    EXPECT_EQ(static_cast<unsigned>(elements), af::count<unsigned>(input));
    EXPECT_TRUE(af::anyTrue<bool>(input));
    EXPECT_TRUE(af::allTrue<bool>(input));

    EXPECT_EQ(static_cast<unsigned>(elements),
              af::count<array>(input).scalar<unsigned>());
    EXPECT_EQ(1, af::anyTrue<array>(input).scalar<char>());
    EXPECT_EQ(1, af::allTrue<array>(input).scalar<char>());
}

TEST(ReduceAllParallel, NaNsRetainExistingTruthiness) {
    SKIP_IF_FAST_MATH_ENABLED();
    const size_t elements = 2 * blockElements + 13;
    vector<float> realValues(elements, 0.f);
    realValues[blockElements - 1] = -0.f;
    realValues[blockElements] = std::numeric_limits<float>::quiet_NaN();
    const array realInput(static_cast<dim_t>(elements), realValues.data());

    EXPECT_EQ(1u, af::count<unsigned>(realInput));
    EXPECT_TRUE(af::anyTrue<bool>(realInput));
    EXPECT_FALSE(af::allTrue<bool>(realInput));

    vector<cfloat> complexValues(elements, cfloat(0.f, 0.f));
    complexValues[blockElements - 1] =
        cfloat(std::numeric_limits<float>::quiet_NaN(), 0.f);
    complexValues[blockElements] =
        cfloat(0.f, std::numeric_limits<float>::quiet_NaN());
    const array complexInput(static_cast<dim_t>(elements),
                             complexValues.data());

    EXPECT_EQ(2u, af::count<unsigned>(complexInput));
    EXPECT_TRUE(af::anyTrue<bool>(complexInput));
    EXPECT_FALSE(af::allTrue<bool>(complexInput));
}

TEST(ReduceAllParallel, GappedFourDimensionalView) {
    const dim4 parentDims(259, 69, 5, 13);
    vector<int> values(parentDims.elements(), 1);
    values[linearIndex(parentDims, 128, 34, 2, 7)] = 0;
    const array parent(parentDims, values.data());
    const array view = parent(span, seq(1, 67), span, span);
    const unsigned elements = static_cast<unsigned>(view.elements());

    EXPECT_EQ(elements - 1, af::count<unsigned>(view));
    EXPECT_TRUE(af::anyTrue<bool>(view));
    EXPECT_FALSE(af::allTrue<bool>(view));
    EXPECT_EQ(0, af::min<int>(view));
    EXPECT_EQ(1, af::max<int>(view));

    EXPECT_EQ(elements - 1, af::count<array>(view).scalar<unsigned>());
    EXPECT_EQ(0, af::min<array>(view).scalar<int>());
    EXPECT_EQ(1, af::max<array>(view).scalar<int>());
}

TEST(ReduceAllParallel, RealMinMaxPreserveOrderedSignedZeroTies) {
    SKIP_IF_FAST_MATH_ENABLED();
    const size_t elements = 9 * blockElements;

    vector<float> minValues(elements, 2.f);
    minValues.front() = 0.f;
    minValues.back()  = -0.f;
    const array minInput(static_cast<dim_t>(elements), minValues.data());
    const float minScalar = af::min<float>(minInput);
    const float minArray  = af::min<array>(minInput).scalar<float>();
    EXPECT_EQ(0.f, minScalar);
    EXPECT_EQ(0.f, minArray);
    EXPECT_TRUE(std::signbit(minScalar));
    EXPECT_TRUE(std::signbit(minArray));

    vector<float> maxValues(elements, -2.f);
    maxValues.front() = 0.f;
    maxValues.back()  = -0.f;
    const array maxInput(static_cast<dim_t>(elements), maxValues.data());
    const float maxScalar = af::max<float>(maxInput);
    const float maxArray  = af::max<array>(maxInput).scalar<float>();
    EXPECT_EQ(0.f, maxScalar);
    EXPECT_EQ(0.f, maxArray);
    EXPECT_TRUE(std::signbit(maxScalar));
    EXPECT_TRUE(std::signbit(maxArray));
}

TEST(ReduceAllParallel, ComplexMinMaxPreserveFirstEqualMagnitude) {
    const size_t elements = 9 * blockElements;

    vector<cfloat> minValues(elements, cfloat(3.f, 0.f));
    const cfloat firstMin(1.f, 0.f);
    minValues.front() = firstMin;
    minValues.back()  = cfloat(0.f, 1.f);
    const array minInput(static_cast<dim_t>(elements), minValues.data());
    expectComplexEqual(firstMin, af::min<cfloat>(minInput));
    expectComplexEqual(firstMin,
                       af::min<array>(minInput).scalar<cfloat>());

    vector<cfloat> maxValues(elements, cfloat(0.25f, 0.f));
    const cfloat firstMax(4.f, 0.f);
    maxValues.front() = firstMax;
    maxValues.back()  = cfloat(0.f, 4.f);
    const array maxInput(static_cast<dim_t>(elements), maxValues.data());
    expectComplexEqual(firstMax, af::max<cfloat>(maxInput));
    expectComplexEqual(firstMax,
                       af::max<array>(maxInput).scalar<cfloat>());
}

TEST(ReduceAllParallel, AllNaNMinMaxIdentities) {
    SKIP_IF_FAST_MATH_ENABLED();
    const size_t elements = 2 * blockElements;
    vector<float> values(elements,
                         std::numeric_limits<float>::quiet_NaN());
    const array input(static_cast<dim_t>(elements), values.data());

    const float minScalar = af::min<float>(input);
    const float maxScalar = af::max<float>(input);
    EXPECT_TRUE(std::isinf(minScalar));
    EXPECT_GT(minScalar, 0.f);
    EXPECT_TRUE(std::isinf(maxScalar));
    EXPECT_LT(maxScalar, 0.f);
    EXPECT_EQ(minScalar, af::min<array>(input).scalar<float>());
    EXPECT_EQ(maxScalar, af::max<array>(input).scalar<float>());
}

TEST(ReduceAllParallel, NaNOnlyPartitionsDoNotMaskValidExtrema) {
    SKIP_IF_FAST_MATH_ENABLED();
    const size_t elements = 9 * blockElements + 13;
    const float nan       = std::numeric_limits<float>::quiet_NaN();

    vector<float> realValues(elements, nan);
    realValues[elements / 2] = -4.f;
    const array realInput(static_cast<dim_t>(elements), realValues.data());
    EXPECT_EQ(-4.f, af::min<float>(realInput));
    EXPECT_EQ(-4.f, af::max<float>(realInput));

    vector<cfloat> complexValues(elements, cfloat(nan, nan));
    const cfloat complexMin(1.f, 0.f);
    const cfloat complexMax(0.f, 2.f);
    complexValues[elements / 2]     = complexMin;
    complexValues[elements / 2 + 1] = complexMax;
    const array complexInput(static_cast<dim_t>(elements),
                             complexValues.data());
    expectComplexEqual(complexMin, af::min<cfloat>(complexInput));
    expectComplexEqual(complexMax, af::max<cfloat>(complexInput));
}
