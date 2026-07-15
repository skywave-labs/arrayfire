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
#include <cstring>
#include <limits>
#include <vector>

using af::array;
using af::cdouble;
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

namespace {

constexpr dim_t dimensionalReduceLength = 257;

dim4 dimensionalReduceDims(const dim_t width, const int dim) {
    dim4 dims(width, 3, 3, 3);
    dims[dim] = dimensionalReduceLength;
    if (dim == 1) {
        dims[3] = 7;
    } else if (dim == 2) {
        dims[3] = 7;
    } else {
        dims[2] = 7;
    }
    return dims;
}

template<typename T>
T realSumPattern(const size_t line, const dim_t reducedIndex) {
    switch (line % 5) {
        case 0: {
            const T large = std::ldexp(T(1), std::numeric_limits<T>::digits);
            const T pattern[] = {large, T(1), -large, T(1)};
            return pattern[reducedIndex % 4];
        }
        case 1: return -T(0);
        case 2:
            if (reducedIndex < 2) { return std::numeric_limits<T>::max(); }
            return reducedIndex == 2 ? -std::numeric_limits<T>::max() : T(0);
        case 3:
            return (reducedIndex & 1) ? -std::numeric_limits<T>::denorm_min()
                                      : std::numeric_limits<T>::denorm_min();
        default:
            return static_cast<T>(static_cast<int>((line + reducedIndex) % 7) -
                                  3);
    }
}

template<typename T>
T sumPattern(const size_t line, const dim_t reducedIndex) {
    return realSumPattern<T>(line, reducedIndex);
}

template<>
cfloat sumPattern<cfloat>(const size_t line, const dim_t reducedIndex) {
    return cfloat(realSumPattern<float>(line, reducedIndex),
                  realSumPattern<float>(line + 2, reducedIndex));
}

template<>
cdouble sumPattern<cdouble>(const size_t line, const dim_t reducedIndex) {
    return cdouble(realSumPattern<double>(line, reducedIndex),
                   realSumPattern<double>(line + 2, reducedIndex));
}

template<typename T>
T hostAdd(const T lhs, const T rhs) {
    return lhs + rhs;
}

template<>
cfloat hostAdd<cfloat>(const cfloat lhs, const cfloat rhs) {
    return cfloat(lhs.real + rhs.real, lhs.imag + rhs.imag);
}

template<>
cdouble hostAdd<cdouble>(const cdouble lhs, const cdouble rhs) {
    return cdouble(lhs.real + rhs.real, lhs.imag + rhs.imag);
}

template<typename T>
T productPattern(const size_t line, const dim_t reducedIndex) {
    switch (line % 5) {
        case 0: {
            const T pattern[] = {T(-1), T(2), T(0.5), T(-1)};
            return pattern[reducedIndex % 4];
        }
        case 1: return reducedIndex == 0 ? -T(0) : T(1);
        case 2:
            if (reducedIndex == 0) { return std::numeric_limits<T>::max(); }
            if (reducedIndex == 1) { return T(2); }
            return reducedIndex == 2 ? T(0.5) : T(1);
        case 3:
            if (reducedIndex == 0) {
                return std::numeric_limits<T>::denorm_min();
            }
            if (reducedIndex == 1) { return T(0.5); }
            return reducedIndex == 2 ? T(2) : T(1);
        default: return ((line + reducedIndex) & 1) ? T(-1) : T(1);
    }
}

template<typename T>
void expectBitwiseEqual(const vector<T> &expected, const vector<T> &actual) {
    ASSERT_EQ(expected.size(), actual.size());
    ASSERT_EQ(0, std::memcmp(expected.data(), actual.data(),
                             expected.size() * sizeof(T)));
}

template<typename T>
void checkDimensionalSum(const dim_t width, const int dim) {
    const dim4 dims = dimensionalReduceDims(width, dim);
    dim4 outputDims = dims;
    outputDims[dim] = 1;
    vector<T> input(static_cast<size_t>(dims.elements()));
    vector<T> expected(static_cast<size_t>(outputDims.elements()));

    for (dim_t w = 0; w < outputDims[3]; ++w) {
        for (dim_t z = 0; z < outputDims[2]; ++z) {
            for (dim_t y = 0; y < outputDims[1]; ++y) {
                for (dim_t x = 0; x < outputDims[0]; ++x) {
                    dim_t coords[] = {x, y, z, w};
                    const size_t outputIndex =
                        linearIndex(outputDims, x, y, z, w);
                    T value = T(0);
                    for (dim_t r = 0; r < dims[dim]; ++r) {
                        coords[dim]        = r;
                        const T inputValue = sumPattern<T>(outputIndex, r);
                        input[linearIndex(dims, coords[0], coords[1], coords[2],
                                          coords[3])] = inputValue;
                        value = hostAdd(inputValue, value);
                    }
                    expected[outputIndex] = value;
                }
            }
        }
    }

    const array output = af::sum(array(dims, input.data()), dim);
    vector<T> actual(expected.size());
    output.host(actual.data());
    expectBitwiseEqual(expected, actual);
}

template<typename T>
void checkDimensionalProduct(const dim_t width, const int dim) {
    const dim4 dims = dimensionalReduceDims(width, dim);
    dim4 outputDims = dims;
    outputDims[dim] = 1;
    vector<T> input(static_cast<size_t>(dims.elements()));
    vector<T> expected(static_cast<size_t>(outputDims.elements()));

    for (dim_t w = 0; w < outputDims[3]; ++w) {
        for (dim_t z = 0; z < outputDims[2]; ++z) {
            for (dim_t y = 0; y < outputDims[1]; ++y) {
                for (dim_t x = 0; x < outputDims[0]; ++x) {
                    dim_t coords[] = {x, y, z, w};
                    const size_t outputIndex =
                        linearIndex(outputDims, x, y, z, w);
                    T value = T(1);
                    for (dim_t r = 0; r < dims[dim]; ++r) {
                        coords[dim]        = r;
                        const T inputValue = productPattern<T>(outputIndex, r);
                        input[linearIndex(dims, coords[0], coords[1], coords[2],
                                          coords[3])] = inputValue;
                        value                         = inputValue * value;
                    }
                    expected[outputIndex] = value;
                }
            }
        }
    }

    const array output = af::product(array(dims, input.data()), dim);
    vector<T> actual(expected.size());
    output.host(actual.data());
    expectBitwiseEqual(expected, actual);
}

template<typename T>
T extremaPattern(const size_t line, const dim_t reducedIndex) {
    const T infinity = std::numeric_limits<T>::infinity();
    const T nan      = std::numeric_limits<T>::quiet_NaN();
    switch (line % 7) {
        case 0:
            if (reducedIndex == dimensionalReduceLength - 1) {
                return (line & 1) ? T(0) : -T(0);
            }
            return (line & 1) ? -T(0) : T(0);
        case 1:
            if (reducedIndex == 0 || reducedIndex == 31 ||
                reducedIndex == dimensionalReduceLength - 1) {
                return nan;
            }
            return static_cast<T>(static_cast<int>((line + reducedIndex) % 13) -
                                  6);
        case 2:
            if (reducedIndex == 3) { return -infinity; }
            if (reducedIndex == 201) { return infinity; }
            return static_cast<T>(static_cast<int>(reducedIndex % 9) - 4);
        case 3: return T(7.25);
        case 4:
            return static_cast<T>(static_cast<int>((3 * line + reducedIndex) %
                                                   23) -
                                  11) /
                   T(8);
        case 5: return nan;
        default:
            return (reducedIndex & 1) ? -std::numeric_limits<T>::denorm_min()
                                      : std::numeric_limits<T>::denorm_min();
    }
}

template<typename T>
void checkDimensionalExtrema(const dim_t width, const int dim) {
    const dim4 dims = dimensionalReduceDims(width, dim);
    dim4 outputDims = dims;
    outputDims[dim] = 1;
    vector<T> input(static_cast<size_t>(dims.elements()));
    vector<T> expectedMin(static_cast<size_t>(outputDims.elements()));
    vector<T> expectedMax(static_cast<size_t>(outputDims.elements()));

    for (dim_t w = 0; w < outputDims[3]; ++w) {
        for (dim_t z = 0; z < outputDims[2]; ++z) {
            for (dim_t y = 0; y < outputDims[1]; ++y) {
                for (dim_t x = 0; x < outputDims[0]; ++x) {
                    dim_t coords[] = {x, y, z, w};
                    const size_t outputIndex =
                        linearIndex(outputDims, x, y, z, w);
                    T minValue = std::numeric_limits<T>::infinity();
                    T maxValue = -std::numeric_limits<T>::infinity();
                    for (dim_t r = 0; r < dims[dim]; ++r) {
                        coords[dim]        = r;
                        const T inputValue = extremaPattern<T>(outputIndex, r);
                        input[linearIndex(dims, coords[0], coords[1], coords[2],
                                          coords[3])] = inputValue;
                        const T minInput =
                            std::isnan(inputValue)
                                ? std::numeric_limits<T>::infinity()
                                : inputValue;
                        const T maxInput =
                            std::isnan(inputValue)
                                ? -std::numeric_limits<T>::infinity()
                                : inputValue;
                        minValue = std::min(minInput, minValue);
                        maxValue = std::max(maxInput, maxValue);
                    }
                    expectedMin[outputIndex] = minValue;
                    expectedMax[outputIndex] = maxValue;
                }
            }
        }
    }

    const array inputArray(dims, input.data());
    const array minOutput = af::min(inputArray, dim);
    const array maxOutput = af::max(inputArray, dim);
    vector<T> actualMin(expectedMin.size());
    vector<T> actualMax(expectedMax.size());
    minOutput.host(actualMin.data());
    maxOutput.host(actualMax.data());
    expectBitwiseEqual(expectedMin, actualMin);
    expectBitwiseEqual(expectedMax, actualMax);
}

void checkGappedDimensionalSum(const int dim) {
    const dim4 dims = dimensionalReduceDims(35, dim);
    dim4 parentDims = dims;
    parentDims[dim] += 2;
    dim4 outputDims = dims;
    outputDims[dim] = 1;
    vector<float> input(static_cast<size_t>(parentDims.elements()), 91.f);
    vector<float> expected(static_cast<size_t>(outputDims.elements()));

    for (dim_t w = 0; w < outputDims[3]; ++w) {
        for (dim_t z = 0; z < outputDims[2]; ++z) {
            for (dim_t y = 0; y < outputDims[1]; ++y) {
                for (dim_t x = 0; x < outputDims[0]; ++x) {
                    dim_t coords[] = {x, y, z, w};
                    const size_t outputIndex =
                        linearIndex(outputDims, x, y, z, w);
                    float value = 0.f;
                    for (dim_t r = 0; r < dims[dim]; ++r) {
                        coords[dim] = r + 1;
                        const float inputValue =
                            sumPattern<float>(outputIndex, r);
                        input[linearIndex(parentDims, coords[0], coords[1],
                                          coords[2], coords[3])] = inputValue;
                        value = inputValue + value;
                    }
                    expected[outputIndex] = value;
                }
            }
        }
    }

    const array parent(parentDims, input.data());
    array view;
    if (dim == 1) {
        view = parent(span, seq(1, dimensionalReduceLength), span, span);
    } else if (dim == 2) {
        view = parent(span, span, seq(1, dimensionalReduceLength), span);
    } else {
        view = parent(span, span, span, seq(1, dimensionalReduceLength));
    }

    const array output = af::sum(view, dim);
    vector<float> actual(expected.size());
    output.host(actual.data());
    expectBitwiseEqual(expected, actual);
}

template<typename T>
void checkGappedDimensionalExtrema(const dim_t width, const int dim) {
    const dim4 dims = dimensionalReduceDims(width, dim);
    dim4 parentDims = dims;
    parentDims[dim] += 2;
    dim4 outputDims = dims;
    outputDims[dim] = 1;
    vector<T> input(static_cast<size_t>(parentDims.elements()), T(91));
    vector<T> expectedMin(static_cast<size_t>(outputDims.elements()));
    vector<T> expectedMax(static_cast<size_t>(outputDims.elements()));

    for (dim_t w = 0; w < outputDims[3]; ++w) {
        for (dim_t z = 0; z < outputDims[2]; ++z) {
            for (dim_t y = 0; y < outputDims[1]; ++y) {
                for (dim_t x = 0; x < outputDims[0]; ++x) {
                    dim_t coords[] = {x, y, z, w};
                    const size_t outputIndex =
                        linearIndex(outputDims, x, y, z, w);
                    T minValue = std::numeric_limits<T>::infinity();
                    T maxValue = -std::numeric_limits<T>::infinity();
                    for (dim_t r = 0; r < dims[dim]; ++r) {
                        coords[dim]        = r + 1;
                        const T inputValue = extremaPattern<T>(outputIndex, r);
                        input[linearIndex(parentDims, coords[0], coords[1],
                                          coords[2], coords[3])] = inputValue;
                        const T minInput =
                            std::isnan(inputValue)
                                ? std::numeric_limits<T>::infinity()
                                : inputValue;
                        const T maxInput =
                            std::isnan(inputValue)
                                ? -std::numeric_limits<T>::infinity()
                                : inputValue;
                        minValue = std::min(minInput, minValue);
                        maxValue = std::max(maxInput, maxValue);
                    }
                    expectedMin[outputIndex] = minValue;
                    expectedMax[outputIndex] = maxValue;
                }
            }
        }
    }

    const array parent(parentDims, input.data());
    array view;
    if (dim == 1) {
        view = parent(span, seq(1, dimensionalReduceLength), span, span);
    } else if (dim == 2) {
        view = parent(span, span, seq(1, dimensionalReduceLength), span);
    } else {
        view = parent(span, span, span, seq(1, dimensionalReduceLength));
    }

    const array minOutput = af::min(view, dim);
    const array maxOutput = af::max(view, dim);
    vector<T> actualMin(expectedMin.size());
    vector<T> actualMax(expectedMax.size());
    minOutput.host(actualMin.data());
    maxOutput.host(actualMax.data());
    expectBitwiseEqual(expectedMin, actualMin);
    expectBitwiseEqual(expectedMax, actualMax);
}

template<typename T>
void checkSumWidths(const dim_t *widths, const size_t count) {
    for (int dim = 1; dim <= 3; ++dim) {
        for (size_t i = 0; i < count; ++i) {
            SCOPED_TRACE(::testing::Message()
                         << "dimension " << dim << ", width " << widths[i]);
            checkDimensionalSum<T>(widths[i], dim);
        }
    }
}

template<typename T>
void checkProductWidths(const dim_t *widths, const size_t count) {
    for (int dim = 1; dim <= 3; ++dim) {
        for (size_t i = 0; i < count; ++i) {
            SCOPED_TRACE(::testing::Message()
                         << "dimension " << dim << ", width " << widths[i]);
            checkDimensionalProduct<T>(widths[i], dim);
        }
    }
}

template<typename T>
void checkExtremaWidths(const dim_t *widths, const size_t count) {
    for (int dim = 1; dim <= 3; ++dim) {
        for (size_t i = 0; i < count; ++i) {
            SCOPED_TRACE(::testing::Message()
                         << "dimension " << dim << ", width " << widths[i]);
            checkDimensionalExtrema<T>(widths[i], dim);
        }
    }
}

template<typename T>
T finiteNanTestValue(const size_t line, const dim_t reducedIndex) {
    return T(1) +
           static_cast<T>(static_cast<int>((line + reducedIndex) % 5) - 2) /
               T(16);
}

template<>
cfloat finiteNanTestValue<cfloat>(const size_t line, const dim_t reducedIndex) {
    return cfloat(finiteNanTestValue<float>(line, reducedIndex),
                  finiteNanTestValue<float>(line + 1, reducedIndex) - 1.f);
}

template<>
cdouble finiteNanTestValue<cdouble>(const size_t line,
                                    const dim_t reducedIndex) {
    return cdouble(finiteNanTestValue<double>(line, reducedIndex),
                   finiteNanTestValue<double>(line + 1, reducedIndex) - 1.0);
}

template<typename T>
T hostNan(const size_t) {
    return std::numeric_limits<T>::quiet_NaN();
}

template<>
cfloat hostNan<cfloat>(const size_t line) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    return (line & 1) ? cfloat(2.f, nan) : cfloat(nan, 2.f);
}

template<>
cdouble hostNan<cdouble>(const size_t line) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return (line & 1) ? cdouble(2.0, nan) : cdouble(nan, 2.0);
}

template<typename T>
bool hostIsNan(const T value) {
    return std::isnan(value);
}

template<>
bool hostIsNan<cfloat>(const cfloat value) {
    return std::isnan(value.real) || std::isnan(value.imag);
}

template<>
bool hostIsNan<cdouble>(const cdouble value) {
    return std::isnan(value.real) || std::isnan(value.imag);
}

template<typename T>
T nanReplacement(const double value) {
    return T(value);
}

template<typename T>
void checkDimensionalSumNanval(const dim_t width, const int dim) {
    const double nanval = -0.375;
    const dim4 dims     = dimensionalReduceDims(width, dim);
    dim4 outputDims     = dims;
    outputDims[dim]     = 1;
    vector<T> input(static_cast<size_t>(dims.elements()));
    vector<T> expected(static_cast<size_t>(outputDims.elements()));

    for (dim_t w = 0; w < outputDims[3]; ++w) {
        for (dim_t z = 0; z < outputDims[2]; ++z) {
            for (dim_t y = 0; y < outputDims[1]; ++y) {
                for (dim_t x = 0; x < outputDims[0]; ++x) {
                    dim_t coords[] = {x, y, z, w};
                    const size_t outputIndex =
                        linearIndex(outputDims, x, y, z, w);
                    T value = T(0);
                    for (dim_t r = 0; r < dims[dim]; ++r) {
                        coords[dim]  = r;
                        T inputValue = finiteNanTestValue<T>(outputIndex, r);
                        if (r == 5 || r == 128) {
                            inputValue = hostNan<T>(outputIndex + r);
                        }
                        input[linearIndex(dims, coords[0], coords[1], coords[2],
                                          coords[3])] = inputValue;
                        const T foldValue             = hostIsNan(inputValue)
                                                            ? nanReplacement<T>(nanval)
                                                            : inputValue;
                        value = hostAdd(foldValue, value);
                    }
                    expected[outputIndex] = value;
                }
            }
        }
    }

    const array inputArray(dims, input.data());
    const array rawOutput = af::sum(inputArray, dim);
    vector<T> raw(expected.size());
    rawOutput.host(raw.data());
    for (size_t i = 0; i < raw.size(); ++i) { EXPECT_TRUE(hostIsNan(raw[i])); }

    const array output = af::sum(inputArray, dim, nanval);
    vector<T> actual(expected.size());
    output.host(actual.data());
    expectBitwiseEqual(expected, actual);
}

template<typename T>
void checkDimensionalProductNanval(const dim_t width, const int dim) {
    const double nanval = 0.75;
    const dim4 dims     = dimensionalReduceDims(width, dim);
    dim4 outputDims     = dims;
    outputDims[dim]     = 1;
    vector<T> input(static_cast<size_t>(dims.elements()));
    vector<T> expected(static_cast<size_t>(outputDims.elements()));

    for (dim_t w = 0; w < outputDims[3]; ++w) {
        for (dim_t z = 0; z < outputDims[2]; ++z) {
            for (dim_t y = 0; y < outputDims[1]; ++y) {
                for (dim_t x = 0; x < outputDims[0]; ++x) {
                    dim_t coords[] = {x, y, z, w};
                    const size_t outputIndex =
                        linearIndex(outputDims, x, y, z, w);
                    T value = T(1);
                    for (dim_t r = 0; r < dims[dim]; ++r) {
                        coords[dim]  = r;
                        T inputValue = finiteNanTestValue<T>(outputIndex, r);
                        if (r == 5 || r == 128) {
                            inputValue = hostNan<T>(outputIndex + r);
                        }
                        input[linearIndex(dims, coords[0], coords[1], coords[2],
                                          coords[3])] = inputValue;
                        const T foldValue             = hostIsNan(inputValue)
                                                            ? nanReplacement<T>(nanval)
                                                            : inputValue;
                        value                         = foldValue * value;
                    }
                    expected[outputIndex] = value;
                }
            }
        }
    }

    const array inputArray(dims, input.data());
    const array rawOutput = af::product(inputArray, dim);
    vector<T> raw(expected.size());
    rawOutput.host(raw.data());
    for (size_t i = 0; i < raw.size(); ++i) { EXPECT_TRUE(hostIsNan(raw[i])); }

    const array output = af::product(inputArray, dim, nanval);
    vector<T> actual(expected.size());
    output.host(actual.data());
    expectBitwiseEqual(expected, actual);
}

template<typename T>
void checkComplexScalarNanvalParity() {
    const double nanval = 0.625;
    const T nan         = hostNan<T>(0);
    const T values[]    = {T(1.25, -0.5), nan, T(-0.75, 0.25), hostNan<T>(1),
                           T(0.5, 0.125)};
    const array input(5, values);

    const T scalarSum = af::sum<T>(input, nanval);
    const T arraySum  = af::sum<array>(input, nanval).scalar<T>();
    EXPECT_EQ(0, std::memcmp(&scalarSum, &arraySum, sizeof(T)));

    const T scalarProduct = af::product<T>(input, nanval);
    const T arrayProduct  = af::product<array>(input, nanval).scalar<T>();
    EXPECT_EQ(0, std::memcmp(&scalarProduct, &arrayProduct, sizeof(T)));
}

}  // namespace

TEST(ReduceDimExactCpu, SumF32PreservesScalarFold) {
    SKIP_IF_FAST_MATH_ENABLED();
    const dim_t widths[] = {7, 8, 9, 31, 32, 35};
    checkSumWidths<float>(widths, sizeof(widths) / sizeof(widths[0]));
}

TEST(ReduceDimExactCpu, SumF64PreservesScalarFold) {
    SKIP_IF_FAST_MATH_ENABLED();
    const dim_t widths[] = {3, 4, 5, 15, 16, 19};
    checkSumWidths<double>(widths, sizeof(widths) / sizeof(widths[0]));
}

TEST(ReduceDimExactCpu, SumC32PreservesScalarFold) {
    SKIP_IF_FAST_MATH_ENABLED();
    const dim_t widths[] = {3, 4, 5, 15, 16, 19};
    checkSumWidths<cfloat>(widths, sizeof(widths) / sizeof(widths[0]));
}

TEST(ReduceDimExactCpu, SumC64PreservesScalarFold) {
    SKIP_IF_FAST_MATH_ENABLED();
    const dim_t widths[] = {1, 2, 3, 7, 8, 11};
    checkSumWidths<cdouble>(widths, sizeof(widths) / sizeof(widths[0]));
}

TEST(ReduceDimExactCpu, ProductF32PreservesScalarFold) {
    SKIP_IF_FAST_MATH_ENABLED();
    const dim_t widths[] = {7, 8, 9, 31, 32, 35};
    checkProductWidths<float>(widths, sizeof(widths) / sizeof(widths[0]));
}

TEST(ReduceDimExactCpu, ProductF64PreservesScalarFold) {
    SKIP_IF_FAST_MATH_ENABLED();
    const dim_t widths[] = {3, 4, 5, 15, 16, 19};
    checkProductWidths<double>(widths, sizeof(widths) / sizeof(widths[0]));
}

TEST(ReduceDimExactCpu, MinMaxF32PreserveScalarFold) {
    SKIP_IF_FAST_MATH_ENABLED();
    const dim_t widths[] = {7, 8, 9, 31, 32, 35};
    checkExtremaWidths<float>(widths, sizeof(widths) / sizeof(widths[0]));
}

TEST(ReduceDimExactCpu, MinMaxF64PreserveScalarFold) {
    SKIP_IF_FAST_MATH_ENABLED();
    const dim_t widths[] = {3, 4, 5, 15, 16, 19};
    checkExtremaWidths<double>(widths, sizeof(widths) / sizeof(widths[0]));
}

TEST(ReduceDimExactCpu, GappedSubarraysPreserveScalarFold) {
    SKIP_IF_FAST_MATH_ENABLED();
    for (int dim = 1; dim <= 3; ++dim) {
        SCOPED_TRACE(::testing::Message() << "dimension " << dim);
        checkGappedDimensionalSum(dim);
        checkGappedDimensionalExtrema<float>(35, dim);
        checkGappedDimensionalExtrema<double>(19, dim);
    }
}

TEST(ReduceDimExactCpu, NanAndNanvalSemantics) {
    SKIP_IF_FAST_MATH_ENABLED();
    for (int dim = 1; dim <= 3; ++dim) {
        SCOPED_TRACE(::testing::Message() << "dimension " << dim);
        checkDimensionalSumNanval<float>(35, dim);
        checkDimensionalSumNanval<double>(19, dim);
        checkDimensionalSumNanval<cfloat>(19, dim);
        checkDimensionalSumNanval<cdouble>(11, dim);
        checkDimensionalProductNanval<float>(35, dim);
        checkDimensionalProductNanval<double>(19, dim);
    }
}

TEST(ReduceDimExactCpu, ComplexScalarNanvalMatchesArrayReturn) {
    SKIP_IF_FAST_MATH_ENABLED();
    checkComplexScalarNanvalParity<cfloat>();
    checkComplexScalarNanvalParity<cdouble>();
}
