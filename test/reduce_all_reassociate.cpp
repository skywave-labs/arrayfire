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
#include <complex>
#include <cstdlib>
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

constexpr size_t blockElements        = 1 << 16;
constexpr size_t minParallelElements  = 1 << 17;
constexpr size_t maximumPartialBlocks = 1024;

int enableReassociatedReduceAll() {
#if defined(_WIN32)
    return _putenv_s("AF_CPU_REDUCE_ALL_REASSOCIATE", "1");
#else
    return setenv("AF_CPU_REDUCE_ALL_REASSOCIATE", "1", 1);
#endif
}

// The production setting is intentionally cached on first use. Configure this
// dedicated test process before any ArrayFire call instead of changing it from
// individual tests.
const int reassociationEnvironmentResult = enableReassociatedReduceAll();

template<typename T>
T makeValue(const double realValue, const double = 0.0) {
    return static_cast<T>(realValue);
}

template<>
cfloat makeValue<cfloat>(const double realValue, const double imagValue) {
    return cfloat(static_cast<float>(realValue), static_cast<float>(imagValue));
}

template<>
cdouble makeValue<cdouble>(const double realValue, const double imagValue) {
    return cdouble(realValue, imagValue);
}

template<typename T>
struct RealType {
    typedef T type;
};

template<>
struct RealType<cfloat> {
    typedef float type;
};

template<>
struct RealType<cdouble> {
    typedef double type;
};

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
T hostMultiply(const T lhs, const T rhs) {
    return lhs * rhs;
}

template<>
cfloat hostMultiply<cfloat>(const cfloat lhs, const cfloat rhs) {
    const std::complex<float> left(lhs.real, lhs.imag);
    const std::complex<float> right(rhs.real, rhs.imag);
    const std::complex<float> result = left * right;
    return cfloat(result.real(), result.imag());
}

template<>
cdouble hostMultiply<cdouble>(const cdouble lhs, const cdouble rhs) {
    const std::complex<double> left(lhs.real, lhs.imag);
    const std::complex<double> right(rhs.real, rhs.imag);
    const std::complex<double> result = left * right;
    return cdouble(result.real(), result.imag());
}

template<typename T>
bool hostIsNaN(const T value) {
    return std::isnan(value);
}

template<>
bool hostIsNaN<cfloat>(const cfloat value) {
    return std::isnan(value.real) || std::isnan(value.imag);
}

template<>
bool hostIsNaN<cdouble>(const cdouble value) {
    return std::isnan(value.real) || std::isnan(value.imag);
}

template<typename T>
T hostOperation(const T input, const T accumulator, const bool product) {
    return product ? hostMultiply(input, accumulator)
                   : hostAdd(input, accumulator);
}

template<typename T>
T transformedValue(const T input, const bool changeNaN, const double nanValue) {
    return changeNaN && hostIsNaN(input) ? makeValue<T>(nanValue) : input;
}

template<typename T>
T hostLeftFold(const vector<T> &values, const bool product,
               const bool changeNaN = false, const double nanValue = 0.0) {
    T result = makeValue<T>(product ? 1.0 : 0.0);
    for (size_t index = 0; index < values.size(); ++index) {
        result =
            hostOperation(transformedValue(values[index], changeNaN, nanValue),
                          result, product);
    }
    return result;
}

template<typename T>
T hostBlockedFold(const vector<T> &values, const bool product,
                  const bool changeNaN = false, const double nanValue = 0.0) {
    const size_t requestedBlocks = 1 + (values.size() - 1) / blockElements;
    const size_t blockCount = std::min(maximumPartialBlocks, requestedBlocks);
    vector<T> partials(blockCount);

    const size_t elementsPerBlock = values.size() / blockCount;
    const size_t remainder        = values.size() % blockCount;
    for (size_t block = 0; block < blockCount; ++block) {
        size_t begin = block * elementsPerBlock + std::min(block, remainder);
        const size_t end =
            begin + elementsPerBlock + (block < remainder ? 1 : 0);

        T partial;
        if (block == 0) {
            partial = makeValue<T>(product ? 1.0 : 0.0);
        } else {
            partial = transformedValue(values[begin], changeNaN, nanValue);
            ++begin;
        }
        for (size_t index = begin; index < end; ++index) {
            partial = hostOperation(
                transformedValue(values[index], changeNaN, nanValue), partial,
                product);
        }
        partials[block] = partial;
    }

    T result = partials[0];
    for (size_t block = 1; block < blockCount; ++block) {
        result = hostOperation(partials[block], result, product);
    }
    return result;
}

template<typename T>
void expectBitwiseEqual(const T expected, const T actual) {
    EXPECT_EQ(0, std::memcmp(&expected, &actual, sizeof(T)));
}

template<typename T>
void checkArrayReduction(const array &input, const T expected,
                         const bool product, const bool changeNaN = false,
                         const double nanValue = 0.0) {
    T scalarResult;
    T arrayResult;
    if (product) {
        scalarResult =
            changeNaN ? af::product<T>(input, nanValue) : af::product<T>(input);
        arrayResult = (changeNaN ? af::product<array>(input, nanValue)
                                 : af::product<array>(input))
                          .scalar<T>();
    } else {
        scalarResult =
            changeNaN ? af::sum<T>(input, nanValue) : af::sum<T>(input);
        arrayResult = (changeNaN ? af::sum<array>(input, nanValue)
                                 : af::sum<array>(input))
                          .scalar<T>();
    }
    expectBitwiseEqual(expected, scalarResult);
    expectBitwiseEqual(expected, arrayResult);
}

template<typename T>
void checkNonAssociativeRealOrComplex() {
    typedef typename RealType<T>::type Real;
    const size_t elements = minParallelElements;

    vector<T> sumValues(elements, makeValue<T>(0.0));
    const Real large =
        std::ldexp(Real(1), std::numeric_limits<Real>::digits + 1);
    sumValues[blockElements - 1] = makeValue<T>(large);
    sumValues[blockElements]     = makeValue<T>(-large);
    sumValues[blockElements + 1] = makeValue<T>(1.0);
    const T leftSum              = hostLeftFold(sumValues, false);
    const T blockedSum           = hostBlockedFold(sumValues, false);
    ASSERT_NE(0, std::memcmp(&leftSum, &blockedSum, sizeof(T)));
    checkArrayReduction(array(static_cast<dim_t>(elements), sumValues.data()),
                        blockedSum, false);

    vector<T> productValues(elements, makeValue<T>(1.0));
    productValues[blockElements - 1] =
        makeValue<T>(std::numeric_limits<Real>::max());
    productValues[blockElements]     = makeValue<T>(2.0);
    productValues[blockElements + 1] = makeValue<T>(0.5);
    const T leftProduct              = hostLeftFold(productValues, true);
    const T blockedProduct           = hostBlockedFold(productValues, true);
    ASSERT_NE(0, std::memcmp(&leftProduct, &blockedProduct, sizeof(T)));
    checkArrayReduction(
        array(static_cast<dim_t>(elements), productValues.data()),
        blockedProduct, true);
}

template<typename T>
void checkNaNReplacementAtBlockSeed() {
    typedef typename RealType<T>::type Real;
    const size_t elements = minParallelElements;
    const Real nan        = std::numeric_limits<Real>::quiet_NaN();
    const double nanValue = -0.375;

    vector<T> sumValues(elements, makeValue<T>(0.25, -0.125));
    sumValues[blockElements] = makeValue<T>(nan, 1.0);
    const T expectedSum = hostBlockedFold(sumValues, false, true, nanValue);
    checkArrayReduction(array(static_cast<dim_t>(elements), sumValues.data()),
                        expectedSum, false, true, nanValue);

    vector<T> productValues(elements, makeValue<T>(1.0));
    productValues[blockElements] = makeValue<T>(1.0, nan);
    const T expectedProduct =
        hostBlockedFold(productValues, true, true, nanValue);
    checkArrayReduction(
        array(static_cast<dim_t>(elements), productValues.data()),
        expectedProduct, true, true, nanValue);
}

template<typename T>
void checkGappedView() {
    const dim4 parentDims(257, 69, 5, 13);
    vector<T> sumParent(static_cast<size_t>(parentDims.elements()));
    vector<T> productParent(sumParent.size(), makeValue<T>(7.0, 3.0));
    vector<T> sumView;
    vector<T> productView;
    sumView.reserve(257 * 67 * 5 * 13);
    productView.reserve(sumView.capacity());

    size_t linear = 0;
    for (dim_t w = 0; w < parentDims[3]; ++w) {
        for (dim_t z = 0; z < parentDims[2]; ++z) {
            for (dim_t y = 0; y < parentDims[1]; ++y) {
                for (dim_t x = 0; x < parentDims[0]; ++x, ++linear) {
                    const int centered = static_cast<int>(linear % 7) - 3;
                    sumParent[linear]  = makeValue<T>(
                        centered, static_cast<int>(linear % 5) - 2);
                    if (y >= 1 && y <= 67) {
                        const T product = makeValue<T>(
                            (productView.size() % 17 == 0) ? -1.0 : 1.0);
                        productParent[linear] = product;
                        sumView.push_back(sumParent[linear]);
                        productView.push_back(product);
                    }
                }
            }
        }
    }

    const array sumParentArray(parentDims, sumParent.data());
    const array productParentArray(parentDims, productParent.data());
    const array sumInput     = sumParentArray(span, seq(1, 67), span, span);
    const array productInput = productParentArray(span, seq(1, 67), span, span);
    checkArrayReduction(sumInput, hostBlockedFold(sumView, false), false);
    checkArrayReduction(productInput, hostBlockedFold(productView, true), true);
}

}  // namespace

TEST(ReduceAllReassociate, ConfigurationIsEnabledBeforeFirstUse) {
    ASSERT_EQ(0, reassociationEnvironmentResult);
}

TEST(ReduceAllReassociate, RealAndComplexUseDocumentedBlockedTree) {
    SKIP_IF_FAST_MATH_ENABLED();
    checkNonAssociativeRealOrComplex<float>();
    checkNonAssociativeRealOrComplex<double>();
    checkNonAssociativeRealOrComplex<cfloat>();
    checkNonAssociativeRealOrComplex<cdouble>();
}

TEST(ReduceAllReassociate, HalfPromotesIntoBlockedFloatFold) {
    SKIP_IF_FAST_MATH_ENABLED();
    const size_t elements = minParallelElements;

    vector<float> sumValues(elements, 0.0F);
    sumValues[blockElements - 1] = 65504.0F;
    sumValues[blockElements]     = -65504.0F;
    sumValues[blockElements + 1] = 0.0009765625F;
    const float leftSum          = hostLeftFold(sumValues, false);
    const float blockedSum       = hostBlockedFold(sumValues, false);
    ASSERT_NE(0, std::memcmp(&leftSum, &blockedSum, sizeof(float)));
    const array sumInput =
        array(static_cast<dim_t>(elements), sumValues.data()).as(f16);
    checkArrayReduction(sumInput, blockedSum, false);

    vector<float> productValues(elements);
    for (size_t index = 0; index < elements; ++index) {
        productValues[index] = (index & 1U) ? 0.5F : 2.0F;
    }
    const array productInput =
        array(static_cast<dim_t>(elements), productValues.data()).as(f16);
    checkArrayReduction(productInput, hostBlockedFold(productValues, true),
                        true);
}

TEST(ReduceAllReassociate, NaNReplacementIncludesBlockSeeds) {
    SKIP_IF_FAST_MATH_ENABLED();
    checkNaNReplacementAtBlockSeed<float>();
    checkNaNReplacementAtBlockSeed<double>();
    checkNaNReplacementAtBlockSeed<cfloat>();
    checkNaNReplacementAtBlockSeed<cdouble>();
}

TEST(ReduceAllReassociate, GeneratedNaNIsNotTreatedAsAnInputNaN) {
    SKIP_IF_FAST_MATH_ENABLED();
    vector<float> values(minParallelElements, 0.0F);
    values[0] = std::numeric_limits<float>::infinity();
    values[1] = -std::numeric_limits<float>::infinity();
    const array input(static_cast<dim_t>(values.size()), values.data());
    EXPECT_TRUE(std::isnan(af::sum<float>(input, 0.625)));
    EXPECT_TRUE(std::isnan(af::sum<array>(input, 0.625).scalar<float>()));
}

TEST(ReduceAllReassociate, GappedRealAndComplexViewsUseLogicalOrder) {
    SKIP_IF_FAST_MATH_ENABLED();
    checkGappedView<float>();
    checkGappedView<cfloat>();
}

TEST(ReduceAllReassociate, LazyInputIsEvaluatedBeforeBlockedReduction) {
    const size_t elements = minParallelElements + 37;
    const array input =
        af::range(dim4(static_cast<dim_t>(elements)), 0, f32) * 0.0F + 1.0F;
    vector<float> values(elements, 1.0F);
    checkArrayReduction(input, hostBlockedFold(values, false), false);
    checkArrayReduction(input, hostBlockedFold(values, true), true);
}

TEST(ReduceAllReassociate, BelowThresholdAndIntegralInputsRemainExact) {
    SKIP_IF_FAST_MATH_ENABLED();
    const size_t elements = minParallelElements - 1;
    vector<float> sumValues(elements, 0.0F);
    const float large =
        std::ldexp(1.0F, std::numeric_limits<float>::digits + 1);
    sumValues[blockElements - 1] = large;
    sumValues[blockElements]     = -large;
    sumValues[blockElements + 1] = 1.0F;
    checkArrayReduction(array(static_cast<dim_t>(elements), sumValues.data()),
                        hostLeftFold(sumValues, false), false);

    vector<int> integerSum(minParallelElements, 0);
    vector<int> integerProduct(minParallelElements, 1);
    for (size_t index = 0; index < integerSum.size(); ++index) {
        integerSum[index] = static_cast<int>(index % 7) - 3;
    }
    integerProduct[blockElements] = -1;
    checkArrayReduction(
        array(static_cast<dim_t>(integerSum.size()), integerSum.data()),
        hostLeftFold(integerSum, false), false);
    checkArrayReduction(
        array(static_cast<dim_t>(integerProduct.size()), integerProduct.data()),
        hostLeftFold(integerProduct, true), true);
}
