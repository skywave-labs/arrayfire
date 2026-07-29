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

#include <limits>
#include <vector>

using af::array;
using af::dim4;

namespace {

constexpr dim_t minimumElements      = 1 << 16;
constexpr dim_t minimumDimensionZero = 64;

template<typename T>
af::dtype typeOf() {
    return static_cast<af::dtype>(af::dtype_traits<T>::af_type);
}

template<typename T>
float tolerance();

template<>
float tolerance<float>() {
    return 1.0e-3F;
}

template<>
float tolerance<double>() {
    return 1.0e-10F;
}

template<typename T>
array readyConstant(T value, const dim4 &dims) {
    array result = af::constant(value, dims, typeOf<T>());
    result.eval();
    return result;
}

template<typename T>
array readyIota(const dim4 &dims) {
    array result = af::iota(dims, dim4(1), typeOf<T>());
    result.eval();
    return result;
}

template<typename T>
array readyRange(const dim4 &dims, int sequenceDimension) {
    array result = af::range(dims, sequenceDimension, typeOf<T>());
    result.eval();
    return result;
}

template<typename T>
array affine(const array &input, T scale, T offset) {
    return input * scale + offset;
}

template<typename T>
void expectDimensionSum(const array &lazyInput, array referenceInput, int dim) {
    array actual = af::sum(lazyInput, dim);
    referenceInput.eval();
    array expected = af::sum(referenceInput, dim);

    ASSERT_ARRAYS_NEAR(expected, actual, tolerance<T>());
}

template<typename T>
void expectWholeArraySum(const array &lazyInput, array referenceInput) {
    array actual = af::sum<array>(lazyInput);
    referenceInput.eval();
    array expected = af::sum<array>(referenceInput);

    ASSERT_ARRAYS_NEAR(expected, actual, tolerance<T>());
}

template<typename T>
class JITReduce : public ::testing::Test {};

using FloatingTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(JITReduce, FloatingTypes);

TYPED_TEST(JITReduce, DimensionZeroUsesMultiplePartialBlocks) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    // A dimension larger than one CUDA reduction block exercises the partial
    // output and final reduction path. The total also meets the fusion cutoff.
    const dim4 dims(32768, 2);
    array base = readyConstant<TypeParam>(TypeParam(1), dims);

    array lazyInput      = affine(base, TypeParam(2), TypeParam(0.5));
    array referenceInput = affine(base, TypeParam(2), TypeParam(0.5));

    expectDimensionSum<TypeParam>(lazyInput, referenceInput, 0);
}

TYPED_TEST(JITReduce, DimensionZeroHandlesNonPowerOfTwoPartialTail) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    const dim4 dims(10003, 7);
    array base            = readyIota<TypeParam>(dims);
    const TypeParam scale = TypeParam(1.0 / 8192.0);

    array lazyInput      = affine(base, scale, TypeParam(0.25));
    array referenceInput = affine(base, scale, TypeParam(0.25));

    expectDimensionSum<TypeParam>(lazyInput, referenceInput, 0);
}

TYPED_TEST(JITReduce, DimensionZeroMapsFourDimensionalLines) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    const dim4 dims(257, 17, 5, 4);
    array base            = readyIota<TypeParam>(dims);
    const TypeParam scale = TypeParam(1.0 / 65536.0);

    array lazyInput      = affine(base, scale, TypeParam(1));
    array referenceInput = affine(base, scale, TypeParam(1));

    expectDimensionSum<TypeParam>(lazyInput, referenceInput, 0);
}

TYPED_TEST(JITReduce, WholeArrayAtEligibilityBoundary) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    const dim4 dims(256, 256);
    ASSERT_EQ(minimumElements, dims.elements());

    array base           = readyConstant<TypeParam>(TypeParam(2), dims);
    array lazyInput      = affine(base, TypeParam(3), TypeParam(1));
    array referenceInput = affine(base, TypeParam(3), TypeParam(1));

    expectWholeArraySum<TypeParam>(lazyInput, referenceInput);
}

TYPED_TEST(JITReduce, MultipleBuffersFeedTheFusedProducer) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    const dim4 dims(16384, 4);
    array left  = readyRange<TypeParam>(dims, 0);
    array right = readyRange<TypeParam>(dims, 1);

    array lazyInput      = left * TypeParam(0.25) + right * TypeParam(0.5);
    array referenceInput = left * TypeParam(0.25) + right * TypeParam(0.5);

    expectDimensionSum<TypeParam>(lazyInput, referenceInput, 0);
}

TYPED_TEST(JITReduce, ReplacesNanForDimensionAndWholeArraySums) {
    SKIP_IF_FAST_MATH_ENABLED();
    SUPPORTED_TYPE_CHECK(TypeParam);

    const dim4 dims(16384, 4);
    std::vector<TypeParam> values(dims.elements(), TypeParam(1));
    const TypeParam nan       = std::numeric_limits<TypeParam>::quiet_NaN();
    values[0]                 = nan;
    values[8191]              = nan;
    values[8192]              = nan;
    values[32773]             = nan;
    values[values.size() - 1] = nan;

    array base(dims, values.data());
    base.eval();

    const double replacement = -4.0;
    array lazyInput          = affine(base, TypeParam(2), TypeParam(1));
    array actualDimension    = af::sum(lazyInput, 0, replacement);
    array actualWhole        = af::sum<array>(lazyInput, replacement);

    array referenceInput = affine(base, TypeParam(2), TypeParam(1));
    referenceInput.eval();
    array expectedDimension = af::sum(referenceInput, 0, replacement);
    array expectedWhole     = af::sum<array>(referenceInput, replacement);

    ASSERT_ARRAYS_NEAR(expectedDimension, actualDimension,
                       tolerance<TypeParam>());
    ASSERT_ARRAYS_NEAR(expectedWhole, actualWhole, tolerance<TypeParam>());
}

TYPED_TEST(JITReduce, CachedKernelReceivesFreshBufferAndScalarArguments) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    const dim4 dims(256, 256);
    array firstBuffer  = readyConstant<TypeParam>(TypeParam(1), dims);
    array secondBuffer = readyConstant<TypeParam>(TypeParam(4), dims);

    // These four expressions have the same node topology. The first pair
    // changes the input allocation; the second pair changes scalar arguments.
    array firstBufferActual =
        af::sum<array>(affine(firstBuffer, TypeParam(2), TypeParam(1)));
    array secondBufferActual =
        af::sum<array>(affine(secondBuffer, TypeParam(2), TypeParam(1)));
    array firstScalarsActual =
        af::sum<array>(affine(firstBuffer, TypeParam(3), TypeParam(2)));
    array secondScalarsActual =
        af::sum<array>(affine(firstBuffer, TypeParam(5), TypeParam(4)));

    array firstBufferReference =
        affine(firstBuffer, TypeParam(2), TypeParam(1));
    array secondBufferReference =
        affine(secondBuffer, TypeParam(2), TypeParam(1));
    array firstScalarsReference =
        affine(firstBuffer, TypeParam(3), TypeParam(2));
    array secondScalarsReference =
        affine(firstBuffer, TypeParam(5), TypeParam(4));
    af::eval(firstBufferReference, secondBufferReference, firstScalarsReference,
             secondScalarsReference);

    ASSERT_ARRAYS_NEAR(af::sum<array>(firstBufferReference), firstBufferActual,
                       tolerance<TypeParam>());
    ASSERT_ARRAYS_NEAR(af::sum<array>(secondBufferReference),
                       secondBufferActual, tolerance<TypeParam>());
    ASSERT_ARRAYS_NEAR(af::sum<array>(firstScalarsReference),
                       firstScalarsActual, tolerance<TypeParam>());
    ASSERT_ARRAYS_NEAR(af::sum<array>(secondScalarsReference),
                       secondScalarsActual, tolerance<TypeParam>());
}

TYPED_TEST(JITReduce, NamedExpressionCanFeedTwoReductionConsumers) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    const dim4 dims(minimumDimensionZero,
                    minimumElements / minimumDimensionZero);
    ASSERT_EQ(minimumElements, dims.elements());
    ASSERT_EQ(minimumDimensionZero, dims[0]);

    array base            = readyConstant<TypeParam>(TypeParam(2), dims);
    array namedExpression = affine(base, TypeParam(2), TypeParam(1));

    array actualDimension = af::sum(namedExpression, 0);
    array actualWhole     = af::sum<array>(namedExpression);

    array referenceInput = affine(base, TypeParam(2), TypeParam(1));
    referenceInput.eval();
    array expectedDimension = af::sum(referenceInput, 0);
    array expectedWhole     = af::sum<array>(referenceInput);

    ASSERT_ARRAYS_NEAR(expectedDimension, actualDimension,
                       tolerance<TypeParam>());
    ASSERT_ARRAYS_NEAR(expectedWhole, actualWhole, tolerance<TypeParam>());
}

TYPED_TEST(JITReduce, ReadySmallAndShortDimensionZeroInputsRemainCorrect) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    {
        const dim4 dims(256, 256);
        array base       = readyConstant<TypeParam>(TypeParam(2), dims);
        array readyInput = affine(base, TypeParam(2), TypeParam(1));
        readyInput.eval();
        array referenceInput = affine(base, TypeParam(2), TypeParam(1));
        referenceInput.eval();

        ASSERT_ARRAYS_NEAR(af::sum(referenceInput, 0), af::sum(readyInput, 0),
                           tolerance<TypeParam>());
        ASSERT_ARRAYS_NEAR(af::sum<array>(referenceInput),
                           af::sum<array>(readyInput), tolerance<TypeParam>());
    }

    {
        const dim4 dims(128, 128);
        array base = readyConstant<TypeParam>(TypeParam(3), dims);
        expectDimensionSum<TypeParam>(affine(base, TypeParam(2), TypeParam(1)),
                                      affine(base, TypeParam(2), TypeParam(1)),
                                      0);
        expectWholeArraySum<TypeParam>(
            affine(base, TypeParam(2), TypeParam(1)),
            affine(base, TypeParam(2), TypeParam(1)));
    }

    {
        // The total is eligible, but dimension zero is deliberately below the
        // minimum fused dimension length.
        const dim4 dims(32, 2048);
        ASSERT_EQ(minimumElements, dims.elements());
        array base = readyConstant<TypeParam>(TypeParam(4), dims);
        expectDimensionSum<TypeParam>(affine(base, TypeParam(2), TypeParam(1)),
                                      affine(base, TypeParam(2), TypeParam(1)),
                                      0);
    }
}

TYPED_TEST(JITReduce, DimensionsOneThroughThreeRemainCorrect) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    const dim4 dims(64, 16, 8, 8);
    ASSERT_EQ(minimumElements, dims.elements());
    array base            = readyIota<TypeParam>(dims);
    const TypeParam scale = TypeParam(1.0 / 65536.0);

    for (int dim = 1; dim < 4; ++dim) {
        SCOPED_TRACE(::testing::Message() << "dimension " << dim);
        expectDimensionSum<TypeParam>(affine(base, scale, TypeParam(1)),
                                      affine(base, scale, TypeParam(1)), dim);
    }
}

TYPED_TEST(JITReduce, NonLinearExpressionShapesRemainCorrect) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    {
        array parent = readyRange<TypeParam>(dim4(256, 512), 0);
        array gapped = parent(af::seq(0, 255, 2), af::span);
        ASSERT_EQ(minimumElements, gapped.elements());

        expectDimensionSum<TypeParam>(
            affine(gapped, TypeParam(0.5), TypeParam(1)),
            affine(gapped, TypeParam(0.5), TypeParam(1)), 0);
    }

    {
        // Broadcasting is represented by singleton-dimension tile/no-op nodes
        // in the JIT expression.
        const dim4 dims(256, 256);
        array rows   = readyRange<TypeParam>(dims, 1);
        array column = readyRange<TypeParam>(dim4(dims[0], 1), 0);

        expectDimensionSum<TypeParam>(rows + column, rows + column, 0);
    }

    {
        const dim4 dims(256, 256);
        array base = readyRange<TypeParam>(dims, 1);

        expectDimensionSum<TypeParam>(af::shift(base, 0, 1) + TypeParam(1),
                                      af::shift(base, 0, 1) + TypeParam(1), 0);
    }

    {
        array base      = readyRange<TypeParam>(dim4(128, 512), 0);
        array lazyInput = af::moddims(affine(base, TypeParam(2), TypeParam(1)),
                                      dim4(256, 256));
        array referenceInput = af::moddims(
            affine(base, TypeParam(2), TypeParam(1)), dim4(256, 256));

        expectDimensionSum<TypeParam>(lazyInput, referenceInput, 0);
    }
}

}  // namespace
