/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using std::endl;
using std::vector;

template<typename T>
class Transpose : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, uint, char, schar,
                         uchar, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Transpose, TestTypes);

template<typename T>
void transposeip_test(dim4 dims, const bool conjugate = false) {
    SUPPORTED_TYPE_CHECK(T);

    af_array inArray  = 0;
    af_array outArray = 0;

    ASSERT_SUCCESS(af_randu(&inArray, dims.ndims(), dims.get(),
                            (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_transpose(&outArray, inArray, conjugate));
    ASSERT_SUCCESS(af_eval(outArray));
    ASSERT_SUCCESS(af_transpose_inplace(inArray, conjugate));

    ASSERT_ARRAYS_EQ(inArray, outArray);

    // cleanup
    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

#define INIT_TEST(Side, D3, D4)                                \
    TYPED_TEST(Transpose, TranposeIP_##Side) {                 \
        transposeip_test<TypeParam>(dim4(Side, Side, D3, D4)); \
    }

INIT_TEST(10, 1, 1);
INIT_TEST(64, 1, 1);
INIT_TEST(300, 1, 1);
INIT_TEST(1000, 1, 1);
INIT_TEST(100, 2, 1);
INIT_TEST(25, 2, 2);

TEST(TransposeInPlace, ParallelOddBatched) {
    transposeip_test<float>(dim4(513, 513, 3, 2));
}

TEST(TransposeInPlace, ParallelConjugateOddBatched) {
    transposeip_test<cfloat>(dim4(513, 513, 2, 2), true);
    transposeip_test<cdouble>(dim4(257, 257, 2, 1), true);
}

TEST(TransposeInPlace, ConjugateSingletonBatches) {
    transposeip_test<cfloat>(dim4(1, 1, 3, 2), true);
    transposeip_test<cdouble>(dim4(1, 1, 2, 3), true);
}

TEST(TransposeInPlace, ManySmallBatches) {
    transposeip_test<cfloat>(dim4(17, 17, 257, 5), true);
}

TEST(TransposeInPlace, PaddedOffsetView) {
    array parent   = randu(dim4(520, 520, 2), c32);
    array view     = parent(af::seq(3, 515), af::seq(2, 514), af::span);
    array expected = transpose(view, true);

    transposeInPlace(view, true);

    ASSERT_ARRAYS_EQ(view, expected);
}

////////////////////////////////////// CPP //////////////////////////////////
//
TEST(TransposeInPlace, CPP) {
    dim4 dims(64, 64, 1, 1);

    array input  = randu(dims);
    array output = transpose(input);
    transposeInPlace(input);

    ASSERT_ARRAYS_EQ(input, output);
}
