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
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using af::seq;
using af::span;
using std::endl;
using std::string;
using std::vector;

template<typename T>
T gradientInputValue(size_t index) {
    return static_cast<T>(static_cast<int>(index % 29) - 14);
}

template<>
cfloat gradientInputValue<cfloat>(size_t index) {
    return cfloat(static_cast<float>(static_cast<int>(index % 29) - 14),
                  static_cast<float>(static_cast<int>(index % 17) - 8));
}

template<>
cdouble gradientInputValue<cdouble>(size_t index) {
    return cdouble(static_cast<double>(static_cast<int>(index % 29) - 14),
                   static_cast<double>(static_cast<int>(index % 17) - 8));
}

template<typename T>
void checkGradientReference(const array& input, const vector<T>& input_values) {
    const dim4 dims = input.dims();
    ASSERT_EQ(static_cast<size_t>(dims.elements()), input_values.size());

    array grad0;
    array grad1;
    af::grad(grad0, grad1, input);

    vector<T> actual0(input_values.size());
    vector<T> actual1(input_values.size());
    grad0.host(actual0.data());
    grad1.host(actual1.data());

    const auto offset = [dims](dim_t x, dim_t y, dim_t z, dim_t w) {
        return static_cast<size_t>(x +
                                   dims[0] * (y + dims[1] * (z + dims[2] * w)));
    };

    for (dim_t w = 0; w < dims[3]; ++w) {
        for (dim_t z = 0; z < dims[2]; ++z) {
            for (dim_t y = 0; y < dims[1]; ++y) {
                const dim_t previous_y = y == 0 ? y : y - 1;
                const dim_t next_y     = y + 1 == dims[1] ? y : y + 1;
                const T y_scale = (y == 0 || y + 1 == dims[1]) ? T(1) : T(0.5);

                for (dim_t x = 0; x < dims[0]; ++x) {
                    const dim_t previous_x = x == 0 ? x : x - 1;
                    const dim_t next_x     = x + 1 == dims[0] ? x : x + 1;
                    const T x_scale =
                        (x == 0 || x + 1 == dims[0]) ? T(1) : T(0.5);
                    const size_t index = offset(x, y, z, w);

                    const T expected0 =
                        x_scale * (input_values[offset(next_x, y, z, w)] -
                                   input_values[offset(previous_x, y, z, w)]);
                    const T expected1 =
                        y_scale * (input_values[offset(x, next_y, z, w)] -
                                   input_values[offset(x, previous_y, z, w)]);

                    ASSERT_EQ(expected0, actual0[index]) << "at: " << index;
                    ASSERT_EQ(expected1, actual1[index]) << "at: " << index;
                }
            }
        }
    }
}

template<typename T>
class Grad : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(af_make_seq(0, 4, 1));
        subMat0.push_back(af_make_seq(2, 6, 1));
        subMat0.push_back(af_make_seq(0, 2, 1));
    }
    vector<af_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;

// register the type list
TYPED_TEST_SUITE(Grad, TestTypes);

template<typename T>
void gradTest(string pTestFile, const unsigned resultIdx0,
              const unsigned resultIdx1, bool isSubRef = false,
              const vector<af_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];

    af_array inArray   = 0;
    af_array tempArray = 0;
    af_array g0Array   = 0;
    af_array g1Array   = 0;

    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        ASSERT_SUCCESS(
            af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(af_gradient(&g0Array, &g1Array, inArray));

    size_t nElems = tests[resultIdx0].size();
    // Get result
    T* grad0Data = new T[tests[resultIdx0].size()];
    ASSERT_SUCCESS(af_get_data_ptr((void*)grad0Data, g0Array));

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], grad0Data[elIter])
            << "at: " << elIter << endl;
    }

    // Get result
    T* grad1Data = new T[tests[resultIdx1].size()];
    ASSERT_SUCCESS(af_get_data_ptr((void*)grad1Data, g1Array));

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx1][elIter], grad1Data[elIter])
            << "at: " << elIter << endl;
    }

    // Delete
    delete[] grad0Data;
    delete[] grad1Data;

    if (inArray != 0) af_release_array(inArray);
    if (g0Array != 0) af_release_array(g0Array);
    if (g1Array != 0) af_release_array(g1Array);
    if (tempArray != 0) af_release_array(tempArray);
}

#define GRAD_INIT(desc, file, resultIdx0, resultIdx1)                \
    TYPED_TEST(Grad, desc) {                                         \
        gradTest<TypeParam>(string(TEST_DIR "/grad/" #file ".test"), \
                            resultIdx0, resultIdx1);                 \
    }

GRAD_INIT(Grad0, grad, 0, 1);
GRAD_INIT(Grad1, grad2D, 0, 1);
GRAD_INIT(Grad2, grad3D, 0, 1);

TYPED_TEST(Grad, VectorBoundariesAndBatches) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    constexpr dim_t widths[]  = {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33};
    constexpr dim_t heights[] = {1, 2, 3, 5};
    for (const dim_t width : widths) {
        for (const dim_t height : heights) {
            const dim4 dims(width, height, 2, 2);
            vector<TypeParam> input_values(
                static_cast<size_t>(dims.elements()));
            for (size_t i = 0; i < input_values.size(); ++i) {
                input_values[i] = gradientInputValue<TypeParam>(i);
            }

            const array input(dims, input_values.data());
            checkGradientReference(input, input_values);
        }
    }
}

TYPED_TEST(Grad, OffsetViewWithNonContiguousRows) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    const dim4 parent_dims(41, 11, 2, 2);
    vector<TypeParam> parent_values(
        static_cast<size_t>(parent_dims.elements()));
    for (size_t i = 0; i < parent_values.size(); ++i) {
        parent_values[i] = gradientInputValue<TypeParam>(i);
    }

    const array parent(parent_dims, parent_values.data());
    const array input = parent(seq(1, 33), seq(1, 9), span, span);
    vector<TypeParam> input_values(static_cast<size_t>(input.elements()));
    input.host(input_values.data());

    checkGradientReference(input, input_values);
}

TYPED_TEST(Grad, StridedColumnsUseScalarFallback) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    af_backend active_backend;
    ASSERT_SUCCESS(af_get_active_backend(&active_backend));
    if (active_backend != AF_BACKEND_CPU) {
        GTEST_SKIP() << "CPU stride-zero fallback regression";
    }

    const dim4 parent_dims(41, 11, 2, 2);
    vector<TypeParam> parent_values(
        static_cast<size_t>(parent_dims.elements()));
    for (size_t i = 0; i < parent_values.size(); ++i) {
        parent_values[i] = gradientInputValue<TypeParam>(i);
    }

    const array parent(parent_dims, parent_values.data());
    const array input = parent(seq(1, 33, 2), seq(1, 9), span, span);
    vector<TypeParam> input_values(static_cast<size_t>(input.elements()));
    input.host(input_values.data());

    checkGradientReference(input, input_values);
}

TYPED_TEST(Grad, ParallelRows) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    af_backend active_backend;
    ASSERT_SUCCESS(af_get_active_backend(&active_backend));
    if (active_backend != AF_BACKEND_CPU) {
        GTEST_SKIP() << "CPU row-scheduling regression";
    }

    const dim4 dims(257, 513);
    vector<TypeParam> input_values(static_cast<size_t>(dims.elements()));
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = gradientInputValue<TypeParam>(i);
    }

    const array input(dims, input_values.data());
    checkGradientReference(input, input_values);
}

/////////////////////////////////////// CPP
//////////////////////////////////////////////
//

TEST(Grad, CPP) {
    const unsigned resultIdx0 = 0;
    const unsigned resultIdx1 = 1;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(string(TEST_DIR "/grad/grad3D.test"),
                                   numDims, in, tests);

    dim4 idims = numDims[0];

    array input(idims, &(in[0].front()));
    array g0, g1;
    grad(g0, g1, input);

    size_t nElems = tests[resultIdx0].size();
    // Get result
    float* grad0Data = new float[tests[resultIdx0].size()];
    g0.host((void*)grad0Data);

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], grad0Data[elIter])
            << "at: " << elIter << endl;
    }

    // Get result
    float* grad1Data = new float[tests[resultIdx1].size()];
    g1.host((void*)grad1Data);

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx1][elIter], grad1Data[elIter])
            << "at: " << elIter << endl;
    }

    // Delete
    delete[] grad0Data;
    delete[] grad1Data;
}

TEST(Grad, MaxDim) {
    using af::constant;
    using af::sum;

    const size_t largeDim = 65535 * 8 + 1;

    array input = constant(1, 2, largeDim);
    array g0, g1;
    grad(g0, g1, input);

    ASSERT_EQ(0.f, sum<float>(g0));
    ASSERT_EQ(0.f, sum<float>(g1));
}
