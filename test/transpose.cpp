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
#include <half.hpp>
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <af/traits.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using af::allTrue;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using af::seq;
using af::span;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Transpose : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat2D.push_back(af_make_seq(2, 7, 1));
        subMat2D.push_back(af_make_seq(2, 7, 1));

        subMat3D.push_back(af_make_seq(2, 7, 1));
        subMat3D.push_back(af_make_seq(2, 7, 1));
        subMat3D.push_back(af_span);
    }
    vector<af_seq> subMat2D;
    vector<af_seq> subMat3D;
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, uint, char, schar,
                         uchar, short, ushort, half_float::half>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Transpose, TestTypes);

template<typename T>
void trsTest(string pTestFile, bool isSubRef = false,
             const vector<af_seq> *seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pTestFile, numDims, in, tests);
    dim4 dims = numDims[0];

    af_array outArray = 0;
    af_array inArray  = 0;
    T *outData;
    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    // check if the test is for indexed Array
    if (isSubRef) {
        dim4 newDims(dims[1] - 4, dims[0] - 4, dims[2], dims[3]);
        af_array subArray = 0;
        ASSERT_SUCCESS(
            af_index(&subArray, inArray, seqv->size(), &seqv->front()));
        ASSERT_SUCCESS(af_transpose(&outArray, subArray, false));
        // destroy the temporary indexed Array
        ASSERT_SUCCESS(af_release_array(subArray));

        dim_t nElems;
        ASSERT_SUCCESS(af_get_elements(&nElems, outArray));
        outData = new T[nElems];
    } else {
        ASSERT_SUCCESS(af_transpose(&outArray, inArray, false));
        outData = new T[dims.elements()];
    }

    ASSERT_SUCCESS(af_get_data_ptr((void *)outData, outArray));

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<T> currGoldBar = tests[testIter];
        size_t nElems         = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << endl;
        }
    }

    // cleanup
    delete[] outData;
    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TYPED_TEST(Transpose, Vector) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/vector.test"));
}

TYPED_TEST(Transpose, VectorBatch) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/vector_batch.test"));
}

TYPED_TEST(Transpose, Square) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/square.test"));
}

TYPED_TEST(Transpose, Rectangle) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/rectangle.test"));
}

TYPED_TEST(Transpose, Rectangle2) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/rectangle2.test"));
}

TYPED_TEST(Transpose, SquareBatch) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/square_batch.test"));
}

TYPED_TEST(Transpose, RectangleBatch) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/rectangle_batch.test"));
}

TYPED_TEST(Transpose, RectangleBatch2) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/rectangle_batch2.test"));
}

TYPED_TEST(Transpose, Square512x512) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/square2.test"));
}

TYPED_TEST(Transpose, SubRef) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/offset.test"), true,
                       &(this->subMat2D));
}

TYPED_TEST(Transpose, SubRefBatch) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/offset_batch.test"), true,
                       &(this->subMat3D));
}

template<typename T>
T transposeInputValue(size_t index) {
    return static_cast<T>(static_cast<int>((17 * index) % 251) - 125);
}

template<>
cfloat transposeInputValue<cfloat>(size_t index) {
    return cfloat(
        static_cast<float>(static_cast<int>((17 * index) % 251) - 125),
        static_cast<float>(static_cast<int>((29 * index) % 127) - 63));
}

template<>
cdouble transposeInputValue<cdouble>(size_t index) {
    return cdouble(
        static_cast<double>(static_cast<int>((17 * index) % 251) - 125),
        static_cast<double>(static_cast<int>((29 * index) % 127) - 63));
}

template<typename T>
T conjugateTransposeValue(const T &value) {
    return value;
}

template<>
cfloat conjugateTransposeValue(const cfloat &value) {
    return cfloat(value.real, -value.imag);
}

template<>
cdouble conjugateTransposeValue(const cdouble &value) {
    return cdouble(value.real, -value.imag);
}

template<typename T>
void checkTransposeReference(const array &input, const vector<T> &input_values,
                             bool conjugate = false) {
    const dim4 dims = input.dims();
    ASSERT_EQ(static_cast<size_t>(dims.elements()), input_values.size());

    const array output = transpose(input, conjugate);
    vector<T> actual(input_values.size());
    output.host(actual.data());

    for (dim_t w = 0; w < dims[3]; ++w) {
        for (dim_t z = 0; z < dims[2]; ++z) {
            for (dim_t y = 0; y < dims[1]; ++y) {
                for (dim_t x = 0; x < dims[0]; ++x) {
                    const size_t input_index = static_cast<size_t>(
                        x + dims[0] * (y + dims[1] * (z + dims[2] * w)));
                    const size_t output_index = static_cast<size_t>(
                        y + dims[1] * (x + dims[0] * (z + dims[2] * w)));
                    const T expected =
                        conjugate
                            ? conjugateTransposeValue(input_values[input_index])
                            : input_values[input_index];
                    ASSERT_EQ(expected, actual[output_index])
                        << "at: " << output_index;
                }
            }
        }
    }
}

template<typename T>
class TransposeTiled : public ::testing::Test {};

using TiledTypes = ::testing::Types<float, double, int, unsigned, long long,
                                    unsigned long long, cfloat, cdouble>;
TYPED_TEST_SUITE(TransposeTiled, TiledTypes);

TYPED_TEST(TransposeTiled, BoundariesAndBatches) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    af_backend active_backend;
    ASSERT_SUCCESS(af_get_active_backend(&active_backend));
    if (active_backend != AF_BACKEND_CPU) {
        GTEST_SKIP() << "CPU tiled-transpose regression";
    }

    const std::array<dim4, 12> shapes = {
        dim4(7, 8),   dim4(8, 7),   dim4(8, 8),      dim4(9, 8),
        dim4(8, 9),   dim4(15, 16), dim4(16, 15),    dim4(16, 16),
        dim4(17, 16), dim4(16, 17), dim4(24, 32, 2), dim4(32, 24, 2, 2)};

    for (const dim4 &dims : shapes) {
        vector<TypeParam> input_values(static_cast<size_t>(dims.elements()));
        for (size_t i = 0; i < input_values.size(); ++i) {
            input_values[i] = transposeInputValue<TypeParam>(i);
        }
        checkTransposeReference(array(dims, input_values.data()), input_values);
    }
}

TYPED_TEST(TransposeTiled, OffsetAndPaddedRows) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    af_backend active_backend;
    ASSERT_SUCCESS(af_get_active_backend(&active_backend));
    if (active_backend != AF_BACKEND_CPU) {
        GTEST_SKIP() << "CPU tiled-transpose regression";
    }

    const dim4 parent_dims(35, 29, 2, 2);
    vector<TypeParam> parent_values(
        static_cast<size_t>(parent_dims.elements()));
    for (size_t i = 0; i < parent_values.size(); ++i) {
        parent_values[i] = transposeInputValue<TypeParam>(i);
    }

    const array parent(parent_dims, parent_values.data());
    const array input = parent(seq(1, 32), seq(2, 25), span, span);
    vector<TypeParam> input_values(static_cast<size_t>(input.elements()));
    input.host(input_values.data());
    checkTransposeReference(input, input_values);

    const array reversed = parent(seq(1, 32), seq(25, 2, -1), span, span);
    vector<TypeParam> reversed_values(static_cast<size_t>(reversed.elements()));
    reversed.host(reversed_values.data());
    checkTransposeReference(reversed, reversed_values);
}

TYPED_TEST(TransposeTiled, ParallelBatchedShapes) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    af_backend active_backend;
    ASSERT_SUCCESS(af_get_active_backend(&active_backend));
    if (active_backend != AF_BACKEND_CPU) {
        GTEST_SKIP() << "CPU tiled-transpose regression";
    }

    for (const dim4 dims : {dim4(513, 257, 2, 2), dim4(512, 256, 2, 2)}) {
        vector<TypeParam> input_values(static_cast<size_t>(dims.elements()));
        for (size_t i = 0; i < input_values.size(); ++i) {
            input_values[i] = transposeInputValue<TypeParam>(i);
        }
        checkTransposeReference(array(dims, input_values.data()), input_values);
    }
}

template<typename T>
class TransposeTiledComplex : public ::testing::Test {};

using TiledComplexTypes = ::testing::Types<cfloat, cdouble>;
TYPED_TEST_SUITE(TransposeTiledComplex, TiledComplexTypes);

TYPED_TEST(TransposeTiledComplex, ConjugateSerialAndParallel) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    af_backend active_backend;
    ASSERT_SUCCESS(af_get_active_backend(&active_backend));
    if (active_backend != AF_BACKEND_CPU) {
        GTEST_SKIP() << "CPU tiled-transpose regression";
    }

    for (const dim4 dims : {dim4(32, 24, 2), dim4(513, 257, 2)}) {
        vector<TypeParam> input_values(static_cast<size_t>(dims.elements()));
        for (size_t i = 0; i < input_values.size(); ++i) {
            input_values[i] = transposeInputValue<TypeParam>(i);
        }
        checkTransposeReference(array(dims, input_values.data()), input_values,
                                true);
    }
}

template<typename Value, typename Bits>
Value transposeValueFromBits(Bits bits) {
    static_assert(sizeof(Value) == sizeof(Bits), "bit width mismatch");
    Value value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

template<typename Bits, typename Value>
Bits transposeValueBits(Value value) {
    static_assert(sizeof(Value) == sizeof(Bits), "bit width mismatch");
    Bits bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template<typename Value, typename Bits, size_t PatternCount>
void checkRealTransposeBitPatterns(
    const std::array<Bits, PatternCount> &patterns) {
    constexpr dim_t side = 16;
    const dim4 dims(side, side);
    vector<Value> input_values(static_cast<size_t>(dims.elements()));
    vector<Value> expected(input_values.size());
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] =
            transposeValueFromBits<Value>(patterns[i % PatternCount]);
    }
    for (dim_t y = 0; y < side; ++y) {
        for (dim_t x = 0; x < side; ++x) {
            expected[static_cast<size_t>(y + side * x)] =
                input_values[static_cast<size_t>(x + side * y)];
        }
    }

    const array output = transpose(array(dims, input_values.data()));
    vector<Value> actual(input_values.size());
    output.host(actual.data());
    ASSERT_EQ(0, std::memcmp(expected.data(), actual.data(),
                             expected.size() * sizeof(Value)));
}

TEST(TransposeTiled, PreservesFloatingPointBits) {
    af_backend active_backend;
    ASSERT_SUCCESS(af_get_active_backend(&active_backend));
    if (active_backend != AF_BACKEND_CPU) {
        GTEST_SKIP() << "CPU bitwise-transpose regression";
    }

    constexpr std::array<std::uint32_t, 8> float_patterns = {
        0x00000000U, 0x80000000U, 0x7F800000U, 0xFF800000U,
        0x7FC12345U, 0xFFC54321U, 0x3F800000U, 0xBF000000U};
    constexpr std::array<std::uint64_t, 8> double_patterns = {
        0x0000000000000000ULL, 0x8000000000000000ULL, 0x7FF0000000000000ULL,
        0xFFF0000000000000ULL, 0x7FF8123456789ABCULL, 0xFFF8543210FEDCBAULL,
        0x3FF0000000000000ULL, 0xBFE0000000000000ULL};
    checkRealTransposeBitPatterns<float>(float_patterns);
    checkRealTransposeBitPatterns<double>(double_patterns);
}

template<typename Complex, typename Value, typename Bits>
void checkComplexConjugateBits(Bits sign_bit) {
    static_assert(sizeof(Complex) == 2 * sizeof(Value));
    constexpr dim_t side = 16;
    const dim4 dims(side, side);
    vector<Complex> input_values(static_cast<size_t>(dims.elements()));
    vector<Complex> expected(input_values.size());
    for (size_t i = 0; i < input_values.size(); ++i) {
        const Bits real_bits =
            static_cast<Bits>(i * 0x101U) ^ (i % 2 == 0 ? sign_bit : 0);
        const Bits imag_bits =
            static_cast<Bits>(i * 0x10001U) ^ (i % 3 == 0 ? sign_bit : 0);
        input_values[i] = Complex(transposeValueFromBits<Value>(real_bits),
                                  transposeValueFromBits<Value>(imag_bits));
    }
    for (dim_t y = 0; y < side; ++y) {
        for (dim_t x = 0; x < side; ++x) {
            const Complex value =
                input_values[static_cast<size_t>(x + side * y)];
            expected[static_cast<size_t>(y + side * x)] =
                Complex(value.real,
                        transposeValueFromBits<Value>(
                            transposeValueBits<Bits>(value.imag) ^ sign_bit));
        }
    }

    const array output = transpose(array(dims, input_values.data()), true);
    vector<Complex> actual(input_values.size());
    output.host(actual.data());
    ASSERT_EQ(0, std::memcmp(expected.data(), actual.data(),
                             expected.size() * sizeof(Complex)));
}

TEST(TransposeTiled, ConjugateFlipsOnlyImaginarySign) {
    af_backend active_backend;
    ASSERT_SUCCESS(af_get_active_backend(&active_backend));
    if (active_backend != AF_BACKEND_CPU) {
        GTEST_SKIP() << "CPU bitwise-transpose regression";
    }

    checkComplexConjugateBits<cfloat, float>(0x80000000U);
    checkComplexConjugateBits<cdouble, double>(0x8000000000000000ULL);
}

////////////////////////////////////// CPP //////////////////////////////////
//
template<typename T>
void trsCPPTest(string pFileName) {
    vector<dim4> numDims;

    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pFileName, numDims, in, tests);
    dim4 dims = numDims[0];

    SUPPORTED_TYPE_CHECK(T);

    array input(dims, &(in[0].front()));
    array output = transpose(input);

    T *outData = new T[dims.elements()];
    output.host((void *)outData);

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<T> currGoldBar = tests[testIter];
        size_t nElems         = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << endl;
        }
    }

    // cleanup
    delete[] outData;
}

TEST(Transpose, CPP_f64) {
    trsCPPTest<double>(string(TEST_DIR "/transpose/rectangle_batch2.test"));
}

TEST(Transpose, CPP_f32) {
    trsCPPTest<float>(string(TEST_DIR "/transpose/rectangle_batch2.test"));
}

template<typename T>
void trsCPPConjTest(dim_t d0, dim_t d1 = 1, dim_t d2 = 1, dim_t d3 = 1) {
    vector<dim4> numDims;

    dim4 dims(d0, d1, d2, d3);

    SUPPORTED_TYPE_CHECK(T);

    array input    = randu(dims, (af_dtype)dtype_traits<T>::af_type);
    array output_t = transpose(input, false);
    array output_c = transpose(input, true);

    T *tData = new T[dims.elements()];
    T *cData = new T[dims.elements()];
    output_t.host((void *)tData);
    output_c.host((void *)cData);

    size_t nElems = dims.elements();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(real(tData[elIter]), real(cData[elIter]), 1e-6)
            << "at: " << elIter << endl;
        ASSERT_NEAR(-imag(tData[elIter]), imag(cData[elIter]), 1e-6)
            << "at: " << elIter << endl;
    }

    // cleanup
    delete[] tData;
    delete[] cData;
}

TEST(Transpose, CPP_c32_CONJ40x40) { trsCPPConjTest<cfloat>(40, 40); }

TEST(Transpose, CPP_c32_CONJ2000x1) { trsCPPConjTest<cfloat>(2000); }

TEST(Transpose, CPP_c32_CONJ20x20x5) { trsCPPConjTest<cfloat>(20, 20, 5); }

TEST(Transpose, OddBatchedDimensions) {
    const dim4 dims(257, 67, 3, 5);
    const dim4 outputDims(dims[1], dims[0], dims[2], dims[3]);
    vector<cfloat> inputData(dims.elements());
    vector<cfloat> transposeGold(dims.elements());
    vector<cfloat> conjugateGold(dims.elements());

    for (dim_t w = 0; w < dims[3]; ++w) {
        for (dim_t z = 0; z < dims[2]; ++z) {
            for (dim_t y = 0; y < dims[1]; ++y) {
                for (dim_t x = 0; x < dims[0]; ++x) {
                    const size_t inputIdx =
                        x + dims[0] * (y + dims[1] * (z + dims[2] * w));
                    const size_t outputIdx =
                        y + dims[1] * (x + dims[0] * (z + dims[2] * w));
                    const cfloat value(
                        static_cast<float>((x + 3 * y + 5 * z + 7 * w) % 97),
                        static_cast<float>((2 * x + y + 11 * z + 13 * w) % 89) -
                            44.0f);
                    inputData[inputIdx]      = value;
                    transposeGold[outputIdx] = value;
                    conjugateGold[outputIdx] = cfloat(value.real, -value.imag);
                }
            }
        }
    }

    const array input(dims, inputData.data());
    const array output          = transpose(input, false);
    const array conjugateOutput = transpose(input, true);

    ASSERT_VEC_ARRAY_EQ(transposeGold, outputDims, output);
    ASSERT_VEC_ARRAY_EQ(conjugateGold, outputDims, conjugateOutput);
}

TEST(Transpose, MaxDim) {
    const size_t largeDim = 65535 * 33 + 1;

    array input  = range(dim4(2, largeDim, 1, 1));
    array gold   = range(dim4(largeDim, 2, 1, 1), 1);
    array output = transpose(input);

    ASSERT_EQ(output.dims(0), (int)largeDim);
    ASSERT_EQ(output.dims(1), 2);
    ASSERT_ARRAYS_EQ(gold, output);

    input  = range(dim4(2, 5, 1, largeDim));
    gold   = range(dim4(5, 2, 1, largeDim), 1);
    output = transpose(input);

    ASSERT_ARRAYS_EQ(gold, output);
}

TEST(Transpose, GFOR) {
    using af::constant;
    using af::max;
    using af::seq;
    using af::span;

    dim4 dims = dim4(100, 100, 3);
    array A   = round(100 * randu(dims));
    array B   = constant(0, 100, 100, 3);

    gfor(seq ii, 3) { B(span, span, ii) = A(span, span, ii).T(); }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = A(span, span, ii).T();
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(Transpose, SNIPPET_blas_func_transpose) {
    // clang-format off
    //! [ex_blas_func_transpose]
    //!
    // Create a, a 2x3 array
    array a = iota(dim4(2, 3));    // a = [0, 2, 4
                                   //      1, 3, 5]

    // Create b, the transpose of a
    array b = transpose(a);        // b = [0, 1,
                                   //      2, 3,
                                   //      4, 5]

    //! [ex_blas_func_transpose]
    // clang-format on

    using std::vector;
    vector<float> gold_b{0, 2, 4, 1, 3, 5};

    ASSERT_VEC_ARRAY_EQ(gold_b, b.dims(), b);
}
