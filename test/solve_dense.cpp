/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// NOTE: Tests are known to fail on OSX when utilizing the CPU and OpenCL
// backends for sizes larger than 128x128 or more. You can read more about it on
// issue https://github.com/arrayfire/arrayfire/issues/1617

#include <gtest/gtest.h>

#include <testHelpers.hpp>
#include <af/algorithm.h>
#include <af/arith.h>
#include <af/blas.h>
#include <af/defines.h>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/lapack.h>
#include <af/traits.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::deviceGC;
using af::dim4;
using af::dtype_traits;
using af::setDevice;
using af::sum;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
void solveTester(const int m, const int n, const int k, const int b, double eps,
                 int targetDevice = -1) {
    if (targetDevice >= 0) setDevice(targetDevice);

    deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

#if 1
    array A  = cpu_randu<T>(dim4(m, n, b));
    array X0 = cpu_randu<T>(dim4(n, k, b));
#else
    array A  = randu(m, n, (dtype)dtype_traits<T>::af_type);
    array X0 = randu(n, k, (dtype)dtype_traits<T>::af_type);
#endif
    array B0 = matmul(A, X0);

    //! [ex_solve]
    array X1 = solve(A, B0);
    //! [ex_solve]

    //! [ex_solve_recon]
    array B1 = matmul(A, X1);
    //! [ex_solve_recon]

    ASSERT_NEAR(
        0,
        sum<typename dtype_traits<T>::base_type>(abs(real(B0 - B1))) / (m * k),
        eps);
    ASSERT_NEAR(
        0,
        sum<typename dtype_traits<T>::base_type>(abs(imag(B0 - B1))) / (m * k),
        eps);
}

#if defined(AF_CUDA)
template<typename T>
void solveFourDimensionalBatchGridTailTester(double tolerance) {
    deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

    // CUDA builds the batched pointer arrays with 256 threads per block.
    // Unequal batch dimensions exercise both strides without allowing a z/w
    // swap to become an output-preserving permutation. The 272 batches leave
    // a 16-entry tail in the second pointer-setup block.
    const dim_t n      = 4;
    const dim_t nrhs   = 2;
    const dim_t batchZ = 17;
    const dim_t batchW = 16;
    const dim4 matrixDims(n, n, batchZ, batchW);
    const dim4 solutionDims(n, nrhs, batchZ, batchW);

    vector<T> matrices(matrixDims.elements(), T(0));
    vector<T> rhs(solutionDims.elements(), T(0));
    vector<T> expected(solutionDims.elements(), T(0));

    using Base = typename dtype_traits<T>::base_type;
    for (dim_t w = 0; w < batchW; ++w) {
        for (dim_t z = 0; z < batchZ; ++z) {
            const dim_t batch = z + batchZ * w;
            for (dim_t row = 0; row < n; ++row) {
                const Base diagonalValue = Base(2) +
                                           Base(batch + 1) / Base(1024) +
                                           Base(row + 1) / Base(16);
                const T diagonal(diagonalValue);
                const dim_t matrixOffset = batch * n * n + row + row * n;
                matrices[matrixOffset]   = diagonal;

                for (dim_t column = 0; column < nrhs; ++column) {
                    const Base solutionValue =
                        Base(0.25) + Base(batch + 1) / Base(512) +
                        Base(row + 1) / Base(32) + Base(column + 1) / Base(64);
                    const dim_t solutionOffset =
                        batch * n * nrhs + row + column * n;
                    expected[solutionOffset] = T(solutionValue);
                    rhs[solutionOffset] = diagonal * expected[solutionOffset];
                }
            }
        }
    }

    array A(matrixDims, matrices.data());
    array B(solutionDims, rhs.data());
    array expectedSolution(solutionDims, expected.data());
    array actualSolution = solve(A, B);

    ASSERT_ARRAYS_NEAR(expectedSolution, actualSolution, tolerance);
}
#endif

template<typename T>
void solveLUTester(const int n, const int k, double eps,
                   int targetDevice = -1) {
    if (targetDevice >= 0) setDevice(targetDevice);

    deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

#if 1
    array A  = cpu_randu<T>(dim4(n, n));
    array X0 = cpu_randu<T>(dim4(n, k));
#else
    array A  = randu(n, n, (dtype)dtype_traits<T>::af_type);
    array X0 = randu(n, k, (dtype)dtype_traits<T>::af_type);
#endif
    array B0 = matmul(A, X0);

    //! [ex_solve_lu]
    array A_lu, pivot;
    lu(A_lu, pivot, A);
    array X1 = solveLU(A_lu, pivot, B0);
    //! [ex_solve_lu]

    array B1 = matmul(A, X1);

    ASSERT_NEAR(
        0,
        sum<typename dtype_traits<T>::base_type>(abs(real(B0 - B1))) / (n * k),
        eps);
    ASSERT_NEAR(
        0,
        sum<typename dtype_traits<T>::base_type>(abs(imag(B0 - B1))) / (n * k),
        eps);
}

template<typename T>
void solveTriangleTester(const int n, const int k, bool is_upper, double eps,
                         int targetDevice = -1) {
    if (targetDevice >= 0) setDevice(targetDevice);

    deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

#if 1
    array A  = cpu_randu<T>(dim4(n, n));
    array X0 = cpu_randu<T>(dim4(n, k));
#else
    array A  = randu(n, n, (dtype)dtype_traits<T>::af_type);
    array X0 = randu(n, k, (dtype)dtype_traits<T>::af_type);
#endif

    array L, U, pivot;
    lu(L, U, pivot, A);

    array AT = is_upper ? U : L;
    array B0 = matmul(AT, X0);
    array X1;

    if (is_upper) {
        //! [ex_solve_upper]
        array X = solve(AT, B0, AF_MAT_UPPER);
        //! [ex_solve_upper]

        X1 = X;
    } else {
        //! [ex_solve_lower]
        array X = solve(AT, B0, AF_MAT_LOWER);
        //! [ex_solve_lower]

        X1 = X;
    }

    array B1 = matmul(AT, X1);

    ASSERT_NEAR(
        0,
        sum<typename dtype_traits<T>::base_type>(af::abs(real(B0 - B1))) /
            (n * k),
        eps);
    ASSERT_NEAR(
        0,
        sum<typename dtype_traits<T>::base_type>(af::abs(imag(B0 - B1))) /
            (n * k),
        eps);
}

template<typename T>
class Solve : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_SUITE(Solve, TestTypes);

template<typename T>
double eps();

template<>
double eps<float>() {
    return 0.01f;
}

template<>
double eps<double>() {
    return 1e-5;
}

template<>
double eps<cfloat>() {
    return 0.015f;
}

template<>
double eps<cdouble>() {
    return 1e-5;
}

TYPED_TEST(Solve, Square) {
    solveTester<TypeParam>(100, 100, 10, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, SquareMultipleOfTwo) {
    solveTester<TypeParam>(96, 96, 16, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, SquareLarge) {
    solveTester<TypeParam>(1000, 1000, 10, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, SquareMultipleOfTwoLarge) {
    solveTester<TypeParam>(2048, 2048, 32, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, SquareBatch) {
    solveTester<TypeParam>(100, 100, 10, 10, eps<TypeParam>());
}

TYPED_TEST(Solve, SquareMultipleOfTwoBatch) {
    solveTester<TypeParam>(96, 96, 16, 10, eps<TypeParam>());
}

#if defined(AF_CUDA)
TYPED_TEST(Solve, SquareFourDimensionalBatchGridTail) {
    solveFourDimensionalBatchGridTailTester<TypeParam>(eps<TypeParam>());
}
#endif

TYPED_TEST(Solve, SquareLargeBatch) {
    solveTester<TypeParam>(1000, 1000, 10, 10, eps<TypeParam>());
}

TYPED_TEST(Solve, SquareMultipleOfTwoLargeBatch) {
    solveTester<TypeParam>(2048, 2048, 32, 10, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresUnderDetermined) {
    solveTester<TypeParam>(80, 100, 20, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresUnderDeterminedMultipleOfTwo) {
    solveTester<TypeParam>(96, 128, 40, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresUnderDeterminedLarge) {
    solveTester<TypeParam>(800, 1000, 200, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresUnderDeterminedMultipleOfTwoLarge) {
    solveTester<TypeParam>(1536, 2048, 400, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresOverDetermined) {
    solveTester<TypeParam>(80, 60, 20, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresOverDeterminedMultipleOfTwo) {
    solveTester<TypeParam>(96, 64, 1, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresOverDeterminedLarge) {
    solveTester<TypeParam>(800, 600, 64, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresOverDeterminedMultipleOfTwoLarge) {
    solveTester<TypeParam>(1536, 1024, 1, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, LU) { solveLUTester<TypeParam>(100, 10, eps<TypeParam>()); }

TYPED_TEST(Solve, LUMultipleOfTwo) {
    solveLUTester<TypeParam>(96, 64, eps<TypeParam>());
}

TYPED_TEST(Solve, LULarge) {
    solveLUTester<TypeParam>(1000, 100, eps<TypeParam>());
}

TYPED_TEST(Solve, LUMultipleOfTwoLarge) {
    solveLUTester<TypeParam>(2048, 512, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleUpper) {
    solveTriangleTester<TypeParam>(100, 10, true, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleUpperMultipleOfTwo) {
    solveTriangleTester<TypeParam>(96, 64, true, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleUpperLarge) {
    solveTriangleTester<TypeParam>(1000, 100, true, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleUpperMultipleOfTwoLarge) {
    solveTriangleTester<TypeParam>(2048, 512, true, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleLower) {
    solveTriangleTester<TypeParam>(100, 10, false, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleLowerMultipleOfTwo) {
    solveTriangleTester<TypeParam>(96, 64, false, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleLowerLarge) {
    solveTriangleTester<TypeParam>(1000, 100, false, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleLowerMultipleOfTwoLarge) {
    solveTriangleTester<TypeParam>(2048, 512, false, eps<TypeParam>());
}

#if !defined(AF_OPENCL)
int nextTargetDeviceId() {
    static int nextId = 0;
    return nextId++;
}

#define SOLVE_LU_TESTS_THREADING(T, eps)                              \
    tests.emplace_back(solveLUTester<T>, 1000, 100, eps,              \
                       nextTargetDeviceId() % numDevices);            \
    tests.emplace_back(solveTriangleTester<T>, 1000, 100, true, eps,  \
                       nextTargetDeviceId() % numDevices);            \
    tests.emplace_back(solveTriangleTester<T>, 1000, 100, false, eps, \
                       nextTargetDeviceId() % numDevices);            \
    tests.emplace_back(solveTester<T>, 1000, 1000, 100, 1, eps,       \
                       nextTargetDeviceId() % numDevices);            \
    tests.emplace_back(solveTester<T>, 800, 1000, 200, 1, eps,        \
                       nextTargetDeviceId() % numDevices);            \
    tests.emplace_back(solveTester<T>, 800, 600, 64, 1, eps,          \
                       nextTargetDeviceId() % numDevices);

TEST(Solve, Threading) {
    cleanSlate();  // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 0;
    ASSERT_SUCCESS(af_get_device_count(&numDevices));
    ASSERT_EQ(true, numDevices > 0);

    SOLVE_LU_TESTS_THREADING(float, 0.01);
    SOLVE_LU_TESTS_THREADING(cfloat, 0.01);
    if (noDoubleTests(f64)) {
        SOLVE_LU_TESTS_THREADING(double, 1E-5);
        SOLVE_LU_TESTS_THREADING(cdouble, 1E-5);
    }

    for (size_t testId = 0; testId < tests.size(); ++testId)
        if (tests[testId].joinable()) tests[testId].join();
}

#undef SOLVE_LU_TESTS_THREADING
#endif
#undef SOLVE_TESTS
