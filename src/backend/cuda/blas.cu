/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>

#include <arith.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <cublas.hpp>
#include <cublas_v2.h>
#include <cudaDataType.hpp>
#include <cuda_runtime.h>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <reduce.hpp>
#include <tile.hpp>
#include <transpose.hpp>
#include <types.hpp>

#include <cassert>
#include <functional>
#include <stdexcept>
#include <string>

using arrayfire::common::half;
using arrayfire::common::kernel_type;
using std::is_same;

namespace arrayfire {
namespace cuda {

namespace {

template<typename Ti, typename To>
__global__ void setBatchedGemmPointers(
    const Ti **lPtrs, const Ti **rPtrs, To **oPtrs, const Ti *lhs,
    const Ti *rhs, To *out, const dim_t lStride2, const dim_t lStride3,
    const dim_t rStride2, const dim_t rStride3, const dim_t oStride2,
    const dim_t oStride3, const int batchDim2, const int batchSize) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batchSize) { return; }

    const int z = id % batchDim2;
    const int w = id / batchDim2;
    lPtrs[id]   = lhs + z * lStride2 + w * lStride3;
    rPtrs[id]   = rhs + z * rStride2 + w * rStride3;
    oPtrs[id]   = out + z * oStride2 + w * oStride3;
}

}  // namespace

cublasOperation_t toCblasTranspose(af_mat_prop opt) {
    cublasOperation_t out = CUBLAS_OP_N;
    switch (opt) {
        case AF_MAT_NONE: out = CUBLAS_OP_N; break;
        case AF_MAT_TRANS: out = CUBLAS_OP_T; break;
        case AF_MAT_CTRANS: out = CUBLAS_OP_C; break;
        default: AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

template<typename T>
using gemm_func_def = std::function<cublasStatus_t(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const T *, const T *, int, const T *, int, const T *, T *, int)>;

template<typename T>
using gemmBatched_func_def = std::function<cublasStatus_t(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const T *, const T **, int, const T **, int, const T *, T **, int, int)>;

template<typename T>
using trsm_func_def = std::function<cublasStatus_t(
    cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
    cublasDiagType_t, int, int, const T *, const T *, int, T *, int)>;

#define BLAS_FUNC_DEF(FUNC) \
    template<typename T>    \
    FUNC##_func_def<T> FUNC##_func();

#define BLAS_FUNC(FUNC, TYPE, PREFIX)           \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &cublas##PREFIX##FUNC;           \
    }

BLAS_FUNC_DEF(gemm)
BLAS_FUNC(gemm, float, S)
BLAS_FUNC(gemm, cfloat, C)
BLAS_FUNC(gemm, double, D)
BLAS_FUNC(gemm, cdouble, Z)
BLAS_FUNC(gemm, __half, H)

BLAS_FUNC_DEF(gemmBatched)
BLAS_FUNC(gemmBatched, float, S)
BLAS_FUNC(gemmBatched, cfloat, C)
BLAS_FUNC(gemmBatched, double, D)
BLAS_FUNC(gemmBatched, cdouble, Z)
BLAS_FUNC(gemmBatched, __half, H)

template<>
gemm_func_def<schar> gemm_func<schar>() {
    TYPE_ERROR(3, af_dtype::s8);
    return gemm_func_def<schar>();
}
template<>
gemmBatched_func_def<schar> gemmBatched_func<schar>() {
    TYPE_ERROR(3, af_dtype::s8);
    return gemmBatched_func_def<schar>();
}

BLAS_FUNC_DEF(trsm)
BLAS_FUNC(trsm, float, S)
BLAS_FUNC(trsm, cfloat, C)
BLAS_FUNC(trsm, double, D)
BLAS_FUNC(trsm, cdouble, Z)

#undef BLAS_FUNC
#undef BLAS_FUNC_DEF

template<typename T, bool conjugate>
struct dot_func_def_t {
    typedef cublasStatus_t (*dot_func_def)(cublasHandle_t, int, const T *, int,
                                           const T *, int, T *);
};

#define BLAS_FUNC_DEF(FUNC)              \
    template<typename T, bool conjugate> \
    typename FUNC##_func_def_t<T, conjugate>::FUNC##_func_def FUNC##_func();

#define BLAS_FUNC(FUNC, TYPE, CONJUGATE, PREFIX)                       \
    template<>                                                         \
    typename FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def       \
        FUNC##_func<TYPE, CONJUGATE>() {                               \
        return (FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def) & \
               cublas##PREFIX##FUNC;                                   \
    }

BLAS_FUNC_DEF(dot)
BLAS_FUNC(dot, float, true, S)
BLAS_FUNC(dot, double, true, D)
BLAS_FUNC(dot, float, false, S)
BLAS_FUNC(dot, double, false, D)

#undef BLAS_FUNC

#define BLAS_FUNC(FUNC, TYPE, CONJUGATE, PREFIX, SUFFIX)               \
    template<>                                                         \
    typename FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def       \
        FUNC##_func<TYPE, CONJUGATE>() {                               \
        return (FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def) & \
               cublas##PREFIX##FUNC##SUFFIX;                           \
    }

BLAS_FUNC_DEF(dot)
BLAS_FUNC(dot, cfloat, true, C, c)
BLAS_FUNC(dot, cdouble, true, Z, c)
BLAS_FUNC(dot, cfloat, false, C, u)
BLAS_FUNC(dot, cdouble, false, Z, u)

#undef BLAS_FUNC
#undef BLAS_FUNC_DEF

template<typename T>
cublasGemmAlgo_t selectGEMMAlgorithm() {
    return CUBLAS_GEMM_DEFAULT;
}

template<>
cublasGemmAlgo_t selectGEMMAlgorithm<common::half>() {
    auto dev              = getDeviceProp(getActiveDeviceId());
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    if (dev.major >= 7) { algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP; }
    return algo;
}

template<>
cublasGemmAlgo_t selectGEMMAlgorithm<__half>() {
    return selectGEMMAlgorithm<common::half>();
}

template<typename Ti, typename To = Ti>
cublasStatus_t gemmDispatch(BlasHandle handle, cublasOperation_t lOpts,
                            cublasOperation_t rOpts, int M, int N, int K,
                            const To *alpha, const Array<Ti> &lhs,
                            dim_t lStride, const Array<Ti> &rhs, dim_t rStride,
                            const To *beta, Array<To> &out, dim_t oleading) {
    auto prop = getDeviceProp(getActiveDeviceId());
#if __CUDACC_VER_MAJOR__ >= 10
    if (prop.major > 3 && __CUDACC_VER_MAJOR__ >= 10) {
        return cublasGemmEx(
            handle, lOpts, rOpts, M, N, K, alpha, lhs.get(), getType<Ti>(),
            lStride, rhs.get(), getType<Ti>(), rStride, beta, out.get(),
            getType<To>(), out.strides()[1],
            getComputeType<To>(),  // Compute type

            // NOTE: When using the CUBLAS_GEMM_DEFAULT_TENSOR_OP algorithm
            // for the cublasGemm*Ex functions, the performance of the
            // fp32 numbers seem to increase dramatically. Their numerical
            // accuracy is also different compared to regular gemm fuctions.
            // The CUBLAS_GEMM_DEFAULT algorithm selection does not experience
            // this change. Does this imply that the TENSOR_OP function
            // performs the computation in fp16 bit even when the compute
            // type is CUDA_R_32F?
            selectGEMMAlgorithm<Ti>());
    } else {
#endif
        using Nt = typename common::kernel_type<Ti>::native;
        return gemm_func<Nt>()(handle, lOpts, rOpts, M, N, K, (Nt *)alpha,
                               (Nt *)lhs.get(), lStride, (Nt *)rhs.get(),
                               rStride, (Nt *)beta, (Nt *)out.get(), oleading);

#if __CUDACC_VER_MAJOR__ >= 10
    }
#endif
}

template<typename Ti, typename To = Ti>
cublasStatus_t gemmBatchedDispatch(BlasHandle handle, cublasOperation_t lOpts,
                                   cublasOperation_t rOpts, int M, int N, int K,
                                   const To *alpha, const Ti **lptrs,
                                   int lStrides, const Ti **rptrs, int rStrides,
                                   const To *beta, To **optrs, int oStrides,
                                   int batchSize) {
    auto prop = getDeviceProp(getActiveDeviceId());
#if __CUDACC_VER_MAJOR__ >= 10
    if (prop.major > 3) {
        return cublasGemmBatchedEx(
            handle, lOpts, rOpts, M, N, K, alpha, (const void **)lptrs,
            getType<Ti>(), lStrides, (const void **)rptrs, getType<Ti>(),
            rStrides, beta, (void **)optrs, getType<To>(), oStrides, batchSize,
            getComputeType<To>(),  // compute type
            // NOTE: When using the CUBLAS_GEMM_DEFAULT_TENSOR_OP algorithm
            // for the cublasGemm*Ex functions, the performance of the
            // fp32 numbers seem to increase dramatically. Their numerical
            // accuracy is also different compared to regular gemm fuctions.
            // The CUBLAS_GEMM_DEFAULT algorithm selection does not experience
            // this change. Does this imply that the TENSOR_OP function
            // performs the computation in fp16 bit even when the compute
            // type is CUDA_R_32F?
            selectGEMMAlgorithm<Ti>());
    } else {
#endif
        using Nt = typename common::kernel_type<Ti>::native;
        return gemmBatched_func<Nt>()(
            handle, lOpts, rOpts, M, N, K, (const Nt *)alpha,
            (const Nt **)lptrs, lStrides, (const Nt **)rptrs, rStrides,
            (const Nt *)beta, (Nt **)optrs, oStrides, batchSize);
#if __CUDACC_VER_MAJOR__ >= 10
    }
#endif
}

template<typename Ti, typename To>
void gemm(Array<To> &out, af_mat_prop optLhs, af_mat_prop optRhs,
          const To *alpha, const Array<Ti> &lhs, const Array<Ti> &rhs,
          const To *beta) {
    const cublasOperation_t lOpts = toCblasTranspose(optLhs);
    const cublasOperation_t rOpts = toCblasTranspose(optRhs);

    const int aRowDim = (lOpts == CUBLAS_OP_N) ? 0 : 1;
    const int aColDim = (lOpts == CUBLAS_OP_N) ? 1 : 0;
    const int bColDim = (rOpts == CUBLAS_OP_N) ? 1 : 0;

    const dim4 lDims = lhs.dims();
    const dim4 rDims = rhs.dims();
    const int M      = lDims[aRowDim];
    const int N      = rDims[bColDim];
    const int K      = lDims[aColDim];
    const dim4 oDims = out.dims();

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();
    dim4 oStrides = out.strides();

    if (oDims.ndims() <= 2) {
        CUBLAS_CHECK((gemmDispatch<Ti, To>(
            blasHandle(), lOpts, rOpts, M, N, K, alpha, lhs, lStrides[1], rhs,
            rStrides[1], beta, out, oStrides[1])));
    } else {
        const int batchSize = oDims[2] * oDims[3];
        const size_t pointerArrayBytes =
            static_cast<size_t>(batchSize) * sizeof(void *);
        auto pointerStorage  = memAlloc<uchar>(3 * pointerArrayBytes);
        uchar *const storage = pointerStorage.get();

        auto d_lptrs = reinterpret_cast<const Ti **>(storage);
        auto d_rptrs =
            reinterpret_cast<const Ti **>(storage + pointerArrayBytes);
        auto d_optrs = reinterpret_cast<To **>(storage + 2 * pointerArrayBytes);

        const dim_t lStride2 = lDims[2] == oDims[2] ? lStrides[2] : dim_t{0};
        const dim_t lStride3 = lDims[3] == oDims[3] ? lStrides[3] : dim_t{0};
        const dim_t rStride2 = rDims[2] == oDims[2] ? rStrides[2] : dim_t{0};
        const dim_t rStride3 = rDims[3] == oDims[3] ? rStrides[3] : dim_t{0};

        constexpr unsigned THREADS = 256;
        const dim3 threads(THREADS);
        const dim3 blocks((batchSize + THREADS - 1) / THREADS);
        CUDA_LAUNCH((setBatchedGemmPointers<Ti, To>), blocks, threads, d_lptrs,
                    d_rptrs, d_optrs, lhs.get(), rhs.get(), out.get(), lStride2,
                    lStride3, rStride2, rStride3, oStrides[2], oStrides[3],
                    oDims[2], batchSize);
        POST_LAUNCH_CHECK();

        CUBLAS_CHECK(gemmBatchedDispatch(
            blasHandle(), lOpts, rOpts, M, N, K, alpha, d_lptrs, lStrides[1],
            d_rptrs, rStrides[1], beta, d_optrs, oStrides[1], batchSize));
    }
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs) {
    auto lhs_ = (optLhs == AF_MAT_NONE ? lhs : conj<T>(lhs));
    auto rhs_ = (optRhs == AF_MAT_NONE ? rhs : conj<T>(rhs));
    auto temp = arithOp<T, af_mul_t>(lhs_, rhs_, lhs_.dims());
    return reduce<af_add_t, T, T>(temp, 0, false, 0);
}

template<typename T>
void trsm(const Array<T> &lhs, Array<T> &rhs, af_mat_prop trans, bool is_upper,
          bool is_left, bool is_unit) {
    // dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M      = rDims[0];
    int N      = rDims[1];

    T alpha = scalar<T>(1);

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();

    CUBLAS_CHECK(trsm_func<T>()(
        blasHandle(), is_left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
        is_upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
        toCblasTranspose(trans),
        is_unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, M, N, &alpha,
        lhs.get(), lStrides[1], rhs.get(), rStrides[1]));
}

#define INSTANTIATE_GEMM(TYPE, OUTTYPE)                                      \
    template void gemm<TYPE>(Array<OUTTYPE> & out, af_mat_prop optLhs,       \
                             af_mat_prop optRhs, const OUTTYPE *alpha,       \
                             const Array<TYPE> &lhs, const Array<TYPE> &rhs, \
                             const OUTTYPE *beta);

INSTANTIATE_GEMM(float, float)
INSTANTIATE_GEMM(cfloat, cfloat)
INSTANTIATE_GEMM(double, double)
INSTANTIATE_GEMM(cdouble, cdouble)
INSTANTIATE_GEMM(half, half)
INSTANTIATE_GEMM(schar, float)

#define INSTANTIATE_DOT(TYPE)                                                  \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs,                     \
                                   const Array<TYPE> &rhs, af_mat_prop optLhs, \
                                   af_mat_prop optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)
INSTANTIATE_DOT(cfloat)
INSTANTIATE_DOT(cdouble)
INSTANTIATE_DOT(half)

#define INSTANTIATE_TRSM(TYPE)                                               \
    template void trsm<TYPE>(const Array<TYPE> &lhs, Array<TYPE> &rhs,       \
                             af_mat_prop trans, bool is_upper, bool is_left, \
                             bool is_unit);

INSTANTIATE_TRSM(float)
INSTANTIATE_TRSM(cfloat)
INSTANTIATE_TRSM(double)
INSTANTIATE_TRSM(cdouble)

}  // namespace cuda
}  // namespace arrayfire
