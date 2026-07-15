/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/reduce_dim_avx2.hpp>

#if (defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__) ||                       \
     (defined(_MSC_VER) && defined(AF_CPU_COMPILE_AVX2)))
#define AF_CPU_HAS_AVX2_INTRINSICS
#include <immintrin.h>
#endif

#if defined(AF_CPU_HAS_AVX2_INTRINSICS) && \
    (defined(__GNUC__) || defined(__clang__))
#define AF_CPU_AVX2_TARGET __attribute__((target("avx2")))
#else
#define AF_CPU_AVX2_TARGET
#endif

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace detail {

#if defined(AF_CPU_HAS_AVX2_INTRINSICS)

namespace {

constexpr size_t tile_bytes = 128;

enum class ReductionOperation { Sum, Product };

struct TileDescriptor {
    dim_t input_offset;
    dim_t output_offset;
    size_t elements;
};

template<typename T>
struct ReductionMetadata {
    dim_t output_dims[4];
    dim_t output_strides[4];
    dim_t input_strides[4];
    dim_t reduction_elements;
    size_t x_tiles;
};

template<typename T>
bool makeReductionMetadata(ReductionMetadata<T> &metadata, Param<T> out,
                           CParam<T> in, const int dim) noexcept {
    if (dim < 0 || dim > 3) { return false; }

    const af::dim4 idims    = in.dims();
    const af::dim4 odims    = out.dims();
    const af::dim4 ostrides = out.strides();
    const af::dim4 istrides = in.strides();
    if (odims[dim] != 1 || odims[0] <= 0 || odims[1] <= 0 || odims[2] <= 0 ||
        odims[3] <= 0) {
        return false;
    }

    for (int axis = 0; axis < 4; ++axis) {
        metadata.output_dims[axis]    = odims[axis];
        metadata.output_strides[axis] = ostrides[axis];
        metadata.input_strides[axis]  = istrides[axis];
    }
    metadata.reduction_elements = idims[dim];

    constexpr size_t tile_elements = tile_bytes / sizeof(T);
    const size_t output_x          = static_cast<size_t>(odims[0]);
    metadata.x_tiles = (output_x + tile_elements - 1) / tile_elements;
    return metadata.reduction_elements >= 0;
}

template<typename T>
inline bool describeTile(TileDescriptor &descriptor,
                         const ReductionMetadata<T> &metadata,
                         const size_t tile) noexcept {
    constexpr size_t tile_elements = tile_bytes / sizeof(T);
    const size_t output_x = static_cast<size_t>(metadata.output_dims[0]);

    const size_t x_tile = tile % metadata.x_tiles;
    size_t row          = tile / metadata.x_tiles;
    dim_t coordinate[4] = {static_cast<dim_t>(x_tile * tile_elements), 0, 0, 0};
    for (int axis = 1; axis < 4; ++axis) {
        const size_t axis_elements =
            static_cast<size_t>(metadata.output_dims[axis]);
        coordinate[axis] = static_cast<dim_t>(row % axis_elements);
        row /= axis_elements;
    }
    if (row != 0) { return false; }

    descriptor.elements =
        std::min(tile_elements, output_x - static_cast<size_t>(coordinate[0]));
    descriptor.input_offset = coordinate[0] * metadata.input_strides[0] +
                              coordinate[1] * metadata.input_strides[1] +
                              coordinate[2] * metadata.input_strides[2] +
                              coordinate[3] * metadata.input_strides[3];
    descriptor.output_offset = coordinate[0] * metadata.output_strides[0] +
                               coordinate[1] * metadata.output_strides[1] +
                               coordinate[2] * metadata.output_strides[2] +
                               coordinate[3] * metadata.output_strides[3];
    return true;
}

template<typename T>
struct RealVector;

template<>
struct RealVector<float> {
    using Type                       = __m256;
    static constexpr size_t elements = 8;

    template<ReductionOperation op>
    AF_CPU_AVX2_TARGET static Type initial() noexcept {
        if constexpr (op == ReductionOperation::Sum) {
            return _mm256_setzero_ps();
        } else {
            return _mm256_set1_ps(1.0F);
        }
    }

    AF_CPU_AVX2_TARGET static Type load(const float *pointer) noexcept {
        return _mm256_loadu_ps(pointer);
    }

    AF_CPU_AVX2_TARGET static void store(float *pointer, Type value) noexcept {
        _mm256_storeu_ps(pointer, value);
    }

    AF_CPU_AVX2_TARGET static Type replaceNaN(Type value,
                                              float replacement) noexcept {
        const Type unordered = _mm256_cmp_ps(value, value, _CMP_UNORD_Q);
        return _mm256_blendv_ps(value, _mm256_set1_ps(replacement), unordered);
    }

    template<ReductionOperation op>
    AF_CPU_AVX2_TARGET static Type combine(Type input,
                                           Type accumulator) noexcept {
        if constexpr (op == ReductionOperation::Sum) {
            return _mm256_add_ps(input, accumulator);
        } else {
            return _mm256_mul_ps(input, accumulator);
        }
    }
};

template<>
struct RealVector<double> {
    using Type                       = __m256d;
    static constexpr size_t elements = 4;

    template<ReductionOperation op>
    AF_CPU_AVX2_TARGET static Type initial() noexcept {
        if constexpr (op == ReductionOperation::Sum) {
            return _mm256_setzero_pd();
        } else {
            return _mm256_set1_pd(1.0);
        }
    }

    AF_CPU_AVX2_TARGET static Type load(const double *pointer) noexcept {
        return _mm256_loadu_pd(pointer);
    }

    AF_CPU_AVX2_TARGET static void store(double *pointer, Type value) noexcept {
        _mm256_storeu_pd(pointer, value);
    }

    AF_CPU_AVX2_TARGET static Type replaceNaN(Type value,
                                              double replacement) noexcept {
        const Type unordered = _mm256_cmp_pd(value, value, _CMP_UNORD_Q);
        return _mm256_blendv_pd(value, _mm256_set1_pd(replacement), unordered);
    }

    template<ReductionOperation op>
    AF_CPU_AVX2_TARGET static Type combine(Type input,
                                           Type accumulator) noexcept {
        if constexpr (op == ReductionOperation::Sum) {
            return _mm256_add_pd(input, accumulator);
        } else {
            return _mm256_mul_pd(input, accumulator);
        }
    }
};

template<typename T>
struct ComplexVector;

template<>
struct ComplexVector<cfloat> {
    using Type                       = __m256;
    using Real                       = float;
    static constexpr size_t elements = 4;

    AF_CPU_AVX2_TARGET static Type initial() noexcept {
        return _mm256_setzero_ps();
    }

    AF_CPU_AVX2_TARGET static Type load(const cfloat *pointer) noexcept {
        return _mm256_loadu_ps(reinterpret_cast<const float *>(pointer));
    }

    AF_CPU_AVX2_TARGET static void store(cfloat *pointer, Type value) noexcept {
        _mm256_storeu_ps(reinterpret_cast<float *>(pointer), value);
    }

    AF_CPU_AVX2_TARGET static Type replaceNaN(Type value,
                                              float replacement) noexcept {
        Type unordered = _mm256_cmp_ps(value, value, _CMP_UNORD_Q);
        unordered      = _mm256_or_ps(
                 unordered, _mm256_permute_ps(unordered, _MM_SHUFFLE(2, 3, 0, 1)));
        const Type replacement_pairs =
            _mm256_setr_ps(replacement, 0.0F, replacement, 0.0F, replacement,
                           0.0F, replacement, 0.0F);
        return _mm256_blendv_ps(value, replacement_pairs, unordered);
    }

    AF_CPU_AVX2_TARGET static Type combine(Type input,
                                           Type accumulator) noexcept {
        return _mm256_add_ps(input, accumulator);
    }
};

template<>
struct ComplexVector<cdouble> {
    using Type                       = __m256d;
    using Real                       = double;
    static constexpr size_t elements = 2;

    AF_CPU_AVX2_TARGET static Type initial() noexcept {
        return _mm256_setzero_pd();
    }

    AF_CPU_AVX2_TARGET static Type load(const cdouble *pointer) noexcept {
        return _mm256_loadu_pd(reinterpret_cast<const double *>(pointer));
    }

    AF_CPU_AVX2_TARGET static void store(cdouble *pointer,
                                         Type value) noexcept {
        _mm256_storeu_pd(reinterpret_cast<double *>(pointer), value);
    }

    AF_CPU_AVX2_TARGET static Type replaceNaN(Type value,
                                              double replacement) noexcept {
        Type unordered = _mm256_cmp_pd(value, value, _CMP_UNORD_Q);
        unordered = _mm256_or_pd(unordered, _mm256_permute_pd(unordered, 0x5));
        const Type replacement_pairs =
            _mm256_setr_pd(replacement, 0.0, replacement, 0.0);
        return _mm256_blendv_pd(value, replacement_pairs, unordered);
    }

    AF_CPU_AVX2_TARGET static Type combine(Type input,
                                           Type accumulator) noexcept {
        return _mm256_add_pd(input, accumulator);
    }
};

template<typename T, ReductionOperation op, size_t vector_count>
AF_CPU_AVX2_TARGET void reduceRealVectors(T *output, const T *input,
                                          const dim_t reduction_stride,
                                          const dim_t reduction_elements,
                                          const bool change_nan,
                                          const T nanval) noexcept {
    using Vector = RealVector<T>;
    typename Vector::Type accumulators[vector_count];
    for (size_t vector = 0; vector < vector_count; ++vector) {
        accumulators[vector] = Vector::template initial<op>();
    }

    for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
        const T *const row = input + reduced * reduction_stride;
        for (size_t vector = 0; vector < vector_count; ++vector) {
            typename Vector::Type value =
                Vector::load(row + vector * Vector::elements);
            if (change_nan) { value = Vector::replaceNaN(value, nanval); }
            accumulators[vector] =
                Vector::template combine<op>(value, accumulators[vector]);
        }
    }

    for (size_t vector = 0; vector < vector_count; ++vector) {
        Vector::store(output + vector * Vector::elements, accumulators[vector]);
    }
}

template<typename T, size_t vector_count>
AF_CPU_AVX2_TARGET void reduceComplexVectors(
    T *output, const T *input, const dim_t reduction_stride,
    const dim_t reduction_elements, const bool change_nan,
    const typename ComplexVector<T>::Real nanval) noexcept {
    using Vector = ComplexVector<T>;
    typename Vector::Type accumulators[vector_count];
    for (size_t vector = 0; vector < vector_count; ++vector) {
        accumulators[vector] = Vector::initial();
    }

    for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
        const T *const row = input + reduced * reduction_stride;
        for (size_t vector = 0; vector < vector_count; ++vector) {
            typename Vector::Type value =
                Vector::load(row + vector * Vector::elements);
            if (change_nan) { value = Vector::replaceNaN(value, nanval); }
            accumulators[vector] = Vector::combine(value, accumulators[vector]);
        }
    }

    for (size_t vector = 0; vector < vector_count; ++vector) {
        Vector::store(output + vector * Vector::elements, accumulators[vector]);
    }
}

template<typename T, ReductionOperation op>
T reduceRealScalar(const T *input, const dim_t reduction_stride,
                   const dim_t reduction_elements, const bool change_nan,
                   const T nanval) noexcept {
    T accumulator = op == ReductionOperation::Sum ? T(0) : T(1);
    for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
        T value = input[reduced * reduction_stride];
        if (change_nan && value != value) { value = nanval; }
        if constexpr (op == ReductionOperation::Sum) {
            accumulator = value + accumulator;
        } else {
            accumulator = value * accumulator;
        }
    }
    return accumulator;
}

template<typename T>
T reduceComplexScalar(const T *input, const dim_t reduction_stride,
                      const dim_t reduction_elements, const bool change_nan,
                      const typename ComplexVector<T>::Real nanval) noexcept {
    using Real = typename ComplexVector<T>::Real;
    T accumulator(Real(0), Real(0));
    for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
        T value = input[reduced * reduction_stride];
        if (change_nan && value != value) { value = T(nanval, Real(0)); }
        accumulator = value + accumulator;
    }
    return accumulator;
}

template<typename T, ReductionOperation op>
AF_CPU_AVX2_TARGET inline void reduceRealTile(
    T *const output_base, const T *const input_base,
    const ReductionMetadata<T> &metadata, const int dim,
    const TileDescriptor &tile, const bool change_nan,
    const double nanval) noexcept {
    T *const output        = output_base + tile.output_offset;
    const T *const input   = input_base + tile.input_offset;
    const T replacement    = static_cast<T>(nanval);
    size_t vector_elements = 0;

    if (metadata.output_strides[0] == 1 && metadata.input_strides[0] == 1) {
        using Vector              = RealVector<T>;
        const size_t vector_count = tile.elements / Vector::elements;
        switch (vector_count) {
            case 4:
                reduceRealVectors<T, op, 4>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 3:
                reduceRealVectors<T, op, 3>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 2:
                reduceRealVectors<T, op, 2>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 1:
                reduceRealVectors<T, op, 1>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            default: break;
        }
        vector_elements = vector_count * Vector::elements;
    }

    for (size_t element = vector_elements; element < tile.elements; ++element) {
        output[element * metadata.output_strides[0]] = reduceRealScalar<T, op>(
            input + element * metadata.input_strides[0],
            metadata.input_strides[dim], metadata.reduction_elements,
            change_nan, replacement);
    }
}

template<typename T>
AF_CPU_AVX2_TARGET inline void reduceComplexTile(
    T *const output_base, const T *const input_base,
    const ReductionMetadata<T> &metadata, const int dim,
    const TileDescriptor &tile, const bool change_nan,
    const double nanval) noexcept {
    static_assert(sizeof(T) == 2 * sizeof(typename ComplexVector<T>::Real),
                  "AVX2 complex reductions require interleaved components");
    T *const output        = output_base + tile.output_offset;
    const T *const input   = input_base + tile.input_offset;
    using Vector           = ComplexVector<T>;
    using Real             = typename Vector::Real;
    const Real replacement = static_cast<Real>(nanval);
    size_t vector_elements = 0;

    if (metadata.output_strides[0] == 1 && metadata.input_strides[0] == 1) {
        const size_t vector_count = tile.elements / Vector::elements;
        switch (vector_count) {
            case 4:
                reduceComplexVectors<T, 4>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 3:
                reduceComplexVectors<T, 3>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 2:
                reduceComplexVectors<T, 2>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 1:
                reduceComplexVectors<T, 1>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            default: break;
        }
        vector_elements = vector_count * Vector::elements;
    }

    for (size_t element = vector_elements; element < tile.elements; ++element) {
        output[element * metadata.output_strides[0]] = reduceComplexScalar(
            input + element * metadata.input_strides[0],
            metadata.input_strides[dim], metadata.reduction_elements,
            change_nan, replacement);
    }
}

template<typename T, ReductionOperation op>
AF_CPU_AVX2_TARGET void reduceRealRange(Param<T> out, CParam<T> in,
                                        const int dim, const bool change_nan,
                                        const double nanval,
                                        const size_t tile_begin,
                                        const size_t tile_end) noexcept {
    ReductionMetadata<T> metadata;
    const bool valid_metadata = makeReductionMetadata(metadata, out, in, dim);
    assert(valid_metadata);
    if (!valid_metadata) { return; }
    T *const output      = out.get();
    const T *const input = in.get();
    for (size_t tile_index = tile_begin; tile_index < tile_end; ++tile_index) {
        TileDescriptor tile;
        if (!describeTile(tile, metadata, tile_index)) { break; }
        reduceRealTile<T, op>(output, input, metadata, dim, tile, change_nan,
                              nanval);
    }
}

template<typename T>
AF_CPU_AVX2_TARGET void reduceComplexRange(Param<T> out, CParam<T> in,
                                           const int dim, const bool change_nan,
                                           const double nanval,
                                           const size_t tile_begin,
                                           const size_t tile_end) noexcept {
    ReductionMetadata<T> metadata;
    const bool valid_metadata = makeReductionMetadata(metadata, out, in, dim);
    assert(valid_metadata);
    if (!valid_metadata) { return; }
    T *const output      = out.get();
    const T *const input = in.get();
    for (size_t tile_index = tile_begin; tile_index < tile_end; ++tile_index) {
        TileDescriptor tile;
        if (!describeTile(tile, metadata, tile_index)) { break; }
        reduceComplexTile(output, input, metadata, dim, tile, change_nan,
                          nanval);
    }
}

}  // namespace

bool isReduceDimAVX2Compiled() noexcept { return true; }

AF_CPU_AVX2_TARGET void reduceDimSumRangeAVX2(Param<float> out,
                                              CParam<float> in, const int dim,
                                              const bool change_nan,
                                              const double nanval,
                                              const size_t tile_begin,
                                              const size_t tile_end) noexcept {
    reduceRealRange<float, ReductionOperation::Sum>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_TARGET void reduceDimSumRangeAVX2(Param<double> out,
                                              CParam<double> in, const int dim,
                                              const bool change_nan,
                                              const double nanval,
                                              const size_t tile_begin,
                                              const size_t tile_end) noexcept {
    reduceRealRange<double, ReductionOperation::Sum>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_TARGET void reduceDimSumRangeAVX2(Param<cfloat> out,
                                              CParam<cfloat> in, const int dim,
                                              const bool change_nan,
                                              const double nanval,
                                              const size_t tile_begin,
                                              const size_t tile_end) noexcept {
    reduceComplexRange(out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_TARGET void reduceDimSumRangeAVX2(Param<cdouble> out,
                                              CParam<cdouble> in, const int dim,
                                              const bool change_nan,
                                              const double nanval,
                                              const size_t tile_begin,
                                              const size_t tile_end) noexcept {
    reduceComplexRange(out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_TARGET void reduceDimProductRangeAVX2(
    Param<float> out, CParam<float> in, const int dim, const bool change_nan,
    const double nanval, const size_t tile_begin,
    const size_t tile_end) noexcept {
    reduceRealRange<float, ReductionOperation::Product>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_TARGET void reduceDimProductRangeAVX2(
    Param<double> out, CParam<double> in, const int dim, const bool change_nan,
    const double nanval, const size_t tile_begin,
    const size_t tile_end) noexcept {
    reduceRealRange<double, ReductionOperation::Product>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

#else

bool isReduceDimAVX2Compiled() noexcept { return false; }

void reduceDimSumRangeAVX2(Param<float>, CParam<float>, int, bool, double,
                           size_t, size_t) noexcept {}

void reduceDimSumRangeAVX2(Param<double>, CParam<double>, int, bool, double,
                           size_t, size_t) noexcept {}

void reduceDimSumRangeAVX2(Param<cfloat>, CParam<cfloat>, int, bool, double,
                           size_t, size_t) noexcept {}

void reduceDimSumRangeAVX2(Param<cdouble>, CParam<cdouble>, int, bool, double,
                           size_t, size_t) noexcept {}

void reduceDimProductRangeAVX2(Param<float>, CParam<float>, int, bool, double,
                               size_t, size_t) noexcept {}

void reduceDimProductRangeAVX2(Param<double>, CParam<double>, int, bool, double,
                               size_t, size_t) noexcept {}

#endif

}  // namespace detail
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire

#undef AF_CPU_AVX2_TARGET
#undef AF_CPU_HAS_AVX2_INTRINSICS
