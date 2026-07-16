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
#define AF_CPU_AVX2_F16C_TARGET __attribute__((target("avx2,f16c")))
#else
#define AF_CPU_AVX2_TARGET
#define AF_CPU_AVX2_F16C_TARGET
#endif

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace detail {

#if defined(AF_CPU_HAS_AVX2_INTRINSICS)

namespace {

constexpr size_t tile_bytes = 128;

enum class ReductionOperation { Sum, Product, Minimum, Maximum };

struct TileDescriptor {
    dim_t input_offset;
    dim_t output_offset;
    size_t elements;
};

template<typename Ti, typename To>
struct ReductionMetadata {
    dim_t output_dims[4];
    dim_t output_strides[4];
    dim_t input_strides[4];
    dim_t reduction_elements;
    size_t x_tiles;
};

template<typename Ti, typename To>
bool makeReductionMetadata(ReductionMetadata<Ti, To> &metadata, Param<To> out,
                           CParam<Ti> in, const int dim) noexcept {
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

    constexpr size_t tile_elements = tile_bytes / sizeof(To);
    const size_t output_x          = static_cast<size_t>(odims[0]);
    metadata.x_tiles = (output_x + tile_elements - 1) / tile_elements;
    return metadata.reduction_elements >= 0;
}

template<typename Ti, typename To>
inline bool describeTile(TileDescriptor &descriptor,
                         const ReductionMetadata<Ti, To> &metadata,
                         const size_t tile) noexcept {
    constexpr size_t tile_elements = tile_bytes / sizeof(To);
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
        } else if constexpr (op == ReductionOperation::Product) {
            return _mm256_set1_ps(1.0F);
        } else if constexpr (op == ReductionOperation::Minimum) {
            return _mm256_set1_ps(std::numeric_limits<float>::infinity());
        } else {
            return _mm256_set1_ps(-std::numeric_limits<float>::infinity());
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
        } else if constexpr (op == ReductionOperation::Product) {
            return _mm256_mul_ps(input, accumulator);
        } else if constexpr (op == ReductionOperation::Minimum) {
            const Type keep_accumulator =
                _mm256_cmp_ps(accumulator, input, _CMP_LT_OQ);
            return _mm256_blendv_ps(input, accumulator, keep_accumulator);
        } else {
            const Type keep_accumulator =
                _mm256_cmp_ps(input, accumulator, _CMP_LT_OQ);
            return _mm256_blendv_ps(input, accumulator, keep_accumulator);
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
        } else if constexpr (op == ReductionOperation::Product) {
            return _mm256_set1_pd(1.0);
        } else if constexpr (op == ReductionOperation::Minimum) {
            return _mm256_set1_pd(std::numeric_limits<double>::infinity());
        } else {
            return _mm256_set1_pd(-std::numeric_limits<double>::infinity());
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
        } else if constexpr (op == ReductionOperation::Product) {
            return _mm256_mul_pd(input, accumulator);
        } else if constexpr (op == ReductionOperation::Minimum) {
            const Type keep_accumulator =
                _mm256_cmp_pd(accumulator, input, _CMP_LT_OQ);
            return _mm256_blendv_pd(input, accumulator, keep_accumulator);
        } else {
            const Type keep_accumulator =
                _mm256_cmp_pd(input, accumulator, _CMP_LT_OQ);
            return _mm256_blendv_pd(input, accumulator, keep_accumulator);
        }
    }
};

struct HalfVector {
    using Half                       = common::half;
    using Type                       = __m256;
    static constexpr size_t elements = 8;

    static_assert(sizeof(Half) == sizeof(std::uint16_t),
                  "F16C reductions require 16-bit half storage");
    static_assert(std::is_trivially_copyable<Half>::value,
                  "F16C reductions require trivially copyable half storage");

    AF_CPU_AVX2_F16C_TARGET static Type load(
        const Half *pointer) noexcept {
        __m128i packed;
        std::memcpy(&packed, pointer, sizeof(packed));
        return _mm256_cvtph_ps(packed);
    }

    AF_CPU_AVX2_F16C_TARGET static void store(Half *pointer,
                                               Type value) noexcept {
        const __m128i packed = _mm256_cvtps_ph(
            value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        std::memcpy(reinterpret_cast<unsigned char *>(pointer), &packed,
                    sizeof(packed));
    }
};

template<typename Ti, typename To>
struct IntegerVector {
    static_assert(std::is_integral<Ti>::value && std::is_integral<To>::value,
                  "integer reduction vectors require integral types");
    static_assert((std::is_same<Ti, To>::value ||
                   (sizeof(To) == 4 && sizeof(Ti) < sizeof(To) &&
                    std::is_signed<Ti>::value == std::is_signed<To>::value)),
                  "unsupported integer reduction conversion");

    using Type                       = __m256i;
    static constexpr size_t elements = 32 / sizeof(To);

    AF_CPU_AVX2_TARGET static Type broadcast(const To value) noexcept {
        if constexpr (sizeof(To) == 1) {
            char bits;
            std::memcpy(&bits, &value, sizeof(bits));
            return _mm256_set1_epi8(bits);
        } else if constexpr (sizeof(To) == 2) {
            short bits;
            std::memcpy(&bits, &value, sizeof(bits));
            return _mm256_set1_epi16(bits);
        } else if constexpr (sizeof(To) == 4) {
            int bits;
            std::memcpy(&bits, &value, sizeof(bits));
            return _mm256_set1_epi32(bits);
        } else {
            long long bits;
            std::memcpy(&bits, &value, sizeof(bits));
            return _mm256_set1_epi64x(bits);
        }
    }

    template<ReductionOperation op>
    AF_CPU_AVX2_TARGET static Type initial() noexcept {
        if constexpr (op == ReductionOperation::Sum) {
            return _mm256_setzero_si256();
        } else if constexpr (op == ReductionOperation::Product) {
            return broadcast(To(1));
        } else if constexpr (op == ReductionOperation::Minimum) {
            return broadcast(std::numeric_limits<To>::max());
        } else {
            return broadcast(std::numeric_limits<To>::lowest());
        }
    }

    AF_CPU_AVX2_TARGET static Type load(const Ti *pointer) noexcept {
        if constexpr (std::is_same<Ti, To>::value) {
            return _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(pointer));
        } else if constexpr (sizeof(Ti) == 1 && std::is_signed<Ti>::value) {
            const __m128i input =
                _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pointer));
            return _mm256_cvtepi8_epi32(input);
        } else if constexpr (sizeof(Ti) == 1) {
            const __m128i input =
                _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pointer));
            return _mm256_cvtepu8_epi32(input);
        } else if constexpr (std::is_signed<Ti>::value) {
            const __m128i input =
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(pointer));
            return _mm256_cvtepi16_epi32(input);
        } else {
            const __m128i input =
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(pointer));
            return _mm256_cvtepu16_epi32(input);
        }
    }

    AF_CPU_AVX2_TARGET static void store(To *pointer, Type value) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(pointer), value);
    }

    AF_CPU_AVX2_TARGET static Type multiply64(Type lhs, Type rhs) noexcept {
        const Type low_product = _mm256_mul_epu32(lhs, rhs);
        const Type lhs_high    = _mm256_srli_epi64(lhs, 32);
        const Type rhs_high    = _mm256_srli_epi64(rhs, 32);
        const Type cross = _mm256_add_epi64(_mm256_mul_epu32(lhs_high, rhs),
                                            _mm256_mul_epu32(lhs, rhs_high));
        return _mm256_add_epi64(low_product, _mm256_slli_epi64(cross, 32));
    }

    AF_CPU_AVX2_TARGET static Type add(Type lhs, Type rhs) noexcept {
        if constexpr (sizeof(To) == 1) {
            return _mm256_add_epi8(lhs, rhs);
        } else if constexpr (sizeof(To) == 2) {
            return _mm256_add_epi16(lhs, rhs);
        } else if constexpr (sizeof(To) == 4) {
            return _mm256_add_epi32(lhs, rhs);
        } else {
            return _mm256_add_epi64(lhs, rhs);
        }
    }

    AF_CPU_AVX2_TARGET static Type multiply(Type lhs, Type rhs) noexcept {
        if constexpr (sizeof(To) == 2) {
            return _mm256_mullo_epi16(lhs, rhs);
        } else if constexpr (sizeof(To) == 4) {
            return _mm256_mullo_epi32(lhs, rhs);
        } else {
            return multiply64(lhs, rhs);
        }
    }

    template<ReductionOperation op>
    AF_CPU_AVX2_TARGET static Type combine(Type input,
                                           Type accumulator) noexcept {
        if constexpr (op == ReductionOperation::Sum) {
            return add(input, accumulator);
        } else if constexpr (op == ReductionOperation::Product) {
            return multiply(input, accumulator);
        } else if constexpr (sizeof(To) == 1) {
            if constexpr (op == ReductionOperation::Minimum) {
                if constexpr (std::is_signed<To>::value) {
                    return _mm256_min_epi8(input, accumulator);
                } else {
                    return _mm256_min_epu8(input, accumulator);
                }
            } else if constexpr (std::is_signed<To>::value) {
                return _mm256_max_epi8(input, accumulator);
            } else {
                return _mm256_max_epu8(input, accumulator);
            }
        } else if constexpr (sizeof(To) == 2) {
            if constexpr (op == ReductionOperation::Minimum) {
                if constexpr (std::is_signed<To>::value) {
                    return _mm256_min_epi16(input, accumulator);
                } else {
                    return _mm256_min_epu16(input, accumulator);
                }
            } else if constexpr (std::is_signed<To>::value) {
                return _mm256_max_epi16(input, accumulator);
            } else {
                return _mm256_max_epu16(input, accumulator);
            }
        } else if constexpr (sizeof(To) == 4) {
            if constexpr (op == ReductionOperation::Minimum) {
                if constexpr (std::is_signed<To>::value) {
                    return _mm256_min_epi32(input, accumulator);
                } else {
                    return _mm256_min_epu32(input, accumulator);
                }
            } else if constexpr (std::is_signed<To>::value) {
                return _mm256_max_epi32(input, accumulator);
            } else {
                return _mm256_max_epu32(input, accumulator);
            }
        } else {
            Type ordered_input       = input;
            Type ordered_accumulator = accumulator;
            if constexpr (!std::is_signed<To>::value) {
                const Type sign_bit = _mm256_set1_epi64x(
                    std::numeric_limits<long long>::lowest());
                ordered_input       = _mm256_xor_si256(input, sign_bit);
                ordered_accumulator = _mm256_xor_si256(accumulator, sign_bit);
            }
            if constexpr (op == ReductionOperation::Minimum) {
                const Type take_input =
                    _mm256_cmpgt_epi64(ordered_accumulator, ordered_input);
                return _mm256_blendv_epi8(accumulator, input, take_input);
            } else {
                const Type take_input =
                    _mm256_cmpgt_epi64(ordered_input, ordered_accumulator);
                return _mm256_blendv_epi8(accumulator, input, take_input);
            }
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

    template<ReductionOperation op>
    AF_CPU_AVX2_TARGET static Type initial() noexcept {
        static_assert(op == ReductionOperation::Sum ||
                          op == ReductionOperation::Product,
                      "complex vectors only support arithmetic reductions");
        if constexpr (op == ReductionOperation::Sum) {
            return _mm256_setzero_ps();
        } else {
            return _mm256_setr_ps(1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F,
                                  1.0F, 0.0F);
        }
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

    template<ReductionOperation op>
    AF_CPU_AVX2_TARGET static Type combine(Type input,
                                           Type accumulator) noexcept {
        static_assert(op == ReductionOperation::Sum ||
                          op == ReductionOperation::Product,
                      "complex vectors only support arithmetic reductions");
        if constexpr (op == ReductionOperation::Sum) {
            return _mm256_add_ps(input, accumulator);
        } else {
            // Keep input on the left and avoid FMA so finite lanes preserve
            // the scalar fold's operation order exactly.
            const Type input_real = _mm256_moveldup_ps(input);
            const Type input_imag = _mm256_movehdup_ps(input);
            const Type accumulator_swapped =
                _mm256_permute_ps(accumulator, 0xB1);
            return _mm256_addsub_ps(
                _mm256_mul_ps(input_real, accumulator),
                _mm256_mul_ps(input_imag, accumulator_swapped));
        }
    }
};

template<>
struct ComplexVector<cdouble> {
    using Type                       = __m256d;
    using Real                       = double;
    static constexpr size_t elements = 2;

    template<ReductionOperation op>
    AF_CPU_AVX2_TARGET static Type initial() noexcept {
        static_assert(op == ReductionOperation::Sum ||
                          op == ReductionOperation::Product,
                      "complex vectors only support arithmetic reductions");
        if constexpr (op == ReductionOperation::Sum) {
            return _mm256_setzero_pd();
        } else {
            return _mm256_setr_pd(1.0, 0.0, 1.0, 0.0);
        }
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

    template<ReductionOperation op>
    AF_CPU_AVX2_TARGET static Type combine(Type input,
                                           Type accumulator) noexcept {
        static_assert(op == ReductionOperation::Sum ||
                          op == ReductionOperation::Product,
                      "complex vectors only support arithmetic reductions");
        if constexpr (op == ReductionOperation::Sum) {
            return _mm256_add_pd(input, accumulator);
        } else {
            // Keep input on the left and avoid FMA so finite lanes preserve
            // the scalar fold's operation order exactly.
            const Type input_real = _mm256_permute_pd(input, 0x0);
            const Type input_imag = _mm256_permute_pd(input, 0xF);
            const Type accumulator_swapped =
                _mm256_permute_pd(accumulator, 0x5);
            return _mm256_addsub_pd(
                _mm256_mul_pd(input_real, accumulator),
                _mm256_mul_pd(input_imag, accumulator_swapped));
        }
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
            if constexpr (op == ReductionOperation::Minimum) {
                value = Vector::replaceNaN(
                    value, std::numeric_limits<T>::infinity());
            } else if constexpr (op == ReductionOperation::Maximum) {
                value = Vector::replaceNaN(
                    value, -std::numeric_limits<T>::infinity());
            } else if (change_nan) {
                value = Vector::replaceNaN(value, nanval);
            }
            accumulators[vector] =
                Vector::template combine<op>(value, accumulators[vector]);
        }
    }

    for (size_t vector = 0; vector < vector_count; ++vector) {
        Vector::store(output + vector * Vector::elements, accumulators[vector]);
    }
}

template<ReductionOperation op, size_t vector_count>
AF_CPU_AVX2_F16C_TARGET void reduceHalfArithmeticVectors(
    float *output, const common::half *input, const dim_t reduction_stride,
    const dim_t reduction_elements, const bool change_nan,
    const float nanval) noexcept {
    static_assert(op == ReductionOperation::Sum ||
                      op == ReductionOperation::Product,
                  "half-to-float vectors only support arithmetic reductions");
    using Vector = RealVector<float>;
    typename Vector::Type accumulators[vector_count];
    for (size_t vector = 0; vector < vector_count; ++vector) {
        accumulators[vector] = Vector::template initial<op>();
    }

    for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
        const common::half *const row = input + reduced * reduction_stride;
        for (size_t vector = 0; vector < vector_count; ++vector) {
            typename Vector::Type value =
                HalfVector::load(row + vector * HalfVector::elements);
            if (change_nan) { value = Vector::replaceNaN(value, nanval); }
            accumulators[vector] =
                Vector::template combine<op>(value, accumulators[vector]);
        }
    }

    for (size_t vector = 0; vector < vector_count; ++vector) {
        Vector::store(output + vector * Vector::elements, accumulators[vector]);
    }
}

template<ReductionOperation op, size_t vector_count>
AF_CPU_AVX2_F16C_TARGET void reduceHalfExtremaVectors(
    common::half *output, const common::half *input,
    const dim_t reduction_stride, const dim_t reduction_elements) noexcept {
    static_assert(op == ReductionOperation::Minimum ||
                      op == ReductionOperation::Maximum,
                  "half vectors only support extrema reductions");
    using Vector = RealVector<float>;
    typename Vector::Type accumulators[vector_count];
    for (size_t vector = 0; vector < vector_count; ++vector) {
        accumulators[vector] = Vector::template initial<op>();
    }

    const float replacement =
        op == ReductionOperation::Minimum
            ? std::numeric_limits<float>::infinity()
            : -std::numeric_limits<float>::infinity();
    for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
        const common::half *const row = input + reduced * reduction_stride;
        for (size_t vector = 0; vector < vector_count; ++vector) {
            typename Vector::Type value =
                HalfVector::load(row + vector * HalfVector::elements);
            value = Vector::replaceNaN(value, replacement);
            accumulators[vector] =
                Vector::template combine<op>(value, accumulators[vector]);
        }
    }

    for (size_t vector = 0; vector < vector_count; ++vector) {
        HalfVector::store(output + vector * HalfVector::elements,
                          accumulators[vector]);
    }
}

template<typename Ti, typename To, ReductionOperation op, size_t vector_count>
AF_CPU_AVX2_TARGET void reduceIntegerVectors(
    To *output, const Ti *input, const dim_t reduction_stride,
    const dim_t reduction_elements) noexcept {
    using Vector = IntegerVector<Ti, To>;
    typename Vector::Type accumulators[vector_count];
    for (size_t vector = 0; vector < vector_count; ++vector) {
        accumulators[vector] = Vector::template initial<op>();
    }

    for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
        const Ti *const row = input + reduced * reduction_stride;
        for (size_t vector = 0; vector < vector_count; ++vector) {
            const typename Vector::Type value =
                Vector::load(row + vector * Vector::elements);
            accumulators[vector] =
                Vector::template combine<op>(value, accumulators[vector]);
        }
    }

    for (size_t vector = 0; vector < vector_count; ++vector) {
        Vector::store(output + vector * Vector::elements, accumulators[vector]);
    }
}

template<typename T, ReductionOperation op, size_t vector_count>
AF_CPU_AVX2_TARGET void reduceComplexVectors(
    T *output, const T *input, const dim_t reduction_stride,
    const dim_t reduction_elements, const bool change_nan,
    const typename ComplexVector<T>::Real nanval) noexcept {
    using Vector = ComplexVector<T>;
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

template<typename T, ReductionOperation op>
T reduceRealScalar(const T *input, const dim_t reduction_stride,
                   const dim_t reduction_elements, const bool change_nan,
                   const T nanval) noexcept {
    T accumulator;
    if constexpr (op == ReductionOperation::Sum) {
        accumulator = T(0);
    } else if constexpr (op == ReductionOperation::Product) {
        accumulator = T(1);
    } else if constexpr (op == ReductionOperation::Minimum) {
        accumulator = std::numeric_limits<T>::infinity();
    } else {
        accumulator = -std::numeric_limits<T>::infinity();
    }
    for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
        T value = input[reduced * reduction_stride];
        if constexpr (op == ReductionOperation::Minimum) {
            if (value != value) { value = std::numeric_limits<T>::infinity(); }
            accumulator = std::min(value, accumulator);
        } else if constexpr (op == ReductionOperation::Maximum) {
            if (value != value) {
                value = -std::numeric_limits<T>::infinity();
            }
            accumulator = std::max(value, accumulator);
        } else if constexpr (op == ReductionOperation::Sum) {
            if (change_nan && value != value) { value = nanval; }
            accumulator = value + accumulator;
        } else {
            if (change_nan && value != value) { value = nanval; }
            accumulator = value * accumulator;
        }
    }
    return accumulator;
}

template<ReductionOperation op>
float reduceHalfArithmeticScalar(const common::half *input,
                                 const dim_t reduction_stride,
                                 const dim_t reduction_elements,
                                 const bool change_nan,
                                 const float nanval) noexcept {
    static_assert(op == ReductionOperation::Sum ||
                      op == ReductionOperation::Product,
                  "half-to-float scalar only supports arithmetic reductions");
    float accumulator =
        op == ReductionOperation::Sum ? 0.0F : 1.0F;
    for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
        float value =
            static_cast<float>(input[reduced * reduction_stride]);
        if (change_nan && value != value) { value = nanval; }
        if constexpr (op == ReductionOperation::Sum) {
            accumulator = value + accumulator;
        } else {
            accumulator = value * accumulator;
        }
    }
    return accumulator;
}

template<ReductionOperation op>
common::half reduceHalfExtremaScalar(const common::half *input,
                                     const dim_t reduction_stride,
                                     const dim_t reduction_elements) noexcept {
    static_assert(op == ReductionOperation::Minimum ||
                      op == ReductionOperation::Maximum,
                  "half scalar only supports extrema reductions");
    float accumulator =
        op == ReductionOperation::Minimum
            ? std::numeric_limits<float>::infinity()
            : -std::numeric_limits<float>::infinity();
    for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
        float value =
            static_cast<float>(input[reduced * reduction_stride]);
        if (value != value) {
            value = op == ReductionOperation::Minimum
                        ? std::numeric_limits<float>::infinity()
                        : -std::numeric_limits<float>::infinity();
        }
        if constexpr (op == ReductionOperation::Minimum) {
            accumulator = std::min(value, accumulator);
        } else {
            accumulator = std::max(value, accumulator);
        }
    }
    return common::half(accumulator);
}

template<typename Ti, typename To, ReductionOperation op>
To reduceIntegerScalar(const Ti *input, const dim_t reduction_stride,
                       const dim_t reduction_elements) noexcept {
    if constexpr (op == ReductionOperation::Sum ||
                  op == ReductionOperation::Product) {
        using Unsigned = typename std::make_unsigned<To>::type;
        Unsigned accumulator =
            op == ReductionOperation::Sum ? Unsigned(0) : Unsigned(1);
        for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
            const To value =
                static_cast<To>(input[reduced * reduction_stride]);
            const Unsigned bits = static_cast<Unsigned>(value);
            if constexpr (op == ReductionOperation::Sum) {
                accumulator += bits;
            } else {
                accumulator *= bits;
            }
        }
        To result;
        std::memcpy(&result, &accumulator, sizeof(result));
        return result;
    } else {
        To accumulator = op == ReductionOperation::Minimum
                             ? std::numeric_limits<To>::max()
                             : std::numeric_limits<To>::lowest();
        for (dim_t reduced = 0; reduced < reduction_elements; ++reduced) {
            const To value =
                static_cast<To>(input[reduced * reduction_stride]);
            if constexpr (op == ReductionOperation::Minimum) {
                accumulator = std::min(value, accumulator);
            } else {
                accumulator = std::max(value, accumulator);
            }
        }
        return accumulator;
    }
}

template<typename T, ReductionOperation op>
T reduceComplexScalar(const T *input, const dim_t reduction_stride,
                      const dim_t reduction_elements, const bool change_nan,
                      const typename ComplexVector<T>::Real nanval) noexcept {
    using Real = typename ComplexVector<T>::Real;
    static_assert(op == ReductionOperation::Sum ||
                      op == ReductionOperation::Product,
                  "complex scalar only supports arithmetic reductions");
    if constexpr (op == ReductionOperation::Product) {
        return reduceDimComplexProductScalar(input, reduction_stride,
                                             reduction_elements, change_nan,
                                             nanval);
    }
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
    const ReductionMetadata<T, T> &metadata, const int dim,
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

template<ReductionOperation op>
AF_CPU_AVX2_F16C_TARGET inline void reduceHalfArithmeticTile(
    float *const output_base, const common::half *const input_base,
    const ReductionMetadata<common::half, float> &metadata, const int dim,
    const TileDescriptor &tile, const bool change_nan,
    const double nanval) noexcept {
    float *const output             = output_base + tile.output_offset;
    const common::half *const input = input_base + tile.input_offset;
    const float replacement         = static_cast<float>(nanval);
    size_t vector_elements          = 0;

    if (metadata.output_strides[0] == 1 && metadata.input_strides[0] == 1) {
        const size_t vector_count = tile.elements / HalfVector::elements;
        switch (vector_count) {
            case 4:
                reduceHalfArithmeticVectors<op, 4>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 3:
                reduceHalfArithmeticVectors<op, 3>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 2:
                reduceHalfArithmeticVectors<op, 2>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 1:
                reduceHalfArithmeticVectors<op, 1>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            default: break;
        }
        vector_elements = vector_count * HalfVector::elements;
    }

    for (size_t element = vector_elements; element < tile.elements; ++element) {
        output[element * metadata.output_strides[0]] =
            reduceHalfArithmeticScalar<op>(
                input + element * metadata.input_strides[0],
                metadata.input_strides[dim], metadata.reduction_elements,
                change_nan, replacement);
    }
}

template<ReductionOperation op>
AF_CPU_AVX2_F16C_TARGET inline void reduceHalfExtremaTile(
    common::half *const output_base, const common::half *const input_base,
    const ReductionMetadata<common::half, common::half> &metadata,
    const int dim, const TileDescriptor &tile) noexcept {
    common::half *const output       = output_base + tile.output_offset;
    const common::half *const input = input_base + tile.input_offset;
    size_t vector_elements          = 0;

    if (metadata.output_strides[0] == 1 && metadata.input_strides[0] == 1) {
        const size_t vector_count = tile.elements / HalfVector::elements;
        switch (vector_count) {
            case 8:
                reduceHalfExtremaVectors<op, 8>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            case 7:
                reduceHalfExtremaVectors<op, 7>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            case 6:
                reduceHalfExtremaVectors<op, 6>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            case 5:
                reduceHalfExtremaVectors<op, 5>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            case 4:
                reduceHalfExtremaVectors<op, 4>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            case 3:
                reduceHalfExtremaVectors<op, 3>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            case 2:
                reduceHalfExtremaVectors<op, 2>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            case 1:
                reduceHalfExtremaVectors<op, 1>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            default: break;
        }
        vector_elements = vector_count * HalfVector::elements;
    }

    for (size_t element = vector_elements; element < tile.elements; ++element) {
        output[element * metadata.output_strides[0]] =
            reduceHalfExtremaScalar<op>(
                input + element * metadata.input_strides[0],
                metadata.input_strides[dim], metadata.reduction_elements);
    }
}

template<typename Ti, typename To, ReductionOperation op>
AF_CPU_AVX2_TARGET inline void reduceIntegerTile(
    To *const output_base, const Ti *const input_base,
    const ReductionMetadata<Ti, To> &metadata, const int dim,
    const TileDescriptor &tile) noexcept {
    To *const output       = output_base + tile.output_offset;
    const Ti *const input  = input_base + tile.input_offset;
    size_t vector_elements = 0;

    if (metadata.output_strides[0] == 1 && metadata.input_strides[0] == 1) {
        using Vector              = IntegerVector<Ti, To>;
        const size_t vector_count = tile.elements / Vector::elements;
        switch (vector_count) {
            case 4:
                reduceIntegerVectors<Ti, To, op, 4>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            case 3:
                reduceIntegerVectors<Ti, To, op, 3>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            case 2:
                reduceIntegerVectors<Ti, To, op, 2>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            case 1:
                reduceIntegerVectors<Ti, To, op, 1>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements);
                break;
            default: break;
        }
        vector_elements = vector_count * Vector::elements;
    }

    for (size_t element = vector_elements; element < tile.elements; ++element) {
        output[element * metadata.output_strides[0]] =
            reduceIntegerScalar<Ti, To, op>(
                input + element * metadata.input_strides[0],
                metadata.input_strides[dim], metadata.reduction_elements);
    }
}

template<typename T, ReductionOperation op>
AF_CPU_AVX2_TARGET inline void reduceComplexTile(
    T *const output_base, const T *const input_base,
    const ReductionMetadata<T, T> &metadata, const int dim,
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
                reduceComplexVectors<T, op, 4>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 3:
                reduceComplexVectors<T, op, 3>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 2:
                reduceComplexVectors<T, op, 2>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            case 1:
                reduceComplexVectors<T, op, 1>(
                    output, input, metadata.input_strides[dim],
                    metadata.reduction_elements, change_nan, replacement);
                break;
            default: break;
        }
        vector_elements = vector_count * Vector::elements;
    }

    for (size_t element = vector_elements; element < tile.elements; ++element) {
        output[element * metadata.output_strides[0]] =
            reduceComplexScalar<T, op>(
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
    ReductionMetadata<T, T> metadata;
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

template<ReductionOperation op>
AF_CPU_AVX2_F16C_TARGET void reduceHalfArithmeticRange(
    Param<float> out, CParam<common::half> in, const int dim,
    const bool change_nan, const double nanval, const size_t tile_begin,
    const size_t tile_end) noexcept {
    ReductionMetadata<common::half, float> metadata;
    const bool valid_metadata = makeReductionMetadata(metadata, out, in, dim);
    assert(valid_metadata);
    if (!valid_metadata) { return; }
    float *const output             = out.get();
    const common::half *const input = in.get();
    for (size_t tile_index = tile_begin; tile_index < tile_end; ++tile_index) {
        TileDescriptor tile;
        if (!describeTile(tile, metadata, tile_index)) { break; }
        reduceHalfArithmeticTile<op>(output, input, metadata, dim, tile,
                                     change_nan, nanval);
    }
}

template<ReductionOperation op>
AF_CPU_AVX2_F16C_TARGET void reduceHalfExtremaRange(
    Param<common::half> out, CParam<common::half> in, const int dim,
    const size_t tile_begin, const size_t tile_end) noexcept {
    ReductionMetadata<common::half, common::half> metadata;
    const bool valid_metadata = makeReductionMetadata(metadata, out, in, dim);
    assert(valid_metadata);
    if (!valid_metadata) { return; }
    common::half *const output       = out.get();
    const common::half *const input = in.get();
    for (size_t tile_index = tile_begin; tile_index < tile_end; ++tile_index) {
        TileDescriptor tile;
        if (!describeTile(tile, metadata, tile_index)) { break; }
        reduceHalfExtremaTile<op>(output, input, metadata, dim, tile);
    }
}

template<typename Ti, typename To, ReductionOperation op>
AF_CPU_AVX2_TARGET void reduceIntegerRange(Param<To> out, CParam<Ti> in,
                                           const int dim,
                                           const size_t tile_begin,
                                           const size_t tile_end) noexcept {
    ReductionMetadata<Ti, To> metadata;
    const bool valid_metadata = makeReductionMetadata(metadata, out, in, dim);
    assert(valid_metadata);
    if (!valid_metadata) { return; }
    To *const output      = out.get();
    const Ti *const input = in.get();
    for (size_t tile_index = tile_begin; tile_index < tile_end; ++tile_index) {
        TileDescriptor tile;
        if (!describeTile(tile, metadata, tile_index)) { break; }
        reduceIntegerTile<Ti, To, op>(output, input, metadata, dim, tile);
    }
}

template<typename T, ReductionOperation op>
AF_CPU_AVX2_TARGET void reduceComplexRange(Param<T> out, CParam<T> in,
                                           const int dim, const bool change_nan,
                                           const double nanval,
                                           const size_t tile_begin,
                                           const size_t tile_end) noexcept {
    ReductionMetadata<T, T> metadata;
    const bool valid_metadata = makeReductionMetadata(metadata, out, in, dim);
    assert(valid_metadata);
    if (!valid_metadata) { return; }
    T *const output      = out.get();
    const T *const input = in.get();
    for (size_t tile_index = tile_begin; tile_index < tile_end; ++tile_index) {
        TileDescriptor tile;
        if (!describeTile(tile, metadata, tile_index)) { break; }
        reduceComplexTile<T, op>(output, input, metadata, dim, tile,
                                 change_nan, nanval);
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
    reduceComplexRange<cfloat, ReductionOperation::Sum>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_TARGET void reduceDimSumRangeAVX2(Param<cdouble> out,
                                              CParam<cdouble> in, const int dim,
                                              const bool change_nan,
                                              const double nanval,
                                              const size_t tile_begin,
                                              const size_t tile_end) noexcept {
    reduceComplexRange<cdouble, ReductionOperation::Sum>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_F16C_TARGET void reduceDimSumRangeAVX2(
    Param<float> out, CParam<common::half> in, const int dim,
    const bool change_nan, const double nanval, const size_t tile_begin,
    const size_t tile_end) noexcept {
    reduceHalfArithmeticRange<ReductionOperation::Sum>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
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

AF_CPU_AVX2_TARGET void reduceDimProductRangeAVX2(
    Param<cfloat> out, CParam<cfloat> in, const int dim, const bool change_nan,
    const double nanval, const size_t tile_begin,
    const size_t tile_end) noexcept {
    reduceComplexRange<cfloat, ReductionOperation::Product>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_TARGET void reduceDimProductRangeAVX2(
    Param<cdouble> out, CParam<cdouble> in, const int dim,
    const bool change_nan, const double nanval, const size_t tile_begin,
    const size_t tile_end) noexcept {
    reduceComplexRange<cdouble, ReductionOperation::Product>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_F16C_TARGET void reduceDimProductRangeAVX2(
    Param<float> out, CParam<common::half> in, const int dim,
    const bool change_nan, const double nanval, const size_t tile_begin,
    const size_t tile_end) noexcept {
    reduceHalfArithmeticRange<ReductionOperation::Product>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_TARGET void reduceDimMinRangeAVX2(
    Param<float> out, CParam<float> in, const int dim, const bool change_nan,
    const double nanval, const size_t tile_begin,
    const size_t tile_end) noexcept {
    reduceRealRange<float, ReductionOperation::Minimum>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_TARGET void reduceDimMinRangeAVX2(
    Param<double> out, CParam<double> in, const int dim, const bool change_nan,
    const double nanval, const size_t tile_begin,
    const size_t tile_end) noexcept {
    reduceRealRange<double, ReductionOperation::Minimum>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_F16C_TARGET void reduceDimMinRangeAVX2(
    Param<common::half> out, CParam<common::half> in, const int dim, const bool,
    const double, const size_t tile_begin, const size_t tile_end) noexcept {
    reduceHalfExtremaRange<ReductionOperation::Minimum>(out, in, dim,
                                                         tile_begin, tile_end);
}

AF_CPU_AVX2_TARGET void reduceDimMaxRangeAVX2(
    Param<float> out, CParam<float> in, const int dim, const bool change_nan,
    const double nanval, const size_t tile_begin,
    const size_t tile_end) noexcept {
    reduceRealRange<float, ReductionOperation::Maximum>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_TARGET void reduceDimMaxRangeAVX2(
    Param<double> out, CParam<double> in, const int dim, const bool change_nan,
    const double nanval, const size_t tile_begin,
    const size_t tile_end) noexcept {
    reduceRealRange<double, ReductionOperation::Maximum>(
        out, in, dim, change_nan, nanval, tile_begin, tile_end);
}

AF_CPU_AVX2_F16C_TARGET void reduceDimMaxRangeAVX2(
    Param<common::half> out, CParam<common::half> in, const int dim, const bool,
    const double, const size_t tile_begin, const size_t tile_end) noexcept {
    reduceHalfExtremaRange<ReductionOperation::Maximum>(out, in, dim,
                                                         tile_begin, tile_end);
}

template<typename Ti, typename To>
AF_CPU_AVX2_TARGET void reduceDimIntegerSumRangeAVX2(
    Param<To> out, CParam<Ti> in, const int dim, const bool, const double,
    const size_t tile_begin, const size_t tile_end) noexcept {
    reduceIntegerRange<Ti, To, ReductionOperation::Sum>(out, in, dim,
                                                        tile_begin, tile_end);
}

template<typename Ti, typename To>
AF_CPU_AVX2_TARGET void reduceDimIntegerProductRangeAVX2(
    Param<To> out, CParam<Ti> in, const int dim, const bool, const double,
    const size_t tile_begin, const size_t tile_end) noexcept {
    reduceIntegerRange<Ti, To, ReductionOperation::Product>(
        out, in, dim, tile_begin, tile_end);
}

template<typename T>
AF_CPU_AVX2_TARGET void reduceDimIntegerMinRangeAVX2(
    Param<T> out, CParam<T> in, const int dim, const bool, const double,
    const size_t tile_begin, const size_t tile_end) noexcept {
    reduceIntegerRange<T, T, ReductionOperation::Minimum>(out, in, dim,
                                                          tile_begin, tile_end);
}

template<typename T>
AF_CPU_AVX2_TARGET void reduceDimIntegerMaxRangeAVX2(
    Param<T> out, CParam<T> in, const int dim, const bool, const double,
    const size_t tile_begin, const size_t tile_end) noexcept {
    reduceIntegerRange<T, T, ReductionOperation::Maximum>(out, in, dim,
                                                          tile_begin, tile_end);
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

void reduceDimSumRangeAVX2(Param<float>, CParam<common::half>, int, bool,
                           double, size_t, size_t) noexcept {}

void reduceDimProductRangeAVX2(Param<float>, CParam<float>, int, bool, double,
                               size_t, size_t) noexcept {}

void reduceDimProductRangeAVX2(Param<double>, CParam<double>, int, bool, double,
                               size_t, size_t) noexcept {}

void reduceDimProductRangeAVX2(Param<cfloat>, CParam<cfloat>, int, bool, double,
                               size_t, size_t) noexcept {}

void reduceDimProductRangeAVX2(Param<cdouble>, CParam<cdouble>, int, bool,
                               double, size_t, size_t) noexcept {}

void reduceDimProductRangeAVX2(Param<float>, CParam<common::half>, int, bool,
                               double, size_t, size_t) noexcept {}

void reduceDimMinRangeAVX2(Param<float>, CParam<float>, int, bool, double,
                           size_t, size_t) noexcept {}

void reduceDimMinRangeAVX2(Param<double>, CParam<double>, int, bool, double,
                           size_t, size_t) noexcept {}

void reduceDimMinRangeAVX2(Param<common::half>, CParam<common::half>, int, bool,
                           double, size_t, size_t) noexcept {}

void reduceDimMaxRangeAVX2(Param<float>, CParam<float>, int, bool, double,
                           size_t, size_t) noexcept {}

void reduceDimMaxRangeAVX2(Param<double>, CParam<double>, int, bool, double,
                           size_t, size_t) noexcept {}

void reduceDimMaxRangeAVX2(Param<common::half>, CParam<common::half>, int, bool,
                           double, size_t, size_t) noexcept {}

template<typename Ti, typename To>
void reduceDimIntegerSumRangeAVX2(Param<To>, CParam<Ti>, int, bool, double,
                                  size_t, size_t) noexcept {}

template<typename Ti, typename To>
void reduceDimIntegerProductRangeAVX2(Param<To>, CParam<Ti>, int, bool, double,
                                      size_t, size_t) noexcept {}

template<typename T>
void reduceDimIntegerMinRangeAVX2(Param<T>, CParam<T>, int, bool, double,
                                  size_t, size_t) noexcept {}

template<typename T>
void reduceDimIntegerMaxRangeAVX2(Param<T>, CParam<T>, int, bool, double,
                                  size_t, size_t) noexcept {}

#endif

#define INSTANTIATE_INTEGER_ARITHMETIC(Ti, To)                              \
    template void reduceDimIntegerSumRangeAVX2<Ti, To>(                     \
        Param<To>, CParam<Ti>, int, bool, double, size_t, size_t) noexcept; \
    template void reduceDimIntegerProductRangeAVX2<Ti, To>(                 \
        Param<To>, CParam<Ti>, int, bool, double, size_t, size_t) noexcept;

INSTANTIATE_INTEGER_ARITHMETIC(signed char, int)
INSTANTIATE_INTEGER_ARITHMETIC(unsigned char, unsigned int)
INSTANTIATE_INTEGER_ARITHMETIC(short, int)
INSTANTIATE_INTEGER_ARITHMETIC(unsigned short, unsigned int)
INSTANTIATE_INTEGER_ARITHMETIC(int, int)
INSTANTIATE_INTEGER_ARITHMETIC(unsigned int, unsigned int)
INSTANTIATE_INTEGER_ARITHMETIC(long long, long long)
INSTANTIATE_INTEGER_ARITHMETIC(unsigned long long, unsigned long long)

#undef INSTANTIATE_INTEGER_ARITHMETIC

#define INSTANTIATE_INTEGER_EXTREMA(T)                                    \
    template void reduceDimIntegerMinRangeAVX2<T>(                        \
        Param<T>, CParam<T>, int, bool, double, size_t, size_t) noexcept; \
    template void reduceDimIntegerMaxRangeAVX2<T>(                        \
        Param<T>, CParam<T>, int, bool, double, size_t, size_t) noexcept;

INSTANTIATE_INTEGER_EXTREMA(signed char)
INSTANTIATE_INTEGER_EXTREMA(unsigned char)
INSTANTIATE_INTEGER_EXTREMA(short)
INSTANTIATE_INTEGER_EXTREMA(unsigned short)
INSTANTIATE_INTEGER_EXTREMA(int)
INSTANTIATE_INTEGER_EXTREMA(unsigned int)
INSTANTIATE_INTEGER_EXTREMA(long long)
INSTANTIATE_INTEGER_EXTREMA(unsigned long long)

#undef INSTANTIATE_INTEGER_EXTREMA

}  // namespace detail
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire

#undef AF_CPU_AVX2_TARGET
#undef AF_CPU_AVX2_F16C_TARGET
#undef AF_CPU_HAS_AVX2_INTRINSICS
