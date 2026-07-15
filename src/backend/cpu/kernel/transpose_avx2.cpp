/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/transpose_avx2.hpp>

#if (defined(__x86_64__) || defined(__amd64__) || defined(_M_X64) || \
     defined(__i386__) || defined(_M_IX86)) &&                       \
    (defined(__GNUC__) || defined(__clang__) ||                      \
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

#include <limits>

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace detail {

#if defined(AF_CPU_HAS_AVX2_INTRINSICS)

namespace {

using Byte = unsigned char;

AF_CPU_AVX2_TARGET inline __m256i load256(const Byte *pointer) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pointer));
}

AF_CPU_AVX2_TARGET inline void store256(Byte *pointer, __m256i value) noexcept {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(pointer), value);
}

AF_CPU_AVX2_TARGET void transpose8x8x32(Byte *output, const Byte *input,
                                        dim_t output_y_bytes,
                                        dim_t input_y_bytes) noexcept {
    const __m256i row0 = load256(input);
    const __m256i row1 = load256(input + input_y_bytes);
    const __m256i row2 = load256(input + 2 * input_y_bytes);
    const __m256i row3 = load256(input + 3 * input_y_bytes);
    const __m256i row4 = load256(input + 4 * input_y_bytes);
    const __m256i row5 = load256(input + 5 * input_y_bytes);
    const __m256i row6 = load256(input + 6 * input_y_bytes);
    const __m256i row7 = load256(input + 7 * input_y_bytes);

    const __m256i pair01_lo = _mm256_unpacklo_epi32(row0, row1);
    const __m256i pair01_hi = _mm256_unpackhi_epi32(row0, row1);
    const __m256i pair23_lo = _mm256_unpacklo_epi32(row2, row3);
    const __m256i pair23_hi = _mm256_unpackhi_epi32(row2, row3);
    const __m256i pair45_lo = _mm256_unpacklo_epi32(row4, row5);
    const __m256i pair45_hi = _mm256_unpackhi_epi32(row4, row5);
    const __m256i pair67_lo = _mm256_unpacklo_epi32(row6, row7);
    const __m256i pair67_hi = _mm256_unpackhi_epi32(row6, row7);

    const __m256i quad0123_0 = _mm256_unpacklo_epi64(pair01_lo, pair23_lo);
    const __m256i quad0123_1 = _mm256_unpackhi_epi64(pair01_lo, pair23_lo);
    const __m256i quad0123_2 = _mm256_unpacklo_epi64(pair01_hi, pair23_hi);
    const __m256i quad0123_3 = _mm256_unpackhi_epi64(pair01_hi, pair23_hi);
    const __m256i quad4567_0 = _mm256_unpacklo_epi64(pair45_lo, pair67_lo);
    const __m256i quad4567_1 = _mm256_unpackhi_epi64(pair45_lo, pair67_lo);
    const __m256i quad4567_2 = _mm256_unpacklo_epi64(pair45_hi, pair67_hi);
    const __m256i quad4567_3 = _mm256_unpackhi_epi64(pair45_hi, pair67_hi);

    store256(output, _mm256_permute2x128_si256(quad0123_0, quad4567_0, 0x20));
    store256(output + output_y_bytes,
             _mm256_permute2x128_si256(quad0123_1, quad4567_1, 0x20));
    store256(output + 2 * output_y_bytes,
             _mm256_permute2x128_si256(quad0123_2, quad4567_2, 0x20));
    store256(output + 3 * output_y_bytes,
             _mm256_permute2x128_si256(quad0123_3, quad4567_3, 0x20));
    store256(output + 4 * output_y_bytes,
             _mm256_permute2x128_si256(quad0123_0, quad4567_0, 0x31));
    store256(output + 5 * output_y_bytes,
             _mm256_permute2x128_si256(quad0123_1, quad4567_1, 0x31));
    store256(output + 6 * output_y_bytes,
             _mm256_permute2x128_si256(quad0123_2, quad4567_2, 0x31));
    store256(output + 7 * output_y_bytes,
             _mm256_permute2x128_si256(quad0123_3, quad4567_3, 0x31));
}

template<bool Conjugate>
AF_CPU_AVX2_TARGET inline __m256i conjugate64(__m256i value) noexcept {
    if constexpr (Conjugate) {
        const __m256i sign =
            _mm256_set1_epi64x(std::numeric_limits<long long>::min());
        return _mm256_xor_si256(value, sign);
    }
    return value;
}

template<bool Conjugate>
AF_CPU_AVX2_TARGET void transpose4x4x64(Byte *output, const Byte *input,
                                        dim_t output_y_bytes,
                                        dim_t input_y_bytes) noexcept {
    const __m256i row0 = load256(input);
    const __m256i row1 = load256(input + input_y_bytes);
    const __m256i row2 = load256(input + 2 * input_y_bytes);
    const __m256i row3 = load256(input + 3 * input_y_bytes);

    const __m256i pair01_lo = _mm256_unpacklo_epi64(row0, row1);
    const __m256i pair01_hi = _mm256_unpackhi_epi64(row0, row1);
    const __m256i pair23_lo = _mm256_unpacklo_epi64(row2, row3);
    const __m256i pair23_hi = _mm256_unpackhi_epi64(row2, row3);

    store256(output, conjugate64<Conjugate>(_mm256_permute2x128_si256(
                         pair01_lo, pair23_lo, 0x20)));
    store256(output + output_y_bytes,
             conjugate64<Conjugate>(
                 _mm256_permute2x128_si256(pair01_hi, pair23_hi, 0x20)));
    store256(output + 2 * output_y_bytes,
             conjugate64<Conjugate>(
                 _mm256_permute2x128_si256(pair01_lo, pair23_lo, 0x31)));
    store256(output + 3 * output_y_bytes,
             conjugate64<Conjugate>(
                 _mm256_permute2x128_si256(pair01_hi, pair23_hi, 0x31)));
}

template<bool Conjugate>
AF_CPU_AVX2_TARGET inline __m256i conjugate128(__m256i value) noexcept {
    if constexpr (Conjugate) {
        constexpr long long sign_bit = std::numeric_limits<long long>::min();
        const __m256i sign = _mm256_set_epi64x(sign_bit, 0, sign_bit, 0);
        return _mm256_xor_si256(value, sign);
    }
    return value;
}

template<bool Conjugate>
AF_CPU_AVX2_TARGET void transpose2x2x128(Byte *output, const Byte *input,
                                         dim_t output_y_bytes,
                                         dim_t input_y_bytes) noexcept {
    const __m256i row0 = load256(input);
    const __m256i row1 = load256(input + input_y_bytes);
    store256(output, conjugate128<Conjugate>(
                         _mm256_permute2x128_si256(row0, row1, 0x20)));
    store256(
        output + output_y_bytes,
        conjugate128<Conjugate>(_mm256_permute2x128_si256(row0, row1, 0x31)));
}

AF_CPU_AVX2_TARGET void transposeTiles32(Byte *output, const Byte *input,
                                         dim_t output_y_stride,
                                         dim_t input_y_stride,
                                         size_t tile_count) noexcept {
    constexpr dim_t element_size = 4;
    const dim_t output_y_bytes   = output_y_stride * element_size;
    const dim_t input_y_bytes    = input_y_stride * element_size;
    for (size_t tile = 0; tile < tile_count; ++tile) {
        transpose8x8x32(output, input, output_y_bytes, input_y_bytes);
        if (tile + 1 < tile_count) {
            output += 8 * element_size;
            input += 8 * input_y_bytes;
        }
    }
}

template<bool Conjugate>
AF_CPU_AVX2_TARGET void transposeTiles64(Byte *output, const Byte *input,
                                         dim_t output_y_stride,
                                         dim_t input_y_stride,
                                         size_t tile_count) noexcept {
    constexpr dim_t element_size = 8;
    const dim_t output_y_bytes   = output_y_stride * element_size;
    const dim_t input_y_bytes    = input_y_stride * element_size;
    for (size_t tile = 0; tile < tile_count; ++tile) {
        transpose4x4x64<Conjugate>(output, input, output_y_bytes,
                                   input_y_bytes);
        transpose4x4x64<Conjugate>(output + 4 * element_size,
                                   input + 4 * input_y_bytes, output_y_bytes,
                                   input_y_bytes);
        transpose4x4x64<Conjugate>(output + 4 * output_y_bytes,
                                   input + 4 * element_size, output_y_bytes,
                                   input_y_bytes);
        transpose4x4x64<Conjugate>(
            output + 4 * output_y_bytes + 4 * element_size,
            input + 4 * input_y_bytes + 4 * element_size, output_y_bytes,
            input_y_bytes);
        if (tile + 1 < tile_count) {
            output += 8 * element_size;
            input += 8 * input_y_bytes;
        }
    }
}

template<bool Conjugate>
AF_CPU_AVX2_TARGET void transposeTiles128(Byte *output, const Byte *input,
                                          dim_t output_y_stride,
                                          dim_t input_y_stride,
                                          size_t tile_count) noexcept {
    constexpr dim_t element_size = 16;
    const dim_t output_y_bytes   = output_y_stride * element_size;
    const dim_t input_y_bytes    = input_y_stride * element_size;
    for (size_t tile = 0; tile < tile_count; ++tile) {
        for (dim_t row = 0; row < 8; row += 2) {
            for (dim_t column = 0; column < 8; column += 2) {
                transpose2x2x128<Conjugate>(
                    output + column * output_y_bytes + row * element_size,
                    input + row * input_y_bytes + column * element_size,
                    output_y_bytes, input_y_bytes);
            }
        }
        if (tile + 1 < tile_count) {
            output += 8 * element_size;
            input += 8 * input_y_bytes;
        }
    }
}

}  // namespace

bool isTransposeAVX2Compiled() noexcept { return true; }

AF_CPU_AVX2_TARGET void transposeTileRunAVX2(void *output, const void *input,
                                             dim_t output_y_stride,
                                             dim_t input_y_stride,
                                             size_t tile_count,
                                             size_t element_size,
                                             bool conjugate) noexcept {
    auto *out      = static_cast<Byte *>(output);
    const auto *in = static_cast<const Byte *>(input);
    switch (element_size) {
        case 4:
            transposeTiles32(out, in, output_y_stride, input_y_stride,
                             tile_count);
            break;
        case 8:
            if (conjugate) {
                transposeTiles64<true>(out, in, output_y_stride, input_y_stride,
                                       tile_count);
            } else {
                transposeTiles64<false>(out, in, output_y_stride,
                                        input_y_stride, tile_count);
            }
            break;
        case 16:
            if (conjugate) {
                transposeTiles128<true>(out, in, output_y_stride,
                                        input_y_stride, tile_count);
            } else {
                transposeTiles128<false>(out, in, output_y_stride,
                                         input_y_stride, tile_count);
            }
            break;
        default: break;
    }
}

#else

bool isTransposeAVX2Compiled() noexcept { return false; }

void transposeTileRunAVX2(void *, const void *, dim_t, dim_t, size_t, size_t,
                          bool) noexcept {}

#endif

}  // namespace detail
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire

#undef AF_CPU_AVX2_TARGET
#undef AF_CPU_HAS_AVX2_INTRINSICS
