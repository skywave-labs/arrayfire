/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/transpose_neon.hpp>

// Advanced SIMD is part of the AArch64 baseline. Keep AArch32 on the scalar
// path until the generated code and thresholds can be validated on native
// hardware.
#if defined(AF_CPU_COMPILE_NEON_TRANSPOSE) && defined(__aarch64__) && \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
#define AF_CPU_HAS_NEON_INTRINSICS
#include <arm_neon.h>
#endif

#include <cstdint>

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace detail {

#if defined(AF_CPU_HAS_NEON_INTRINSICS)

namespace {

using Byte = std::uint8_t;

inline uint32x4_t load32x4(const Byte *pointer) noexcept {
    return vreinterpretq_u32_u8(vld1q_u8(pointer));
}

inline void store32x4(Byte *pointer, uint32x4_t value) noexcept {
    vst1q_u8(pointer, vreinterpretq_u8_u32(value));
}

inline uint64x2_t load64x2(const Byte *pointer) noexcept {
    return vreinterpretq_u64_u8(vld1q_u8(pointer));
}

inline void store64x2(Byte *pointer, uint64x2_t value) noexcept {
    vst1q_u8(pointer, vreinterpretq_u8_u64(value));
}

void transpose4x4x32(Byte *output, const Byte *input, dim_t output_y_bytes,
                     dim_t input_y_bytes) noexcept {
    const uint32x4_t row0 = load32x4(input);
    const uint32x4_t row1 = load32x4(input + input_y_bytes);
    const uint32x4_t row2 = load32x4(input + 2 * input_y_bytes);
    const uint32x4_t row3 = load32x4(input + 3 * input_y_bytes);

    const uint32x4_t pair01_even = vtrn1q_u32(row0, row1);
    const uint32x4_t pair01_odd  = vtrn2q_u32(row0, row1);
    const uint32x4_t pair23_even = vtrn1q_u32(row2, row3);
    const uint32x4_t pair23_odd  = vtrn2q_u32(row2, row3);

    const uint64x2_t even01 = vreinterpretq_u64_u32(pair01_even);
    const uint64x2_t even23 = vreinterpretq_u64_u32(pair23_even);
    const uint64x2_t odd01  = vreinterpretq_u64_u32(pair01_odd);
    const uint64x2_t odd23  = vreinterpretq_u64_u32(pair23_odd);

    store32x4(output, vreinterpretq_u32_u64(vtrn1q_u64(even01, even23)));
    store32x4(output + output_y_bytes,
              vreinterpretq_u32_u64(vtrn1q_u64(odd01, odd23)));
    store32x4(output + 2 * output_y_bytes,
              vreinterpretq_u32_u64(vtrn2q_u64(even01, even23)));
    store32x4(output + 3 * output_y_bytes,
              vreinterpretq_u32_u64(vtrn2q_u64(odd01, odd23)));
}

template<bool Conjugate>
inline uint64x2_t conjugate64(uint64x2_t value) noexcept {
    if constexpr (Conjugate) {
        uint32x4_t sign = vdupq_n_u32(0);
        sign            = vsetq_lane_u32(UINT32_C(1) << 31U, sign, 1);
        sign            = vsetq_lane_u32(UINT32_C(1) << 31U, sign, 3);
        return vreinterpretq_u64_u32(
            veorq_u32(vreinterpretq_u32_u64(value), sign));
    }
    return value;
}

template<bool Conjugate>
void transpose2x2x64(Byte *output, const Byte *input, dim_t output_y_bytes,
                     dim_t input_y_bytes) noexcept {
    const uint64x2_t row0 = load64x2(input);
    const uint64x2_t row1 = load64x2(input + input_y_bytes);

    store64x2(output, conjugate64<Conjugate>(vtrn1q_u64(row0, row1)));
    store64x2(output + output_y_bytes,
              conjugate64<Conjugate>(vtrn2q_u64(row0, row1)));
}

void transposeTiles32(Byte *output, const Byte *input, dim_t output_y_stride,
                      dim_t input_y_stride, size_t tile_count) noexcept {
    constexpr dim_t element_size = 4;
    const dim_t output_y_bytes   = output_y_stride * element_size;
    const dim_t input_y_bytes    = input_y_stride * element_size;
    for (size_t tile = 0; tile < tile_count; ++tile) {
        transpose4x4x32(output, input, output_y_bytes, input_y_bytes);
        transpose4x4x32(output + 4 * element_size, input + 4 * input_y_bytes,
                        output_y_bytes, input_y_bytes);
        transpose4x4x32(output + 4 * output_y_bytes, input + 4 * element_size,
                        output_y_bytes, input_y_bytes);
        transpose4x4x32(output + 4 * output_y_bytes + 4 * element_size,
                        input + 4 * input_y_bytes + 4 * element_size,
                        output_y_bytes, input_y_bytes);
        if (tile + 1 < tile_count) {
            output += 8 * element_size;
            input += 8 * input_y_bytes;
        }
    }
}

template<bool Conjugate>
void transposeTiles64(Byte *output, const Byte *input, dim_t output_y_stride,
                      dim_t input_y_stride, size_t tile_count) noexcept {
    constexpr dim_t element_size = 8;
    const dim_t output_y_bytes   = output_y_stride * element_size;
    const dim_t input_y_bytes    = input_y_stride * element_size;
    for (size_t tile = 0; tile < tile_count; ++tile) {
        for (dim_t row = 0; row < 8; row += 2) {
            for (dim_t column = 0; column < 8; column += 2) {
                transpose2x2x64<Conjugate>(
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

bool isTransposeNEONCompiled() noexcept { return true; }

void transposeTileRunNEON(void *output, const void *input,
                          dim_t output_y_stride, dim_t input_y_stride,
                          size_t tile_count, size_t element_size,
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
        default: break;
    }
}

#else

bool isTransposeNEONCompiled() noexcept { return false; }

void transposeTileRunNEON(void *, const void *, dim_t, dim_t, size_t, size_t,
                          bool) noexcept {}

#endif

}  // namespace detail
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire

#undef AF_CPU_HAS_NEON_INTRINSICS
