/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>

#include <cstddef>

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace detail {

/// Returns true when the AVX2 transpose implementation was compiled into
/// afcpu.
bool isTransposeAVX2Compiled() noexcept;

// This function may only be called after isAVX2Supported() returns true. The
// input and output pointers may be unaligned. Strides are measured in elements
// and x strides must be one. Each tile is 8x8; consecutive tiles advance along
// the input tile-x direction. Conjugation is supported for 8- and 16-byte
// complex elements; callers must pass false for 4-byte elements.
void transposeTileRunAVX2(void *output, const void *input,
                          dim_t output_y_stride, dim_t input_y_stride,
                          size_t tile_count, size_t element_size,
                          bool conjugate) noexcept;

}  // namespace detail
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
