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

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace detail {

/// Returns true when the AVX2 row implementations were compiled into afcpu.
bool isGradientAVX2Compiled() noexcept;

// These functions may only be called after isAVX2Supported() returns true.
// Their pointers may be unaligned; each row must have unit stride.
void gradientRowAVX2(float *grad0, float *grad1, const float *current,
                     const float *previous, const float *next, dim_t width,
                     float vertical_scale) noexcept;

void gradientRowAVX2(double *grad0, double *grad1, const double *current,
                     const double *previous, const double *next, dim_t width,
                     double vertical_scale) noexcept;

}  // namespace detail
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
