/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

namespace arrayfire {
namespace cpu {
namespace detail {

/// Returns true when both the processor and operating system support AVX2.
///
/// The result is detected once and cached for the lifetime of the process.
/// Builds without CPUID integration, non-x86 builds, and unsupported compilers
/// conservatively return false.
bool isAVX2Supported() noexcept;

}  // namespace detail
}  // namespace cpu
}  // namespace arrayfire
