/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>

#include <cstddef>

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace detail {

/// Returns true when the AVX2 dimensional-reduction implementations were
/// compiled into afcpu.
bool isReduceDimAVX2Compiled() noexcept;

// These functions may only be called after isAVX2Supported() returns true.
// A tile contains 128 output bytes along dimension zero. Tile indices advance
// through x tiles first, followed by output dimensions one through three.
void reduceDimSumRangeAVX2(Param<float> out, CParam<float> in, int dim,
                           bool change_nan, double nanval, size_t tile_begin,
                           size_t tile_end) noexcept;
void reduceDimSumRangeAVX2(Param<double> out, CParam<double> in, int dim,
                           bool change_nan, double nanval, size_t tile_begin,
                           size_t tile_end) noexcept;
void reduceDimSumRangeAVX2(Param<cfloat> out, CParam<cfloat> in, int dim,
                           bool change_nan, double nanval, size_t tile_begin,
                           size_t tile_end) noexcept;
void reduceDimSumRangeAVX2(Param<cdouble> out, CParam<cdouble> in, int dim,
                           bool change_nan, double nanval, size_t tile_begin,
                           size_t tile_end) noexcept;

void reduceDimProductRangeAVX2(Param<float> out, CParam<float> in, int dim,
                               bool change_nan, double nanval,
                               size_t tile_begin, size_t tile_end) noexcept;
void reduceDimProductRangeAVX2(Param<double> out, CParam<double> in, int dim,
                               bool change_nan, double nanval,
                               size_t tile_begin, size_t tile_end) noexcept;

}  // namespace detail
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
