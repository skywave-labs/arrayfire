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
#include <common/half.hpp>

#include <cstddef>

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace detail {

/// Returns true when the AVX2 dimensional-reduction implementations were
/// compiled into afcpu.
bool isReduceDimAVX2Compiled() noexcept;

// These functions may only be called after isAVX2Supported() returns true.
// Overloads accepting half input additionally require isF16CSupported().
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
void reduceDimSumRangeAVX2(Param<float> out, CParam<common::half> in, int dim,
                           bool change_nan, double nanval, size_t tile_begin,
                           size_t tile_end) noexcept;

void reduceDimProductRangeAVX2(Param<float> out, CParam<float> in, int dim,
                               bool change_nan, double nanval,
                               size_t tile_begin, size_t tile_end) noexcept;
void reduceDimProductRangeAVX2(Param<double> out, CParam<double> in, int dim,
                               bool change_nan, double nanval,
                               size_t tile_begin, size_t tile_end) noexcept;
void reduceDimProductRangeAVX2(Param<float> out, CParam<common::half> in,
                               int dim, bool change_nan, double nanval,
                               size_t tile_begin, size_t tile_end) noexcept;

void reduceDimMinRangeAVX2(Param<float> out, CParam<float> in, int dim,
                           bool change_nan, double nanval, size_t tile_begin,
                           size_t tile_end) noexcept;
void reduceDimMinRangeAVX2(Param<double> out, CParam<double> in, int dim,
                           bool change_nan, double nanval, size_t tile_begin,
                           size_t tile_end) noexcept;
void reduceDimMinRangeAVX2(Param<common::half> out, CParam<common::half> in,
                           int dim, bool change_nan, double nanval,
                           size_t tile_begin, size_t tile_end) noexcept;

void reduceDimMaxRangeAVX2(Param<float> out, CParam<float> in, int dim,
                           bool change_nan, double nanval, size_t tile_begin,
                           size_t tile_end) noexcept;
void reduceDimMaxRangeAVX2(Param<double> out, CParam<double> in, int dim,
                           bool change_nan, double nanval, size_t tile_begin,
                           size_t tile_end) noexcept;
void reduceDimMaxRangeAVX2(Param<common::half> out, CParam<common::half> in,
                           int dim, bool change_nan, double nanval,
                           size_t tile_begin, size_t tile_end) noexcept;

// Integral reductions use separate entry points because 8- and 16-bit sum
// and product inputs are widened to 32-bit outputs by the public API.
template<typename Ti, typename To>
void reduceDimIntegerSumRangeAVX2(Param<To> out, CParam<Ti> in, int dim,
                                  bool change_nan, double nanval,
                                  size_t tile_begin, size_t tile_end) noexcept;

template<typename Ti, typename To>
void reduceDimIntegerProductRangeAVX2(Param<To> out, CParam<Ti> in, int dim,
                                      bool change_nan, double nanval,
                                      size_t tile_begin,
                                      size_t tile_end) noexcept;

template<typename T>
void reduceDimIntegerMinRangeAVX2(Param<T> out, CParam<T> in, int dim,
                                  bool change_nan, double nanval,
                                  size_t tile_begin, size_t tile_end) noexcept;

template<typename T>
void reduceDimIntegerMaxRangeAVX2(Param<T> out, CParam<T> in, int dim,
                                  bool change_nan, double nanval,
                                  size_t tile_begin, size_t tile_end) noexcept;

}  // namespace detail
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
