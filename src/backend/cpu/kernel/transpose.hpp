/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <common/complex.hpp>
#include <cpu_features.hpp>
#include <err_cpu.hpp>
#include <kernel/transpose_avx2.hpp>
#include <kernel/transpose_neon.hpp>
#include <parallel.hpp>
#include <utility.hpp>

#include <algorithm>
#include <type_traits>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
T getConjugate(const T &in) {
    // For non-complex types return same
    return in;
}

template<>
cfloat getConjugate(const cfloat &in) {
    return std::conj(in);
}

template<>
cdouble getConjugate(const cdouble &in) {
    return std::conj(in);
}

namespace detail {

static_assert(sizeof(cfloat) == 2 * sizeof(float));
static_assert(sizeof(cdouble) == 2 * sizeof(double));

enum class TransposeKernel { SCALAR, AVX2, NEON };

template<typename T>
constexpr bool isVectorizedTransposeType =
    (std::is_same<T, float>::value || std::is_same<T, double>::value ||
     std::is_same<T, int>::value || std::is_same<T, uint>::value ||
     std::is_same<T, intl>::value || std::is_same<T, uintl>::value ||
     std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value) &&
    (sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16);

template<typename T>
TransposeKernel selectTransposeKernel(dim_t output_x_stride,
                                      dim_t input_x_stride) {
    if constexpr (isVectorizedTransposeType<T>) {
        if (output_x_stride == 1 && input_x_stride == 1) {
            if (arrayfire::cpu::detail::isAVX2Supported() &&
                isTransposeAVX2Compiled()) {
                return TransposeKernel::AVX2;
            }
            if (isTransposeNEONCompiled()) { return TransposeKernel::NEON; }
        }
    }
    return TransposeKernel::SCALAR;
}

}  // namespace detail

template<typename T, bool conjugate>
void transpose_tile_scalar(T *output, const T *input, dim_t output_x_stride,
                           dim_t output_y_stride, dim_t input_x_stride,
                           dim_t input_y_stride) {
    constexpr int tile_size = 8;
    for (int x = 0; x < tile_size; ++x) {
        for (int y = 0; y < tile_size; ++y) {
            if constexpr (conjugate) {
                output[y * output_y_stride] =
                    getConjugate(input[y * input_x_stride]);
            } else {
                output[y * output_y_stride] = input[y * input_x_stride];
            }
        }
        if (x + 1 < tile_size) {
            input += input_y_stride;
            output += output_x_stride;
        }
    }
}

template<typename T, bool conjugate>
void transpose_tile_run(T *output, const T *input, dim_t output_x_stride,
                        dim_t output_y_stride, dim_t input_x_stride,
                        dim_t input_y_stride, size_t tile_count,
                        detail::TransposeKernel transpose_kernel) {
    if constexpr (detail::isVectorizedTransposeType<T>) {
        if (transpose_kernel == detail::TransposeKernel::AVX2) {
            constexpr bool conjugate_values =
                conjugate && common::is_complex<T>::value;
            detail::transposeTileRunAVX2(output, input, output_y_stride,
                                         input_y_stride, tile_count, sizeof(T),
                                         conjugate_values);
            return;
        }
        if (transpose_kernel == detail::TransposeKernel::NEON) {
            constexpr bool conjugate_values =
                conjugate && common::is_complex<T>::value;
            detail::transposeTileRunNEON(output, input, output_y_stride,
                                         input_y_stride, tile_count, sizeof(T),
                                         conjugate_values);
            return;
        }
    }

    constexpr dim_t tile_size = 8;
    for (size_t tile = 0; tile < tile_count; ++tile) {
        transpose_tile_scalar<T, conjugate>(output, input, output_x_stride,
                                            output_y_stride, input_x_stride,
                                            input_y_stride);
        if (tile + 1 < tile_count) {
            output += tile_size * output_x_stride;
            input += tile_size * input_y_stride;
        }
    }
}

template<typename T, bool conjugate>
void transpose_out(Param<T> output, CParam<T> input) {
    const af::dim4 odims    = output.dims();
    const af::dim4 ostrides = output.strides();
    const af::dim4 istrides = input.strides();

    T *const out      = output.get();
    const T *const in = input.get();

    const dim_t output_width  = odims[0];
    const dim_t output_height = odims[1];
    const dim_t depth         = odims[2];
    const dim_t batches       = odims[3];

    const dim_t output_x_stride = ostrides[0];
    const dim_t output_y_stride = ostrides[1];
    const dim_t output_z_stride = ostrides[2];
    const dim_t output_w_stride = ostrides[3];
    const dim_t input_x_stride  = istrides[0];
    const dim_t input_y_stride  = istrides[1];
    const dim_t input_z_stride  = istrides[2];
    const dim_t input_w_stride  = istrides[3];

    constexpr size_t tile_size = 8;
    const size_t x_tiles =
        (static_cast<size_t>(output_width) + tile_size - 1) / tile_size;
    const size_t y_tiles =
        (static_cast<size_t>(output_height) + tile_size - 1) / tile_size;
    const size_t full_x_tiles = static_cast<size_t>(output_width) / tile_size;
    const size_t tile_count   = x_tiles * y_tiles * static_cast<size_t>(depth) *
                              static_cast<size_t>(batches);

    const detail::TransposeKernel transpose_kernel =
        detail::selectTransposeKernel<T>(output_x_stride, input_x_stride);
    const bool all_tiles_full =
        output_width % tile_size == 0 && output_height % tile_size == 0;

    constexpr size_t min_elements_per_task = 1 << 16;
    constexpr size_t tile_elements         = tile_size * tile_size;
    constexpr size_t min_tiles_per_task = min_elements_per_task / tile_elements;

    parallelForRange(
        tile_count, min_tiles_per_task, [=](size_t begin, size_t end) {
            size_t linear = begin;
            size_t tile_x = linear % x_tiles;
            linear /= x_tiles;
            size_t tile_y = linear % y_tiles;
            linear /= y_tiles;
            dim_t z = static_cast<dim_t>(linear % depth);
            dim_t w = static_cast<dim_t>(linear / depth);

            if (all_tiles_full) {
                const T *tile_input =
                    in +
                    static_cast<dim_t>(tile_y * tile_size) * input_x_stride +
                    static_cast<dim_t>(tile_x * tile_size) * input_y_stride +
                    z * input_z_stride + w * input_w_stride;
                T *tile_output =
                    out +
                    static_cast<dim_t>(tile_x * tile_size) * output_x_stride +
                    static_cast<dim_t>(tile_y * tile_size) * output_y_stride +
                    z * output_z_stride + w * output_w_stride;

                size_t tile = begin;
                while (tile < end) {
                    const size_t run = std::min(end - tile, x_tiles - tile_x);
                    transpose_tile_run<T, conjugate>(
                        tile_output, tile_input, output_x_stride,
                        output_y_stride, input_x_stride, input_y_stride, run,
                        transpose_kernel);
                    tile += run;
                    tile_x += run;
                    if (tile == end) { break; }
                    if (tile_x < x_tiles) {
                        tile_input += static_cast<dim_t>(run * tile_size) *
                                      input_y_stride;
                        tile_output += static_cast<dim_t>(run * tile_size) *
                                       output_x_stride;
                        continue;
                    }

                    tile_x = 0;
                    if (++tile_y == y_tiles) {
                        tile_y = 0;
                        if (++z == depth) {
                            z = 0;
                            ++w;
                        }
                    }
                    tile_input = in +
                                 static_cast<dim_t>(tile_y * tile_size) *
                                     input_x_stride +
                                 z * input_z_stride + w * input_w_stride;
                    tile_output = out +
                                  static_cast<dim_t>(tile_y * tile_size) *
                                      output_y_stride +
                                  z * output_z_stride + w * output_w_stride;
                }
                return;
            }

            size_t tile = begin;
            while (tile < end) {
                const dim_t x_begin = static_cast<dim_t>(tile_x * tile_size);
                const dim_t y_begin = static_cast<dim_t>(tile_y * tile_size);
                const dim_t x_end =
                    std::min<dim_t>(output_width, x_begin + tile_size);
                const dim_t y_end =
                    std::min<dim_t>(output_height, y_begin + tile_size);

                const T *tile_input = in + y_begin * input_x_stride +
                                      x_begin * input_y_stride +
                                      z * input_z_stride + w * input_w_stride;
                T *tile_output = out + x_begin * output_x_stride +
                                 y_begin * output_y_stride +
                                 z * output_z_stride + w * output_w_stride;
                size_t run = 1;
                if (x_end - x_begin == tile_size &&
                    y_end - y_begin == tile_size) {
                    run = std::min(end - tile, full_x_tiles - tile_x);
                    transpose_tile_run<T, conjugate>(
                        tile_output, tile_input, output_x_stride,
                        output_y_stride, input_x_stride, input_y_stride, run,
                        transpose_kernel);
                } else {
                    // Read contiguous input rows and scatter within a
                    // cache-sized tile. Reversing these loops makes every
                    // input access strided.
                    for (dim_t x = x_begin; x < x_end; ++x) {
                        const T *input_ptr =
                            in + y_begin * input_x_stride + x * input_y_stride +
                            z * input_z_stride + w * input_w_stride;
                        T *output_ptr = out + x * output_x_stride +
                                        y_begin * output_y_stride +
                                        z * output_z_stride +
                                        w * output_w_stride;
                        for (dim_t y = y_begin; y < y_end; ++y) {
                            if constexpr (conjugate) {
                                *output_ptr = getConjugate(*input_ptr);
                            } else {
                                *output_ptr = *input_ptr;
                            }
                            if (y + 1 < y_end) {
                                input_ptr += input_x_stride;
                                output_ptr += output_y_stride;
                            }
                        }
                    }
                }

                tile += run;
                tile_x += run;
                if (tile_x == x_tiles) {
                    tile_x = 0;
                    if (++tile_y == y_tiles) {
                        tile_y = 0;
                        if (++z == depth) {
                            z = 0;
                            ++w;
                        }
                    }
                }
            }
        });
}

template<typename T, bool conjugate>
void transpose_serial(Param<T> output, CParam<T> input) {
    const af::dim4 odims    = output.dims();
    const af::dim4 ostrides = output.strides();
    const af::dim4 istrides = input.strides();

    T *const out      = output.get();
    const T *const in = input.get();

    const dim_t output_width  = odims[0];
    const dim_t output_height = odims[1];
    const dim_t depth         = odims[2];
    const dim_t batches       = odims[3];

    const dim_t output_x_stride = ostrides[0];
    const dim_t output_y_stride = ostrides[1];
    const dim_t output_z_stride = ostrides[2];
    const dim_t output_w_stride = ostrides[3];
    const dim_t input_x_stride  = istrides[0];
    const dim_t input_y_stride  = istrides[1];
    const dim_t input_z_stride  = istrides[2];
    const dim_t input_w_stride  = istrides[3];

    constexpr dim_t tile_size = 8;
    const dim_t full_width    = output_width / tile_size * tile_size;
    const dim_t full_height   = output_height / tile_size * tile_size;
    const size_t full_x_tiles = static_cast<size_t>(full_width / tile_size);
    const detail::TransposeKernel transpose_kernel =
        detail::selectTransposeKernel<T>(output_x_stride, input_x_stride);

    for (dim_t w = 0; w < batches; ++w) {
        for (dim_t z = 0; z < depth; ++z) {
            T *const output_batch =
                out + z * output_z_stride + w * output_w_stride;
            const T *const input_batch =
                in + z * input_z_stride + w * input_w_stride;

            for (dim_t y = 0; y < full_height; y += tile_size) {
                if (full_x_tiles > 0) {
                    transpose_tile_run<T, conjugate>(
                        output_batch + y * output_y_stride,
                        input_batch + y * input_x_stride, output_x_stride,
                        output_y_stride, input_x_stride, input_y_stride,
                        full_x_tiles, transpose_kernel);
                }

                for (dim_t x = full_width; x < output_width; ++x) {
                    T *output_ptr = output_batch + x * output_x_stride +
                                    y * output_y_stride;
                    const T *input_ptr =
                        input_batch + y * input_x_stride + x * input_y_stride;
                    for (dim_t offset = 0; offset < tile_size; ++offset) {
                        if constexpr (conjugate) {
                            *output_ptr = getConjugate(*input_ptr);
                        } else {
                            *output_ptr = *input_ptr;
                        }
                        if (offset + 1 < tile_size) {
                            output_ptr += output_y_stride;
                            input_ptr += input_x_stride;
                        }
                    }
                }
            }

            for (dim_t y = full_height; y < output_height; ++y) {
                for (dim_t x = 0; x < output_width; ++x) {
                    T *output_ptr = output_batch + x * output_x_stride +
                                    y * output_y_stride;
                    const T *input_ptr =
                        input_batch + y * input_x_stride + x * input_y_stride;
                    if constexpr (conjugate) {
                        *output_ptr = getConjugate(*input_ptr);
                    } else {
                        *output_ptr = *input_ptr;
                    }
                }
            }
        }
    }
}

template<typename T>
void transpose_real(Param<T> output, CParam<T> input) {
    constexpr dim_t min_parallel_elements = 1 << 17;
    if (getParallelThreadCount() == 1 ||
        output.dims().elements() < min_parallel_elements) {
        transpose_serial<T, false>(output, input);
    } else {
        transpose_out<T, false>(output, input);
    }
}

template<typename T>
void transpose_conj(Param<T> output, CParam<T> input) {
    constexpr dim_t min_parallel_elements = 1 << 17;
    if (getParallelThreadCount() == 1 ||
        output.dims().elements() < min_parallel_elements) {
        transpose_serial<T, true>(output, input);
    } else {
        transpose_out<T, true>(output, input);
    }
}

template<typename T>
void transpose(Param<T> out, CParam<T> in, const bool conjugate) {
    return (conjugate ? transpose_conj<T>(out, in)
                      : transpose_real<T>(out, in));
}

template<typename T, bool conjugate>
void transpose_inplace_serial(Param<T> input) {
    const af::dim4 idims    = input.dims();
    const af::dim4 istrides = input.strides();

    T *in = input.get();

    for (dim_t l = 0; l < idims[3]; ++l) {
        for (dim_t k = 0; k < idims[2]; ++k) {
            const dim_t batch_offset = k * istrides[2] + l * istrides[3];
            for (dim_t j = 0; j < idims[1]; ++j) {
                if constexpr (conjugate) {
                    const dim_t diagonal =
                        batch_offset + j * (istrides[0] + istrides[1]);
                    in[diagonal] = getConjugate(in[diagonal]);
                }

                // Run only the bottom triangle. Each value is swapped with
                // its corresponding value in the upper triangle.
                for (dim_t i = j + 1; i < idims[0]; ++i) {
                    const dim_t lower =
                        batch_offset + i * istrides[0] + j * istrides[1];
                    const dim_t upper =
                        batch_offset + j * istrides[0] + i * istrides[1];
                    if constexpr (conjugate) {
                        const T lower_value = in[lower];
                        in[lower]           = getConjugate(in[upper]);
                        in[upper]           = getConjugate(lower_value);
                    } else {
                        std::swap(in[lower], in[upper]);
                    }
                }
            }
        }
    }
}

inline size_t triangularNumber(size_t row) {
    return row % 2 == 0 ? (row / 2) * (row + 1) : row * ((row + 1) / 2);
}

inline size_t triangularRow(size_t index, size_t row_count) {
    size_t low  = 0;
    size_t high = row_count;
    while (low + 1 < high) {
        const size_t middle = low + (high - low) / 2;
        if (triangularNumber(middle) <= index) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

template<typename T, bool conjugate>
void transpose_inplace_tiled(Param<T> input) {
    const af::dim4 idims    = input.dims();
    const af::dim4 istrides = input.strides();
    T *const in             = input.get();

    constexpr size_t tile_size = 32;
    const size_t tile_count =
        (static_cast<size_t>(idims[0]) + tile_size - 1) / tile_size;
    const size_t pairs_per_matrix = triangularNumber(tile_count);
    const size_t matrix_count =
        static_cast<size_t>(idims[2]) * static_cast<size_t>(idims[3]);
    const size_t pair_count = pairs_per_matrix * matrix_count;

    constexpr size_t min_elements_per_task = 1 << 16;
    constexpr size_t tile_elements         = tile_size * tile_size;
    constexpr size_t min_pairs_per_task = min_elements_per_task / tile_elements;

    parallelForRange(
        pair_count, min_pairs_per_task, [=](size_t begin, size_t end) {
            size_t matrix           = begin / pairs_per_matrix;
            const size_t pair_index = begin % pairs_per_matrix;
            size_t tile_row         = triangularRow(pair_index, tile_count);
            size_t tile_column      = pair_index - triangularNumber(tile_row);
            dim_t z                 = static_cast<dim_t>(matrix % idims[2]);
            dim_t w                 = static_cast<dim_t>(matrix / idims[2]);

            while (begin < end) {
                const dim_t row_begin =
                    static_cast<dim_t>(tile_row * tile_size);
                const dim_t row_end =
                    std::min<dim_t>(idims[0], row_begin + tile_size);
                const dim_t column_begin =
                    static_cast<dim_t>(tile_column * tile_size);
                const dim_t column_end =
                    std::min<dim_t>(idims[1], column_begin + tile_size);
                const dim_t batch_offset = z * istrides[2] + w * istrides[3];

                if (tile_row == tile_column) {
                    for (dim_t y = row_begin; y < row_end; ++y) {
                        if constexpr (conjugate) {
                            const dim_t diagonal =
                                batch_offset + y * (istrides[0] + istrides[1]);
                            in[diagonal] = getConjugate(in[diagonal]);
                        }
                        for (dim_t x = y + 1; x < row_end; ++x) {
                            const dim_t lower = batch_offset + x * istrides[0] +
                                                y * istrides[1];
                            const dim_t upper = batch_offset + y * istrides[0] +
                                                x * istrides[1];
                            if constexpr (conjugate) {
                                const T lower_value = in[lower];
                                in[lower]           = getConjugate(in[upper]);
                                in[upper]           = getConjugate(lower_value);
                            } else {
                                std::swap(in[lower], in[upper]);
                            }
                        }
                    }
                } else {
                    for (dim_t y = column_begin; y < column_end; ++y) {
                        for (dim_t x = row_begin; x < row_end; ++x) {
                            const dim_t lower = batch_offset + x * istrides[0] +
                                                y * istrides[1];
                            const dim_t upper = batch_offset + y * istrides[0] +
                                                x * istrides[1];
                            if constexpr (conjugate) {
                                const T lower_value = in[lower];
                                in[lower]           = getConjugate(in[upper]);
                                in[upper]           = getConjugate(lower_value);
                            } else {
                                std::swap(in[lower], in[upper]);
                            }
                        }
                    }
                }

                ++begin;
                if (++tile_column > tile_row) {
                    tile_column = 0;
                    if (++tile_row == tile_count) {
                        tile_row = 0;
                        if (++z == idims[2]) {
                            z = 0;
                            ++w;
                        }
                    }
                }
            }
        });
}

template<typename T>
void transpose_inplace(Param<T> in, const bool conjugate) {
    constexpr dim_t min_tiled_elements = 1 << 17;
    if (in.dims().elements() < min_tiled_elements) {
        return (conjugate ? transpose_inplace_serial<T, true>(in)
                          : transpose_inplace_serial<T, false>(in));
    }
    return (conjugate ? transpose_inplace_tiled<T, true>(in)
                      : transpose_inplace_tiled<T, false>(in));
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
