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
#include <err_cpu.hpp>
#include <parallel.hpp>
#include <utility.hpp>

#include <algorithm>

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

template<typename T, bool conjugate>
void transpose_tile(T *output, const T *input, dim_t output_y_stride,
                    dim_t input_y_stride) {
    constexpr int tile_size = 8;
    for (int x = 0; x < tile_size; ++x) {
        for (int y = 0; y < tile_size; ++y) {
            if constexpr (conjugate) {
                output[y * output_y_stride] = getConjugate(input[y]);
            } else {
                output[y * output_y_stride] = input[y];
            }
        }
        input += input_y_stride;
        ++output;
    }
}

template<typename T, bool conjugate>
void transpose_out(Param<T> output, CParam<T> input) {
    const af::dim4 odims    = output.dims();
    const af::dim4 ostrides = output.strides();
    const af::dim4 istrides = input.strides();

    T *const out      = output.get();
    const T *const in = input.get();

    constexpr size_t tile_size = 8;
    const size_t x_tiles =
        (static_cast<size_t>(odims[0]) + tile_size - 1) / tile_size;
    const size_t y_tiles =
        (static_cast<size_t>(odims[1]) + tile_size - 1) / tile_size;
    const size_t tile_count = x_tiles * y_tiles *
                              static_cast<size_t>(odims[2]) *
                              static_cast<size_t>(odims[3]);

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
            dim_t z = static_cast<dim_t>(linear % odims[2]);
            dim_t w = static_cast<dim_t>(linear / odims[2]);

            if (odims[0] % tile_size == 0 && odims[1] % tile_size == 0) {
                const dim_t input_tile_stride  = tile_size * istrides[1];
                const dim_t output_tile_stride = tile_size * ostrides[0];
                const T *tile_input =
                    in + static_cast<dim_t>(tile_y * tile_size) * istrides[0] +
                    static_cast<dim_t>(tile_x * tile_size) * istrides[1] +
                    z * istrides[2] + w * istrides[3];
                T *tile_output =
                    out + static_cast<dim_t>(tile_x * tile_size) * ostrides[0] +
                    static_cast<dim_t>(tile_y * tile_size) * ostrides[1] +
                    z * ostrides[2] + w * ostrides[3];

                for (size_t tile = begin; tile < end; ++tile) {
                    transpose_tile<T, conjugate>(tile_output, tile_input,
                                                 ostrides[1], istrides[1]);
                    if (++tile_x < x_tiles) {
                        tile_input += input_tile_stride;
                        tile_output += output_tile_stride;
                        continue;
                    }

                    tile_x = 0;
                    if (++tile_y == y_tiles) {
                        tile_y = 0;
                        if (++z == odims[2]) {
                            z = 0;
                            ++w;
                        }
                    }
                    tile_input =
                        in +
                        static_cast<dim_t>(tile_y * tile_size) * istrides[0] +
                        z * istrides[2] + w * istrides[3];
                    tile_output =
                        out +
                        static_cast<dim_t>(tile_y * tile_size) * ostrides[1] +
                        z * ostrides[2] + w * ostrides[3];
                }
                return;
            }

            for (size_t tile = begin; tile < end; ++tile) {
                const dim_t x_begin = static_cast<dim_t>(tile_x * tile_size);
                const dim_t y_begin = static_cast<dim_t>(tile_y * tile_size);
                const dim_t x_end =
                    std::min<dim_t>(odims[0], x_begin + tile_size);
                const dim_t y_end =
                    std::min<dim_t>(odims[1], y_begin + tile_size);

                const T *tile_input = in + y_begin * istrides[0] +
                                      x_begin * istrides[1] + z * istrides[2] +
                                      w * istrides[3];
                T *tile_output = out + x_begin * ostrides[0] +
                                 y_begin * ostrides[1] + z * ostrides[2] +
                                 w * ostrides[3];
                if (x_end - x_begin == tile_size &&
                    y_end - y_begin == tile_size) {
                    transpose_tile<T, conjugate>(tile_output, tile_input,
                                                 ostrides[1], istrides[1]);
                } else {
                    // Read contiguous input rows and scatter within a
                    // cache-sized tile. Reversing these loops makes every
                    // input access strided.
                    for (dim_t x = x_begin; x < x_end; ++x) {
                        const T *input_ptr = in + y_begin * istrides[0] +
                                             x * istrides[1] + z * istrides[2] +
                                             w * istrides[3];
                        T *output_ptr = out + x * ostrides[0] +
                                        y_begin * ostrides[1] +
                                        z * ostrides[2] + w * ostrides[3];
                        for (dim_t y = y_begin; y < y_end; ++y) {
                            if constexpr (conjugate) {
                                *output_ptr = getConjugate(*input_ptr);
                            } else {
                                *output_ptr = *input_ptr;
                            }
                            input_ptr += istrides[0];
                            output_ptr += ostrides[1];
                        }
                    }
                }

                if (++tile_x == x_tiles) {
                    tile_x = 0;
                    if (++tile_y == y_tiles) {
                        tile_y = 0;
                        if (++z == odims[2]) {
                            z = 0;
                            ++w;
                        }
                    }
                }
            }
        });
}

template<typename T>
void transpose_real_serial(Param<T> output, CParam<T> input) {
    const af::dim4 odims    = output.dims();
    const af::dim4 ostrides = output.strides();
    const af::dim4 istrides = input.strides();

    T *out            = output.get();
    T const *const in = input.get();

    constexpr int tile_size = 8;
    const dim_t odims1_down = floor(odims[1] / tile_size) * tile_size;
    const dim_t odims0_down = floor(odims[0] / tile_size) * tile_size;

    for (dim_t l = 0; l < odims[3]; ++l) {
        for (dim_t k = 0; k < odims[2]; ++k) {
            T *out_      = out + l * ostrides[3] + k * ostrides[2];
            const T *in_ = in + l * istrides[3] + k * istrides[2];

            if (odims1_down > 0) {
                for (dim_t j = 0; j <= odims1_down; j += tile_size) {
                    for (dim_t i = 0; i < odims0_down; i += tile_size) {
                        transpose_tile<T, false>(out_, in_, ostrides[1],
                                                 istrides[1]);
                        out_ += tile_size;
                        in_ += istrides[1] * tile_size;
                    }

                    for (dim_t jj = 0; jj < tile_size; ++jj) {
                        for (dim_t i = odims0_down; i < odims[0]; ++i) {
                            *out_ = *in_;
                            ++out_;
                            in_ += istrides[1];
                        }
                        out_ += ostrides[1] - (odims[0] - odims0_down);
                        in_ -= (odims[0] - odims0_down) * istrides[1] - 1;
                    }
                    out_ = out + l * ostrides[3] + k * ostrides[2] +
                           j * ostrides[1];
                    in_ = in + l * istrides[3] + k * istrides[2] + j;
                }
            }
            for (dim_t j = odims1_down; j < odims[1]; ++j) {
                out_ =
                    out + l * ostrides[3] + k * ostrides[2] + j * ostrides[1];
                in_ = in + l * istrides[3] + k * istrides[2] + j;
                for (dim_t i = 0; i < odims[0]; ++i) {
                    *out_ = *in_;
                    ++out_;
                    in_ += istrides[1];
                }
            }
        }
    }
}

template<typename T>
void transpose_conj_serial(Param<T> output, CParam<T> input) {
    const af::dim4 odims    = output.dims();
    const af::dim4 ostrides = output.strides();
    const af::dim4 istrides = input.strides();

    T *out            = output.get();
    T const *const in = input.get();
    for (dim_t l = 0; l < odims[3]; ++l) {
        for (dim_t k = 0; k < odims[2]; ++k) {
            for (dim_t j = 0; j < odims[1]; ++j) {
                for (dim_t i = 0; i < odims[0]; ++i) {
                    const dim_t in_idx  = getIdx(istrides, j, i, k, l);
                    const dim_t out_idx = getIdx(ostrides, i, j, k, l);
                    out[out_idx]        = getConjugate(in[in_idx]);
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
        transpose_real_serial(output, input);
    } else {
        transpose_out<T, false>(output, input);
    }
}

template<typename T>
void transpose_conj(Param<T> output, CParam<T> input) {
    constexpr dim_t min_parallel_elements = 1 << 17;
    if (getParallelThreadCount() == 1 ||
        output.dims().elements() < min_parallel_elements) {
        transpose_conj_serial(output, input);
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
