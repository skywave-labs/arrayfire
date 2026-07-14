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
void transpose_inplace(Param<T> input) {
    const af::dim4 idims    = input.dims();
    const af::dim4 istrides = input.strides();

    T *in = input.get();

    for (dim_t l = 0; l < idims[3]; ++l) {
        for (dim_t k = 0; k < idims[2]; ++k) {
            // Outermost loop handles batch mode
            // if input has no data along third dimension
            // this loop runs only once
            //
            // Run only bottom triangle. std::swap swaps with upper triangle
            for (dim_t j = 0; j < idims[1]; ++j) {
                for (dim_t i = j + 1; i < idims[0]; ++i) {
                    // calculate array indices based on offsets and strides
                    // the helper getIdx takes care of indices
                    const dim_t iIdx = getIdx(istrides, j, i, k, l);
                    const dim_t oIdx = getIdx(istrides, i, j, k, l);
                    if (conjugate) {
                        in[iIdx] = getConjugate(in[iIdx]);
                        in[oIdx] = getConjugate(in[oIdx]);
                        std::swap(in[iIdx], in[oIdx]);
                    } else {
                        std::swap(in[iIdx], in[oIdx]);
                    }
                }
            }
        }
    }
}

template<typename T>
void transpose_inplace(Param<T> in, const bool conjugate) {
    return (conjugate ? transpose_inplace<T, true>(in)
                      : transpose_inplace<T, false>(in));
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
