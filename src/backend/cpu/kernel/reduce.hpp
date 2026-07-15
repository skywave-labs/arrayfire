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
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <common/half.hpp>
#include <cpu_features.hpp>
#include <kernel/reduce_dim_avx2.hpp>
#include <parallel.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#define AF_CPU_REDUCE_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define AF_CPU_REDUCE_NOINLINE __attribute__((noinline))
#else
#define AF_CPU_REDUCE_NOINLINE
#endif

namespace arrayfire {
namespace cpu {
namespace kernel {

namespace detail {

constexpr size_t minAVX2DimensionalInputElements = 1U << 12U;

template<typename Ti, typename To>
constexpr bool isAVX2IntegralArithmeticPair =
    (std::is_same<Ti, int>::value && std::is_same<To, int>::value) ||
    (std::is_same<Ti, uint>::value && std::is_same<To, uint>::value) ||
    (std::is_same<Ti, intl>::value && std::is_same<To, intl>::value) ||
    (std::is_same<Ti, uintl>::value && std::is_same<To, uintl>::value) ||
    (std::is_same<Ti, schar>::value && std::is_same<To, int>::value) ||
    (std::is_same<Ti, uchar>::value && std::is_same<To, uint>::value) ||
    (std::is_same<Ti, short>::value && std::is_same<To, int>::value) ||
    (std::is_same<Ti, ushort>::value && std::is_same<To, uint>::value);

template<typename T>
constexpr bool isAVX2IntegralExtremaType =
    std::is_same<T, schar>::value || std::is_same<T, uchar>::value ||
    std::is_same<T, short>::value || std::is_same<T, ushort>::value ||
    std::is_same<T, int>::value || std::is_same<T, uint>::value ||
    std::is_same<T, intl>::value || std::is_same<T, uintl>::value;

template<typename Ti, typename To>
constexpr bool isAVX2SumPair =
    (std::is_same<Ti, To>::value &&
     (std::is_same<Ti, float>::value || std::is_same<Ti, double>::value ||
      std::is_same<Ti, cfloat>::value || std::is_same<Ti, cdouble>::value)) ||
    isAVX2IntegralArithmeticPair<Ti, To>;

template<typename Ti, typename To>
constexpr bool isAVX2ProductPair =
    (std::is_same<Ti, To>::value &&
     (std::is_same<Ti, float>::value || std::is_same<Ti, double>::value)) ||
    isAVX2IntegralArithmeticPair<Ti, To>;

template<typename Ti, typename To>
constexpr bool isAVX2ExtremaPair =
    std::is_same<Ti, To>::value &&
    (std::is_same<Ti, float>::value || std::is_same<Ti, double>::value ||
     isAVX2IntegralExtremaType<Ti>);

template<af_op_t op, typename Ti, typename To>
constexpr bool isAVX2ReduceSupported =
    (op == af_add_t && isAVX2SumPair<Ti, To>) ||
    (op == af_mul_t && isAVX2ProductPair<Ti, To>) ||
    ((op == af_min_t || op == af_max_t) && isAVX2ExtremaPair<Ti, To>);

template<af_op_t op, typename Ti, typename To>
AF_CPU_REDUCE_NOINLINE bool tryReduceDimAVX2(
    Param<To> out, CParam<Ti> in, const int dim, bool change_nan, double nanval,
    const af::dim4 &odims, const af::dim4 &ostrides, const af::dim4 &idims,
    const af::dim4 &istrides) {
    constexpr bool supported_sum = op == af_add_t && isAVX2SumPair<Ti, To>;
    constexpr bool supported_product =
        op == af_mul_t && isAVX2ProductPair<Ti, To>;
    constexpr bool supported_minimum =
        op == af_min_t && isAVX2ExtremaPair<Ti, To>;
    constexpr bool supported_maximum =
        op == af_max_t && isAVX2ExtremaPair<Ti, To>;
    constexpr bool supported_integral =
        isAVX2IntegralArithmeticPair<Ti, To> ||
        (std::is_same<Ti, To>::value && isAVX2IntegralExtremaType<Ti>);

    if constexpr (!supported_sum && !supported_product && !supported_minimum &&
                  !supported_maximum) {
        return false;
    } else {
        constexpr size_t vector_bytes             = 32;
        constexpr size_t tile_bytes               = 128;
        constexpr size_t vector_elements          = vector_bytes / sizeof(To);
        constexpr size_t tile_elements            = tile_bytes / sizeof(To);
        constexpr size_t target_elements_per_task = 1U << 16U;

        if (dim < 1 || dim > 3 ||
            odims[0] < static_cast<dim_t>(vector_elements) || idims[dim] < 2 ||
            istrides[0] != 1 || ostrides[0] != 1) {
            return false;
        }

        if (static_cast<size_t>(idims.elements()) <
            minAVX2DimensionalInputElements) {
            return false;
        }

        if (!arrayfire::cpu::detail::isAVX2Supported() ||
            !isReduceDimAVX2Compiled()) {
            return false;
        }

        const size_t width            = static_cast<size_t>(odims[0]);
        const size_t reduced_elements = static_cast<size_t>(idims[dim]);

        const size_t x_tiles = 1 + (width - 1) / tile_elements;
        size_t row_count     = 1;
        for (int axis = 1; axis < 4; ++axis) {
            const size_t axis_elements = static_cast<size_t>(odims[axis]);
            if (axis_elements != 0 &&
                row_count >
                    std::numeric_limits<size_t>::max() / axis_elements) {
                return false;
            }
            row_count *= axis_elements;
        }
        if (row_count > std::numeric_limits<size_t>::max() / x_tiles) {
            return false;
        }
        const size_t tile_count = x_tiles * row_count;

        const size_t work_per_tile =
            reduced_elements >
                    std::numeric_limits<size_t>::max() / tile_elements
                ? std::numeric_limits<size_t>::max()
                : reduced_elements * tile_elements;
        const size_t min_tiles_per_task =
            work_per_tile >= target_elements_per_task
                ? 1
                : 1 + (target_elements_per_task - 1) / work_per_tile;

        parallelForRange(
            tile_count, min_tiles_per_task,
            [=](const size_t begin, const size_t end) {
                if constexpr (supported_sum && supported_integral) {
                    reduceDimIntegerSumRangeAVX2(out, in, dim, change_nan,
                                                 nanval, begin, end);
                } else if constexpr (supported_product && supported_integral) {
                    reduceDimIntegerProductRangeAVX2(out, in, dim, change_nan,
                                                     nanval, begin, end);
                } else if constexpr (supported_minimum && supported_integral) {
                    reduceDimIntegerMinRangeAVX2(out, in, dim, change_nan,
                                                 nanval, begin, end);
                } else if constexpr (supported_maximum && supported_integral) {
                    reduceDimIntegerMaxRangeAVX2(out, in, dim, change_nan,
                                                 nanval, begin, end);
                } else if constexpr (supported_sum) {
                    reduceDimSumRangeAVX2(out, in, dim, change_nan, nanval,
                                          begin, end);
                } else if constexpr (supported_product) {
                    reduceDimProductRangeAVX2(out, in, dim, change_nan, nanval,
                                              begin, end);
                } else if constexpr (supported_minimum) {
                    reduceDimMinRangeAVX2(out, in, dim, change_nan, nanval,
                                          begin, end);
                } else {
                    reduceDimMaxRangeAVX2(out, in, dim, change_nan, nanval,
                                          begin, end);
                }
            });
        return true;
    }
}

}  // namespace detail

template<af_op_t op, typename Ti, typename To, int D>
struct reduce_dim {
    void operator()(Param<To> out, const dim_t outOffset, CParam<Ti> in,
                    const dim_t inOffset, const int dim, bool change_nan,
                    double nanval) {
        static const int D1 = D - 1;
        reduce_dim<op, Ti, To, D1> reduce_dim_next;

        const af::dim4 ostrides = out.strides();
        const af::dim4 istrides = in.strides();
        const af::dim4 odims    = out.dims();

        for (dim_t i = 0; i < odims[D1]; i++) {
            reduce_dim_next(out, outOffset + i * ostrides[D1], in,
                            inOffset + i * istrides[D1], dim, change_nan,
                            nanval);
        }
    }
};

template<af_op_t op, typename Ti, typename To>
struct reduce_dim<op, Ti, To, 0> {
    common::Transform<data_t<Ti>, compute_t<To>, op> transform;
    common::Binary<compute_t<To>, op> reduce;
    void operator()(Param<To> out, const dim_t outOffset, CParam<Ti> in,
                    const dim_t inOffset, const int dim, bool change_nan,
                    double nanval) {
        const af::dim4 istrides = in.strides();
        const af::dim4 idims    = in.dims();

        data_t<To> *const outPtr      = out.get() + outOffset;
        data_t<Ti> const *const inPtr = in.get() + inOffset;
        dim_t stride                  = istrides[dim];

        compute_t<To> out_val = common::Binary<compute_t<To>, op>::init();
        for (dim_t i = 0; i < idims[dim]; i++) {
            compute_t<To> in_val = transform(inPtr[i * stride]);
            if (change_nan) in_val = IS_NAN(in_val) ? nanval : in_val;
            out_val = reduce(in_val, out_val);
        }

        *outPtr = data_t<To>(out_val);
    }
};

template<af_op_t op, typename Ti, typename To>
void reduce_dim_parallel(Param<To> out, CParam<Ti> in, const int dim,
                         bool change_nan, double nanval) {
    const af::dim4 odims    = out.dims();
    const af::dim4 ostrides = out.strides();
    const af::dim4 idims    = in.dims();
    const af::dim4 istrides = in.strides();
    if constexpr (detail::isAVX2ReduceSupported<op, Ti, To>) {
        if (static_cast<size_t>(idims.elements()) >=
                detail::minAVX2DimensionalInputElements &&
            detail::tryReduceDimAVX2<op, Ti, To>(out, in, dim, change_nan,
                                                 nanval, odims, ostrides, idims,
                                                 istrides)) {
            return;
        }
    }

    const size_t reduced_elements =
        std::max<size_t>(1, static_cast<size_t>(idims[dim]));

    constexpr size_t min_input_elements_per_task = 1 << 16;
    const size_t min_lines_per_task =
        std::max<size_t>(1, min_input_elements_per_task / reduced_elements);
    parallelForEach(
        odims, min_lines_per_task, [=](dim_t x, dim_t y, dim_t z, dim_t w) {
            const dim_t out_offset = x * ostrides[0] + y * ostrides[1] +
                                     z * ostrides[2] + w * ostrides[3];
            const dim_t in_offset = x * istrides[0] + y * istrides[1] +
                                    z * istrides[2] + w * istrides[3];
            reduce_dim<op, Ti, To, 0> reduce_line;
            reduce_line(out, out_offset, in, in_offset, dim, change_nan,
                        nanval);
        });
}

template<typename Tk>
void n_reduced_keys(Param<Tk> okeys, int *n_reduced, CParam<Tk> keys) {
    const af::dim4 kdims = keys.dims();

    Tk *const outKeysPtr      = okeys.get();
    Tk const *const inKeysPtr = keys.get();

    int nkeys      = 0;
    Tk current_key = inKeysPtr[0];
    for (dim_t i = 0; i < kdims[0]; i++) {
        Tk keyval = inKeysPtr[i];

        if (keyval != current_key) {
            outKeysPtr[nkeys] = current_key;
            current_key       = keyval;
            ++nkeys;
        }

        if (i == (kdims[0] - 1)) { outKeysPtr[nkeys] = current_key; }
    }

    *n_reduced = nkeys + 1;
}

template<af_op_t op, typename Ti, typename Tk, typename To, int D>
struct reduce_dim_by_key {
    void operator()(Param<To> ovals, const dim_t ovOffset, CParam<Tk> keys,
                    CParam<Ti> vals, const dim_t vOffset, int *n_reduced,
                    const int dim, bool change_nan, double nanval) {
        static const int D1 = D - 1;
        reduce_dim_by_key<op, Ti, Tk, To, D1> reduce_by_key_dim_next;

        const af::dim4 ovstrides = ovals.strides();
        const af::dim4 vstrides  = vals.strides();
        const af::dim4 vdims     = ovals.dims();

        if (D1 == dim) {
            reduce_by_key_dim_next(ovals, ovOffset, keys, vals, vOffset,
                                   n_reduced, dim, change_nan, nanval);
        } else {
            for (dim_t i = 0; i < vdims[D1]; i++) {
                reduce_by_key_dim_next(ovals, ovOffset + (i * ovstrides[D1]),
                                       keys, vals, vOffset + (i * vstrides[D1]),
                                       n_reduced, dim, change_nan, nanval);
            }
        }
    }
};

template<af_op_t op, typename Ti, typename Tk, typename To>
struct reduce_dim_by_key<op, Ti, Tk, To, 0> {
    common::Transform<data_t<Ti>, compute_t<To>, op> transform;
    common::Binary<compute_t<To>, op> reduce;
    void operator()(Param<To> ovals, const dim_t ovOffset, CParam<Tk> keys,
                    CParam<Ti> vals, const dim_t vOffset, int *n_reduced,
                    const int dim, bool change_nan, double nanval) {
        const af::dim4 vstrides = vals.strides();
        const af::dim4 vdims    = vals.dims();

        const af::dim4 ovstrides = ovals.strides();

        data_t<Tk> const *const inKeysPtr = keys.get();
        data_t<Ti> const *const inValsPtr = vals.get();
        data_t<To> *const outValsPtr      = ovals.get();

        int keyidx                = 0;
        compute_t<Tk> current_key = compute_t<Tk>(inKeysPtr[0]);
        compute_t<To> out_val     = reduce.init();

        dim_t istride = vstrides[dim];
        dim_t ostride = ovstrides[dim];

        for (dim_t i = 0; i < vdims[dim]; i++) {
            compute_t<Tk> keyval = inKeysPtr[i];

            if (keyval == current_key) {
                compute_t<To> in_val =
                    transform(inValsPtr[vOffset + (i * istride)]);
                if (change_nan) in_val = IS_NAN(in_val) ? nanval : in_val;
                out_val = reduce(in_val, out_val);

            } else {
                outValsPtr[ovOffset + (keyidx * ostride)] = out_val;

                current_key = keyval;
                out_val     = transform(inValsPtr[vOffset + (i * istride)]);
                if (change_nan) out_val = IS_NAN(out_val) ? nanval : out_val;
                ++keyidx;
            }

            if (i == (vdims[dim] - 1)) {
                outValsPtr[ovOffset + (keyidx * ostride)] = out_val;
            }
        }
    }
};

template<af_op_t op, typename Ti, typename To>
struct reduce_all {
    common::Transform<data_t<Ti>, compute_t<To>, op> transform;
    common::Binary<compute_t<To>, op> reduce;

    static constexpr bool exact_parallel_reduction =
        op == af_min_t || op == af_max_t || op == af_notzero_t ||
        op == af_or_t || op == af_and_t;

    static compute_t<To> reduceRange(CParam<Ti> in, const size_t begin,
                                     const size_t end, const bool is_linear,
                                     const bool change_nan,
                                     const double nanval) {
        const af::dim4 dims            = in.dims();
        const af::dim4 strides         = in.strides();
        const data_t<Ti> *const in_ptr = in.get();
        common::Transform<data_t<Ti>, compute_t<To>, op> local_transform;
        common::Binary<compute_t<To>, op> local_reduce;
        compute_t<To> value =
            common::Binary<compute_t<To>, op>::init();

        const auto accumulate = [&](const data_t<Ti> input) {
            compute_t<To> input_value = local_transform(input);
            if (change_nan) {
                input_value = IS_NAN(input_value) ? nanval : input_value;
            }
            value = local_reduce(input_value, value);

            if constexpr (op == af_or_t) {
                return value != compute_t<To>(0);
            } else if constexpr (op == af_and_t) {
                return value == compute_t<To>(0);
            } else {
                return false;
            }
        };

        if (is_linear) {
            for (size_t item = begin; item < end; ++item) {
                if (accumulate(in_ptr[item])) { break; }
            }
            return value;
        }

        size_t linear = begin;
        std::array<dim_t, 4> coord;
        for (int dim = 0; dim < 4; ++dim) {
            coord[dim] =
                static_cast<dim_t>(linear % static_cast<size_t>(dims[dim]));
            linear /= static_cast<size_t>(dims[dim]);
        }

        for (size_t item = begin; item < end; ++item) {
            // Keep the existing reduce_all assumption that stride 0 is one.
            const dim_t offset = coord[0] + coord[1] * strides[1] +
                                 coord[2] * strides[2] +
                                 coord[3] * strides[3];
            if (accumulate(in_ptr[offset])) { break; }

            for (int dim = 0; dim < 4; ++dim) {
                if (++coord[dim] < dims[dim]) { break; }
                coord[dim] = 0;
            }
        }
        return value;
    }

    void operator()(Param<To> out, CParam<Ti> in, bool change_nan,
                    double nanval) {
        // Decrement dimension of select dimension
        af::dim4 dims            = in.dims();
        af::dim4 strides         = in.strides();
        const data_t<Ti> *inPtr  = in.get();
        data_t<To> *const outPtr = out.get();

        if constexpr (exact_parallel_reduction) {
            constexpr size_t block_elements        = 1 << 16;
            constexpr size_t min_parallel_elements = 1 << 17;
            constexpr size_t max_blocks            = 1024;
            const size_t elements = static_cast<size_t>(dims.elements());

            if (elements >= min_parallel_elements &&
                getParallelThreadCount() > 1) {
                const size_t requested_blocks =
                    1 + (elements - 1) / block_elements;
                const size_t block_count =
                    std::min(max_blocks, requested_blocks);
                std::vector<compute_t<To>> partials(block_count);

                dim_t contiguous_elements = 1;
                bool is_linear            = true;
                for (int dim = 0; dim < 4; ++dim) {
                    if (dims[dim] > 1 && strides[dim] != contiguous_elements) {
                        is_linear = false;
                        break;
                    }
                    contiguous_elements *= dims[dim];
                }

                parallelForRange(
                    block_count, 1,
                    [&](const size_t block_begin, const size_t block_end) {
                        const size_t elements_per_block =
                            elements / block_count;
                        const size_t remainder = elements % block_count;
                        for (size_t block = block_begin; block < block_end;
                             ++block) {
                            const size_t begin =
                                block * elements_per_block +
                                std::min(block, remainder);
                            const size_t end =
                                begin + elements_per_block +
                                (block < remainder ? 1 : 0);
                            partials[block] =
                                reduceRange(in, begin, end, is_linear,
                                            change_nan, nanval);
                        }
                    });

                compute_t<To> out_val =
                    common::Binary<compute_t<To>, op>::init();
                for (const compute_t<To> partial : partials) {
                    out_val = reduce(partial, out_val);
                }
                *outPtr = data_t<To>(out_val);
                return;
            }
        }

        compute_t<To> out_val = common::Binary<compute_t<To>, op>::init();

        for (dim_t l = 0; l < dims[3]; l++) {
            dim_t off3 = l * strides[3];

            for (dim_t k = 0; k < dims[2]; k++) {
                dim_t off2 = k * strides[2];

                for (dim_t j = 0; j < dims[1]; j++) {
                    dim_t off1 = j * strides[1];

                    for (dim_t i = 0; i < dims[0]; i++) {
                        dim_t idx = i + off1 + off2 + off3;

                        compute_t<To> in_val = transform(inPtr[idx]);
                        if (change_nan) {
                            in_val = IS_NAN(in_val) ? nanval : in_val;
                        }
                        out_val = reduce(in_val, out_val);
                    }
                }
            }
        }

        *outPtr = data_t<To>(out_val);
    }
};

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire

#undef AF_CPU_REDUCE_NOINLINE
