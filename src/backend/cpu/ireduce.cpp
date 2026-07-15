/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <ireduce.hpp>
#include <kernel/ireduce.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <parallel.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace cpu {
namespace {

// Require at least two full worker tasks before paying dispatch overhead.
constexpr size_t min_parallel_dimensional_elements = 1 << 17;

bool shouldParallelizeDimensionalIReduce(const dim4 &idims,
                                         const dim4 &odims) {
    return static_cast<size_t>(idims.elements()) >=
               min_parallel_dimensional_elements &&
           odims.elements() > 1 && getParallelThreadCount() > 1;
}

struct RaggedRange {
    size_t begin;
    size_t end;
    size_t work;
};

std::array<dim_t, 4> lineCoordinates(const dim4 &dims, size_t line) {
    std::array<dim_t, 4> coord;
    for (int dim = 0; dim < 4; ++dim) {
        coord[dim] =
            static_cast<dim_t>(line % static_cast<size_t>(dims[dim]));
        line /= static_cast<size_t>(dims[dim]);
    }
    return coord;
}

void advanceLineCoordinates(std::array<dim_t, 4> &coord, const dim4 &dims) {
    for (int dim = 0; dim < 4; ++dim) {
        if (++coord[dim] < dims[dim]) { return; }
        coord[dim] = 0;
    }
}

dim_t coordinateOffset(const std::array<dim_t, 4> &coord,
                       const dim4 &strides) {
    return coord[0] * strides[0] + coord[1] * strides[1] +
           coord[2] * strides[2] + coord[3] * strides[3];
}

dim_t raggedLimit(CParam<uint> rlen, const dim_t rlen_offset,
                  const dim_t reduced_elements) {
    return std::min(reduced_elements,
                    static_cast<dim_t>(rlen.get()[rlen_offset]));
}

size_t raggedLineWork(const dim_t lim) {
    // A zero-length line still initializes its result from element zero.
    return std::max<size_t>(1, static_cast<size_t>(lim));
}

size_t raggedWorkAtLine(CParam<uint> rlen, const dim4 &odims,
                        const dim4 &rstrides, const dim_t reduced_elements,
                        const size_t line, const bool rlen_is_linear) {
    const dim_t rlen_offset =
        rlen_is_linear
            ? static_cast<dim_t>(line)
            : coordinateOffset(lineCoordinates(odims, line), rstrides);
    return raggedLineWork(
        raggedLimit(rlen, rlen_offset, reduced_elements));
}

bool isClearlyShortRaggedReduction(CParam<uint> rlen, const dim4 &odims,
                                   const dim4 &rstrides,
                                   const dim_t reduced_elements,
                                   const bool rlen_is_linear) {
    constexpr size_t sample_count          = 64;
    constexpr size_t min_parallel_work     = 1 << 17;
    constexpr size_t short_work_confidence = 3;
    constexpr size_t confidence_scale      = 4;

    const size_t line_count = static_cast<size_t>(odims.elements());
    if (line_count <= sample_count) { return false; }

    size_t sampled_work = 0;
    size_t max_work     = 0;
    const auto stratum_boundary = [line_count](const size_t boundary) {
        return (line_count / sample_count) * boundary +
               (line_count % sample_count) * boundary / sample_count;
    };
    for (size_t sample = 0; sample < sample_count; ++sample) {
        const size_t begin = stratum_boundary(sample);
        const size_t end   = stratum_boundary(sample + 1);

        // SplitMix64 provides deterministic jitter within every stratum to
        // decorrelate the probes from common periodic length patterns.
        uint64_t sample_hash = sample + UINT64_C(0x9E3779B97F4A7C15);
        sample_hash = (sample_hash ^ (sample_hash >> 30)) *
                      UINT64_C(0xBF58476D1CE4E5B9);
        sample_hash = (sample_hash ^ (sample_hash >> 27)) *
                      UINT64_C(0x94D049BB133111EB);
        sample_hash ^= sample_hash >> 31;
        const size_t line =
            begin + static_cast<size_t>(sample_hash % (end - begin));
        const size_t work = raggedWorkAtLine(
            rlen, odims, rstrides, reduced_elements, line, rlen_is_linear);
        sampled_work += work;
        max_work = std::max(max_work, work);
    }

    // Only bypass the exact scan when the estimate is comfortably below the
    // dispatch threshold or nearly all sampled work belongs to one line.
    constexpr size_t chunk_work = 1 << 16;
    constexpr size_t short_parallel_work =
        min_parallel_work * short_work_confidence / confidence_scale;
    constexpr size_t short_independent_work =
        chunk_work * short_work_confidence / confidence_scale;
    const auto estimate_work = [line_count](const size_t work) {
        return work > std::numeric_limits<size_t>::max() / line_count
                   ? std::numeric_limits<size_t>::max()
                   : work * line_count / sample_count;
    };
    const size_t estimated_work = estimate_work(sampled_work);
    const size_t estimated_independent_work =
        estimate_work(sampled_work - max_work);
    return (estimated_work < short_parallel_work && max_work < chunk_work) ||
           estimated_independent_work < short_independent_work;
}

template<af_op_t op, typename T>
void reduceRaggedLines(Param<T> output, Param<uint> loc, CParam<T> input,
                       const int dim, CParam<uint> rlen, const size_t begin,
                       const size_t end) {
    const dim4 odims    = output.dims();
    const dim4 ostrides = output.strides();
    const dim4 idims    = input.dims();
    const dim4 istrides = input.strides();
    const dim4 rstrides = rlen.strides();
    T *const out        = output.get();
    uint *const indices = loc.get();
    const T *const in   = input.get();
    const dim_t stride  = istrides[dim];
    std::array<dim_t, 4> coord = lineCoordinates(odims, begin);

    for (size_t line = begin; line < end; ++line) {
        const dim_t out_offset  = coordinateOffset(coord, ostrides);
        const dim_t in_offset   = coordinateOffset(coord, istrides);
        const dim_t rlen_offset = coordinateOffset(coord, rstrides);
        const int lim = static_cast<int>(
            raggedLimit(rlen, rlen_offset, idims[dim]));
        kernel::ireduce_line<op, T>(out, indices, out_offset, in, in_offset,
                                    stride, lim);
        advanceLineCoordinates(coord, odims);
    }
}

template<af_op_t op, typename T>
void reduceRaggedSerial(Param<T> output, Param<uint> loc, CParam<T> input,
                        const int dim, CParam<uint> rlen) {
    switch (input.dims().ndims()) {
        case 1:
            return kernel::ireduce_dim<op, T, 1>()(output, loc, 0, input, 0,
                                                   dim, rlen);
        case 2:
            return kernel::ireduce_dim<op, T, 2>()(output, loc, 0, input, 0,
                                                   dim, rlen);
        case 3:
            return kernel::ireduce_dim<op, T, 3>()(output, loc, 0, input, 0,
                                                   dim, rlen);
        default:
            return kernel::ireduce_dim<op, T, 4>()(output, loc, 0, input, 0,
                                                   dim, rlen);
    }
}

std::vector<RaggedRange> coalesceRaggedRanges(
    const std::vector<RaggedRange> &ranges, const size_t total_work,
    const size_t max_ranges) {
    if (ranges.size() <= max_ranges) { return ranges; }

    const size_t target_work =
        1 + (total_work - 1) / std::max<size_t>(1, max_ranges);
    std::vector<RaggedRange> merged;
    merged.reserve(max_ranges);
    RaggedRange current = ranges.front();
    for (size_t i = 1; i < ranges.size(); ++i) {
        const RaggedRange &next = ranges[i];
        if ((current.work >= target_work || next.work >= target_work) &&
            merged.size() + 1 < max_ranges) {
            merged.push_back(current);
            current = next;
        } else {
            current.end = next.end;
            current.work += next.work;
        }
    }
    merged.push_back(current);
    return merged;
}

template<af_op_t op, typename T>
void ireduceRaggedDimParallel(Param<T> output, Param<uint> loc,
                              CParam<T> input, const int dim,
                              CParam<uint> rlen,
                              const bool rlen_is_linear) {
    const dim4 odims    = output.dims();
    const dim4 idims    = input.dims();
    const dim4 rstrides = rlen.strides();
    const size_t line_count = static_cast<size_t>(odims.elements());

    constexpr size_t min_parallel_work = 1 << 17;
    constexpr size_t chunk_work        = 1 << 16;

    if (isClearlyShortRaggedReduction(rlen, odims, rstrides, idims[dim],
                                      rlen_is_linear)) {
        if (rlen_is_linear) {
            reduceRaggedSerial<op, T>(output, loc, input, dim, rlen);
        } else {
            reduceRaggedLines<op, T>(output, loc, input, dim, rlen, 0,
                                     line_count);
        }
        return;
    }

    size_t total_work    = 0;
    size_t max_line_work = 0;
    if (rlen_is_linear) {
        for (size_t line = 0; line < line_count; ++line) {
            const size_t work = raggedLineWork(
                raggedLimit(rlen, static_cast<dim_t>(line), idims[dim]));
            total_work += work;
            max_line_work = std::max(max_line_work, work);
        }
    } else {
        std::array<dim_t, 4> coord = lineCoordinates(odims, 0);
        for (size_t line = 0; line < line_count; ++line) {
            const dim_t rlen_offset = coordinateOffset(coord, rstrides);
            const size_t work = raggedLineWork(
                raggedLimit(rlen, rlen_offset, idims[dim]));
            total_work += work;
            max_line_work = std::max(max_line_work, work);
            advanceLineCoordinates(coord, odims);
        }
    }

    if (total_work < min_parallel_work ||
        total_work - max_line_work <= chunk_work) {
        if (rlen_is_linear) {
            reduceRaggedSerial<op, T>(output, loc, input, dim, rlen);
        } else {
            reduceRaggedLines<op, T>(output, loc, input, dim, rlen, 0,
                                     line_count);
        }
        return;
    }

    size_t range_begin = 0;
    size_t range_work  = 0;
    std::vector<RaggedRange> ranges;
    std::array<dim_t, 4> coord = lineCoordinates(odims, 0);
    for (size_t line = 0; line < line_count; ++line) {
        const dim_t rlen_offset =
            rlen_is_linear ? static_cast<dim_t>(line)
                           : coordinateOffset(coord, rstrides);
        const size_t line_work = raggedLineWork(
            raggedLimit(rlen, rlen_offset, idims[dim]));
        advanceLineCoordinates(coord, odims);

        if (range_work > 0 && line_work >= chunk_work) {
            ranges.push_back(RaggedRange{range_begin, line, range_work});
            range_begin = line;
            range_work  = 0;
        }
        range_work += line_work;
        if (range_work >= chunk_work) {
            ranges.push_back(RaggedRange{range_begin, line + 1, range_work});
            range_begin = line + 1;
            range_work  = 0;
        }
    }
    if (range_begin < line_count) {
        ranges.push_back(
            RaggedRange{range_begin, line_count, range_work});
    }
    if (ranges.size() < 2) {
        if (rlen_is_linear) {
            reduceRaggedSerial<op, T>(output, loc, input, dim, rlen);
        } else {
            reduceRaggedLines<op, T>(output, loc, input, dim, rlen, 0,
                                     line_count);
        }
        return;
    }

    constexpr size_t max_ragged_tasks = 1024;
    const size_t useful_tasks =
        std::min(max_ragged_tasks, 4 * getParallelThreadCount());
    ranges = coalesceRaggedRanges(ranges, total_work, useful_tasks);
    parallelFor(ranges.size(), [&](const size_t task) {
        const RaggedRange &range = ranges[task];
        reduceRaggedLines<op, T>(output, loc, input, dim, rlen, range.begin,
                                 range.end);
    });
}

template<typename T>
struct IndexedPartial {
    T value{};
    double key{0};
    uint index{0};
    bool valid{false};
};

template<af_op_t op, typename T>
void updatePartial(IndexedPartial<T> &partial, const T value,
                   const uint index) {
    const double key = kernel::cabs(value);
    if (std::isnan(key)) { return; }

    if (!partial.valid) {
        partial.value = value;
        partial.key   = key;
        partial.index = index;
        partial.valid = true;
        return;
    }

    kernel::MinMaxOp<op, T>::update(partial.value, partial.key, partial.index,
                                    value, key, index);
}

template<af_op_t op, typename T>
IndexedPartial<T> ireduceRange(CParam<T> in, const size_t begin,
                               const size_t end, const bool is_linear) {
    const af::dim4 dims       = in.dims();
    const af::dim4 strides    = in.strides();
    const T *const input      = in.get();
    IndexedPartial<T> partial = {};

    if (is_linear) {
        for (size_t item = begin; item < end; ++item) {
            updatePartial<op>(partial, input[item], static_cast<uint>(item));
        }
        return partial;
    }

    size_t linear = begin;
    std::array<dim_t, 4> coord;
    for (int dim = 0; dim < 4; ++dim) {
        coord[dim] =
            static_cast<dim_t>(linear % static_cast<size_t>(dims[dim]));
        linear /= static_cast<size_t>(dims[dim]);
    }

    size_t item = begin;
    while (item < end) {
        // Keep the existing ireduce_all assumption that stride 0 is one.
        const dim_t offset = coord[0] + coord[1] * strides[1] +
                             coord[2] * strides[2] +
                             coord[3] * strides[3];
        const size_t run =
            std::min(end - item, static_cast<size_t>(dims[0] - coord[0]));
        for (size_t x = 0; x < run; ++x) {
            updatePartial<op>(partial, input[offset + static_cast<dim_t>(x)],
                              static_cast<uint>(item + x));
        }
        item += run;

        coord[0] += static_cast<dim_t>(run);
        if (coord[0] < dims[0]) { continue; }
        coord[0] = 0;
        for (int dim = 1; dim < 4; ++dim) {
            if (++coord[dim] < dims[dim]) { break; }
            coord[dim] = 0;
        }
    }
    return partial;
}

}  // namespace

template<af_op_t op, typename T>
using ireduce_dim_func =
    std::function<void(Param<T>, Param<uint>, const dim_t, CParam<T>,
                       const dim_t, const int, CParam<uint>)>;

template<af_op_t op, typename T>
void ireduce(Array<T> &out, Array<uint> &loc, const Array<T> &in,
             const int dim) {
    dim4 odims       = in.dims();
    odims[dim]       = 1;
    Array<uint> rlen = createEmptyArray<uint>(af::dim4(0));
    static const ireduce_dim_func<op, T> ireduce_funcs[] = {
        kernel::ireduce_dim<op, T, 1>(), kernel::ireduce_dim<op, T, 2>(),
        kernel::ireduce_dim<op, T, 3>(), kernel::ireduce_dim<op, T, 4>()};

    if (shouldParallelizeDimensionalIReduce(in.dims(), odims)) {
        getQueue().enqueue(kernel::ireduce_dim_parallel<op, T>, out, loc, in,
                           dim, rlen);
    } else {
        getQueue().enqueue(ireduce_funcs[in.ndims() - 1], out, loc, 0, in, 0,
                           dim, rlen);
    }
}

template<af_op_t op, typename T>
void rreduce(Array<T> &out, Array<uint> &loc, const Array<T> &in, const int dim,
             const Array<uint> &rlen) {
    dim4 odims = in.dims();
    odims[dim] = 1;

    static const ireduce_dim_func<op, T> ireduce_funcs[] = {
        kernel::ireduce_dim<op, T, 1>(), kernel::ireduce_dim<op, T, 2>(),
        kernel::ireduce_dim<op, T, 3>(), kernel::ireduce_dim<op, T, 4>()};

    const bool should_parallelize =
        shouldParallelizeDimensionalIReduce(in.dims(), odims);
    const bool rlen_is_linear = rlen.isLinear();
    if (should_parallelize) {
        getQueue().enqueue(ireduceRaggedDimParallel<op, T>, out, loc, in, dim,
                           rlen, rlen_is_linear);
    } else if (!rlen_is_linear) {
        getQueue().enqueue(reduceRaggedLines<op, T>, out, loc, in, dim, rlen,
                           0, static_cast<size_t>(odims.elements()));
    } else {
        getQueue().enqueue(ireduce_funcs[in.ndims() - 1], out, loc, 0, in, 0,
                           dim, rlen);
    }
}

template<af_op_t op, typename T>
T ireduce_all(unsigned *loc, const Array<T> &in) {
    in.eval();
    getQueue().sync();

    const af::dim4 dims       = in.dims();
    const af::dim4 strides    = in.strides();
    const CParam<T> input     = in;
    const T *const inPtr      = in.get();

    constexpr size_t block_elements        = 1 << 16;
    constexpr size_t min_parallel_elements = 1 << 17;
    constexpr size_t max_blocks            = 1024;
    const size_t elements = static_cast<size_t>(dims.elements());

    if (elements >= min_parallel_elements &&
        elements - 1 <= std::numeric_limits<uint>::max() &&
        getParallelThreadCount() > 1) {
        const size_t requested_blocks =
            1 + (elements - 1) / block_elements;
        const size_t block_count = std::min(max_blocks, requested_blocks);
        std::vector<IndexedPartial<T>> partials(block_count);
        const size_t elements_per_block = elements / block_count;
        const size_t remainder          = elements % block_count;

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
                for (size_t block = block_begin; block < block_end; ++block) {
                    const size_t begin = block * elements_per_block +
                                         std::min(block, remainder);
                    const size_t end = begin + elements_per_block +
                                       (block < remainder ? 1 : 0);
                    partials[block] =
                        ireduceRange<op, T>(input, begin, end, is_linear);
                }
            });

        // Preserve the serial path's special initialization from logical
        // element zero. In particular, an initial NaN retains index zero and
        // an operation identity until a comparable candidate can replace it.
        kernel::MinMaxOp<op, T> Op(inPtr[0], 0);
        for (const IndexedPartial<T> &partial : partials) {
            if (partial.valid) {
                Op.consider(partial.value, partial.key, partial.index);
            }
        }

        *loc = Op.m_idx;
        return Op.m_val;
    }

    kernel::MinMaxOp<op, T> Op(inPtr[0], 0);
    dim_t idx = 0;

    for (dim_t l = 0; l < dims[3]; l++) {
        dim_t off3 = l * strides[3];

        for (dim_t k = 0; k < dims[2]; k++) {
            dim_t off2 = k * strides[2];

            for (dim_t j = 0; j < dims[1]; j++) {
                dim_t off1 = j * strides[1];

                for (dim_t i = 0; i < dims[0]; i++) {
                    dim_t d_idx = i + off1 + off2 + off3;
                    Op(inPtr[d_idx], idx++);
                }
            }
        }
    }

    *loc = Op.m_idx;
    return Op.m_val;
}

#define INSTANTIATE(ROp, T)                                           \
    template void ireduce<ROp, T>(Array<T> & out, Array<uint> & loc,  \
                                  const Array<T> &in, const int dim); \
    template void rreduce<ROp, T>(Array<T> & out, Array<uint> & loc,  \
                                  const Array<T> &in, const int dim,  \
                                  const Array<uint> &rlen);           \
    template T ireduce_all<ROp, T>(unsigned *loc, const Array<T> &in);

// min
INSTANTIATE(af_min_t, float)
INSTANTIATE(af_min_t, double)
INSTANTIATE(af_min_t, cfloat)
INSTANTIATE(af_min_t, cdouble)
INSTANTIATE(af_min_t, int)
INSTANTIATE(af_min_t, uint)
INSTANTIATE(af_min_t, intl)
INSTANTIATE(af_min_t, uintl)
INSTANTIATE(af_min_t, char)
INSTANTIATE(af_min_t, schar)
INSTANTIATE(af_min_t, uchar)
INSTANTIATE(af_min_t, short)
INSTANTIATE(af_min_t, ushort)
INSTANTIATE(af_min_t, half)

// max
INSTANTIATE(af_max_t, float)
INSTANTIATE(af_max_t, double)
INSTANTIATE(af_max_t, cfloat)
INSTANTIATE(af_max_t, cdouble)
INSTANTIATE(af_max_t, int)
INSTANTIATE(af_max_t, uint)
INSTANTIATE(af_max_t, intl)
INSTANTIATE(af_max_t, uintl)
INSTANTIATE(af_max_t, char)
INSTANTIATE(af_max_t, schar)
INSTANTIATE(af_max_t, uchar)
INSTANTIATE(af_max_t, short)
INSTANTIATE(af_max_t, ushort)
INSTANTIATE(af_max_t, half)

}  // namespace cpu
}  // namespace arrayfire
