/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/dim4.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <utility>

namespace arrayfire {
namespace cpu {

/// Returns the maximum number of threads used by parallel CPU kernels.
size_t getParallelThreadCount();

/// Executes each task exactly once and waits for all tasks to complete.
///
/// The calling thread executes one task. Remaining tasks are scheduled on a
/// process-wide worker pool so repeated JIT evaluations do not create threads.
void parallelFor(size_t task_count, std::function<void(size_t)> function);

/// Splits [0, item_count) into contiguous ranges and evaluates each range once.
///
/// Ranges contain at least min_items_per_task unless there is only one range.
/// The calling thread participates through parallelFor.
void parallelForRange(size_t item_count, size_t min_items_per_task,
                      std::function<void(size_t begin, size_t end)> function);

/// Visits every coordinate in dims exactly once using contiguous linear ranges.
template<typename Function>
void parallelForEach(const af::dim4 &dims, size_t min_items_per_task,
                     Function function) {
    const size_t item_count = static_cast<size_t>(dims.elements());
    parallelForRange(
        item_count, min_items_per_task,
        [dims, function = std::move(function)](size_t begin, size_t end) {
            size_t linear = begin;
            std::array<dim_t, 4> coord;
            for (int dim = 0; dim < 4; ++dim) {
                coord[dim] =
                    static_cast<dim_t>(linear % static_cast<size_t>(dims[dim]));
                linear /= static_cast<size_t>(dims[dim]);
            }

            for (size_t item = begin; item < end; ++item) {
                function(coord[0], coord[1], coord[2], coord[3]);
                for (int dim = 0; dim < 4; ++dim) {
                    if (++coord[dim] < dims[dim]) { break; }
                    coord[dim] = 0;
                }
            }
        });
}

}  // namespace cpu
}  // namespace arrayfire
