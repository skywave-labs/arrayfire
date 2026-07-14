/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cstddef>
#include <functional>

namespace arrayfire {
namespace cpu {

/// Returns the maximum number of threads used by parallel CPU kernels.
size_t getParallelThreadCount();

/// Executes each task exactly once and waits for all tasks to complete.
///
/// The calling thread executes one task. Remaining tasks are scheduled on a
/// process-wide worker pool so repeated JIT evaluations do not create threads.
void parallelFor(size_t task_count, std::function<void(size_t)> function);

}  // namespace cpu
}  // namespace arrayfire
