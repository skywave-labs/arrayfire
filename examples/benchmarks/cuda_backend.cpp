/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <af/cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;
using Work  = std::function<af::array()>;

struct Options {
    int device;
    int iterations;
    dim_t size;
    std::string selectedCase;

    Options() : device(0), iterations(20), size(2048), selectedCase("all") {}
};

struct Metrics {
    double firstCallMs;
    double deviceMs;
    double wallMs;
    double enqueueUs;
    size_t poolBytes;
    size_t poolBuffers;
    size_t lockedBytes;
    size_t lockedBuffers;
};

double elapsedMs(const Clock::time_point &begin, const Clock::time_point &end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void checkCuda(cudaError_t error, const char *operation) {
    if (error == cudaSuccess) { return; }

    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}

bool selected(const Options &options, const char *name) {
    return options.selectedCase == "all" || options.selectedCase == name;
}

Options volumeOptions(const Options &options) {
    Options volume = options;
    volume.size = std::max<dim_t>(16, std::min<dim_t>(96, options.size / 32));
    return volume;
}

Options neuralOptions(const Options &options) {
    Options neural = options;
    neural.size = std::max<dim_t>(32, std::min<dim_t>(512, options.size / 4));
    return neural;
}

void materialize(const af::array &value) {
    value.eval();
    af::sync();
}

Metrics measure(const Work &work, int iterations, cudaStream_t stream) {
    af::sync();

    size_t initialPoolBytes   = 0;
    size_t initialPoolBuffers = 0;
    size_t initialLockBytes   = 0;
    size_t initialLockBuffers = 0;
    af::deviceMemInfo(&initialPoolBytes, &initialPoolBuffers, &initialLockBytes,
                      &initialLockBuffers);

    const Clock::time_point firstBegin = Clock::now();
    {
        af::array first = work();
        first.eval();
        af::sync();
    }
    const double firstCallMs = elapsedMs(firstBegin, Clock::now());

    for (int i = 0; i < 3; ++i) {
        af::array warm = work();
        warm.eval();
    }
    af::sync();

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent  = nullptr;
    checkCuda(cudaEventCreate(&startEvent), "cudaEventCreate(start)");
    try {
        checkCuda(cudaEventCreate(&stopEvent), "cudaEventCreate(stop)");
        checkCuda(cudaEventRecord(startEvent, stream),
                  "cudaEventRecord(start)");

        const Clock::time_point wallBegin = Clock::now();
        af::array result;
        for (int i = 0; i < iterations; ++i) {
            result = work();
            result.eval();
        }
        checkCuda(cudaEventRecord(stopEvent, stream), "cudaEventRecord(stop)");
        checkCuda(cudaEventSynchronize(stopEvent),
                  "cudaEventSynchronize(stop)");
        const double wallMs = elapsedMs(wallBegin, Clock::now()) / iterations;

        float deviceBatchMs = 0.0f;
        checkCuda(cudaEventElapsedTime(&deviceBatchMs, startEvent, stopEvent),
                  "cudaEventElapsedTime");

        result = af::array();
        af::sync();

        const Clock::time_point enqueueBegin = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            result = work();
            result.eval();
        }
        const double enqueueUs =
            elapsedMs(enqueueBegin, Clock::now()) * 1000.0 / iterations;
        af::sync();
        result = af::array();

        size_t poolBytes   = 0;
        size_t poolBuffers = 0;
        size_t lockBytes   = 0;
        size_t lockBuffers = 0;
        af::deviceMemInfo(&poolBytes, &poolBuffers, &lockBytes, &lockBuffers);

        checkCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy(stop)");
        checkCuda(cudaEventDestroy(startEvent), "cudaEventDestroy(start)");

        Metrics metrics = {
            firstCallMs,
            static_cast<double>(deviceBatchMs) / iterations,
            wallMs,
            enqueueUs,
            poolBytes - std::min(poolBytes, initialPoolBytes),
            poolBuffers - std::min(poolBuffers, initialPoolBuffers),
            lockBytes - std::min(lockBytes, initialLockBytes),
            lockBuffers - std::min(lockBuffers, initialLockBuffers)};
        return metrics;
    } catch (...) {
        if (stopEvent) { cudaEventDestroy(stopEvent); }
        cudaEventDestroy(startEvent);
        throw;
    }
}

void printMetrics(const char *name, const Options &options,
                  const Metrics &metrics) {
    std::cout << name << ',' << options.size << ',' << options.iterations << ','
              << std::fixed << std::setprecision(6) << metrics.firstCallMs
              << ',' << metrics.deviceMs << ',' << metrics.wallMs << ','
              << metrics.enqueueUs << ',' << metrics.poolBytes << ','
              << metrics.poolBuffers << ',' << metrics.lockedBytes << ','
              << metrics.lockedBuffers << '\n';
}

void runJitContiguous(const Options &options, cudaStream_t stream) {
    if (!selected(options, "jit_contiguous")) { return; }

    af::deviceGC();
    af::array lhs = af::randu(options.size, options.size, f32);
    af::array rhs = af::randu(options.size, options.size, f32);
    materialize(lhs);
    materialize(rhs);

    Work work = [lhs, rhs]() {
        return af::sin(lhs) * af::cos(rhs) +
               af::sqrt(af::abs(lhs - rhs) + 1.0f);
    };
    printMetrics("jit_contiguous", options,
                 measure(work, options.iterations, stream));
}

void runJitGapped(const Options &options, cudaStream_t stream) {
    if (!selected(options, "jit_gapped")) { return; }

    af::deviceGC();
    af::array lhsBase = af::randu(options.size * 2, options.size, f32);
    af::array rhsBase = af::randu(options.size * 2, options.size, f32);
    materialize(lhsBase);
    materialize(rhsBase);
    af::array lhs = lhsBase(af::seq(0, options.size * 2 - 2, 2), af::span);
    af::array rhs = rhsBase(af::seq(0, options.size * 2 - 2, 2), af::span);

    Work work = [lhs, rhs]() {
        return af::sin(lhs) * af::cos(rhs) +
               af::sqrt(af::abs(lhs - rhs) + 1.0f);
    };
    printMetrics("jit_gapped", options,
                 measure(work, options.iterations, stream));
}

void runReduction(const Options &options, cudaStream_t stream) {
    if (!selected(options, "reduce_dim0")) { return; }

    af::deviceGC();
    af::array input = af::randu(options.size, options.size, f32);
    materialize(input);

    Work work = [input]() { return af::sum(input, 0); };
    printMetrics("reduce_dim0", options,
                 measure(work, options.iterations, stream));
}

af::array reductionProducer(const af::array &lhs, const af::array &rhs) {
    return af::sin(lhs) * af::cos(rhs) + af::sqrt(af::abs(lhs - rhs) + 1.0f);
}

af::array f64ReductionProducer(const af::array &lhs, const af::array &rhs) {
    return af::sin(lhs) * af::cos(rhs) + af::sqrt(af::abs(lhs - rhs) + 1.0);
}

af::array f64DeepReductionProducer(const af::array &lhs,
                                   const af::array &rhs) {
    af::array value = lhs * 0.75 + rhs * 0.25 + 0.125;
    value           = af::sin(value) + af::cos(lhs - rhs);
    value = value * value + af::sqrt(af::abs(lhs + rhs) + 1.0);
    return af::tanh(value) + value * 0.5;
}

enum class F64Producer { Affine, Complex, Deep };

void runJitReductions(const Options &options, cudaStream_t stream) {
    const bool runDim0 = selected(options, "reduce_jit_dim0") ||
                         selected(options, "reduce_jit_dim0_materialized");
    const bool runAll = selected(options, "reduce_jit_all") ||
                        selected(options, "reduce_jit_all_materialized");
    const bool runReuse = selected(options, "reduce_jit_reuse") ||
                          selected(options, "reduce_jit_reuse_materialized");
    if (!runDim0 && !runAll && !runReuse) { return; }

    Options reduction = options;
    reduction.size    = std::max<dim_t>(256, options.size);
    af::deviceGC();
    af::array lhs = af::randu(reduction.size, reduction.size, f32);
    af::array rhs = af::randu(reduction.size, reduction.size, f32);
    materialize(lhs);
    materialize(rhs);

    if (selected(options, "reduce_jit_dim0")) {
        af::deviceGC();
        Work work = [lhs, rhs]() {
            return af::sum(reductionProducer(lhs, rhs), 0);
        };
        printMetrics("reduce_jit_dim0", reduction,
                     measure(work, reduction.iterations, stream));
    }

    if (selected(options, "reduce_jit_dim0_materialized")) {
        af::deviceGC();
        Work work = [lhs, rhs]() {
            af::array expression = reductionProducer(lhs, rhs);
            expression.eval();
            return af::sum(expression, 0);
        };
        printMetrics("reduce_jit_dim0_materialized", reduction,
                     measure(work, reduction.iterations, stream));
    }

    if (selected(options, "reduce_jit_all")) {
        af::deviceGC();
        Work work = [lhs, rhs]() {
            return af::sum<af::array>(reductionProducer(lhs, rhs));
        };
        printMetrics("reduce_jit_all", reduction,
                     measure(work, reduction.iterations, stream));
    }

    if (selected(options, "reduce_jit_all_materialized")) {
        af::deviceGC();
        Work work = [lhs, rhs]() {
            af::array expression = reductionProducer(lhs, rhs);
            expression.eval();
            return af::sum<af::array>(expression);
        };
        printMetrics("reduce_jit_all_materialized", reduction,
                     measure(work, reduction.iterations, stream));
    }

    if (selected(options, "reduce_jit_reuse")) {
        af::deviceGC();
        Work work = [lhs, rhs]() {
            af::array expression = reductionProducer(lhs, rhs);
            return af::sum(expression, 0) + af::sum(expression, 0);
        };
        printMetrics("reduce_jit_reuse", reduction,
                     measure(work, reduction.iterations, stream));
    }

    if (selected(options, "reduce_jit_reuse_materialized")) {
        af::deviceGC();
        Work work = [lhs, rhs]() {
            af::array expression = reductionProducer(lhs, rhs);
            expression.eval();
            return af::sum(expression, 0) + af::sum(expression, 0);
        };
        printMetrics("reduce_jit_reuse_materialized", reduction,
                     measure(work, reduction.iterations, stream));
    }
}

void runJitReductionRegressions(const Options &options, cudaStream_t stream) {
    auto runF64 = [&](const char *name, dim_t size, bool preEvaluate,
                      F64Producer producer) {
        if (!selected(options, name)) { return; }

        Options reduction = options;
        reduction.size    = size;
        af::deviceGC();
        af::array lhs = af::randu(size, size, f64);
        af::array rhs = af::randu(size, size, f64);
        materialize(lhs);
        materialize(rhs);

        Work work = [lhs, rhs, preEvaluate, producer]() {
            af::array expression;
            if (producer == F64Producer::Affine) {
                expression = lhs * 1.25 + rhs * 0.5 + 0.25;
            } else if (producer == F64Producer::Complex) {
                expression = f64ReductionProducer(lhs, rhs);
            } else {
                expression = f64DeepReductionProducer(lhs, rhs);
            }
            if (preEvaluate) { expression.eval(); }
            return af::sum<af::array>(expression);
        };
        printMetrics(name, reduction,
                     measure(work, reduction.iterations, stream));
    };

    runF64("reduce_jit_all_f64_affine_256", 256, false,
           F64Producer::Affine);
    runF64("reduce_jit_all_f64_affine_256_materialized", 256, true,
           F64Producer::Affine);
    runF64("reduce_jit_all_f64_complex_256", 256, false,
           F64Producer::Complex);
    runF64("reduce_jit_all_f64_complex_256_materialized", 256, true,
           F64Producer::Complex);
    runF64("reduce_jit_all_f64_deep_1024", 1024, false, F64Producer::Deep);
    runF64("reduce_jit_all_f64_deep_1024_materialized", 1024, true,
           F64Producer::Deep);

    const bool runReuse = selected(options, "reduce_jit_all_reuse4") ||
                          selected(options,
                                   "reduce_jit_all_reuse4_materialized");
    if (runReuse) {
        Options reduction = options;
        reduction.size    = 1024;
        af::deviceGC();
        af::array lhs = af::randu(reduction.size, reduction.size, f32);
        af::array rhs = af::randu(reduction.size, reduction.size, f32);
        materialize(lhs);
        materialize(rhs);

        auto run = [&](const char *name, bool materialized) {
            if (!selected(options, name)) { return; }
            Work work = [lhs, rhs, materialized]() {
                af::array expression = reductionProducer(lhs, rhs);
                if (materialized) { expression.eval(); }
                af::array result = af::sum<af::array>(expression);
                for (int consumer = 1; consumer < 4; ++consumer) {
                    result += af::sum<af::array>(expression);
                }
                return result;
            };
            printMetrics(name, reduction,
                         measure(work, reduction.iterations, stream));
        };
        run("reduce_jit_all_reuse4", false);
        run("reduce_jit_all_reuse4_materialized", true);
    }

    const bool runLongLine =
        selected(options, "reduce_jit_dim0_complex_32768x2") ||
        selected(options, "reduce_jit_dim0_complex_32768x2_materialized");
    if (runLongLine) {
        Options reduction = options;
        reduction.size    = 32768;
        af::deviceGC();
        af::array lhs = af::randu(32768, 2, f32);
        af::array rhs = af::randu(32768, 2, f32);
        materialize(lhs);
        materialize(rhs);

        auto run = [&](const char *name, bool materialized) {
            if (!selected(options, name)) { return; }
            Work work = [lhs, rhs, materialized]() {
                af::array expression = reductionProducer(lhs, rhs);
                if (materialized) { expression.eval(); }
                return af::sum(expression, 0);
            };
            printMetrics(name, reduction,
                         measure(work, reduction.iterations, stream));
        };
        run("reduce_jit_dim0_complex_32768x2", false);
        run("reduce_jit_dim0_complex_32768x2_materialized", true);
    }
}

void runShortJitReduction(const Options &options, cudaStream_t stream) {
    if (!selected(options, "reduce_jit_short_dim0")) { return; }

    af::deviceGC();
    constexpr dim_t lineLength = 32;
    const dim_t lines          = std::max<dim_t>(2048, options.size * 32);
    af::array lhs              = af::randu(lineLength, lines, f32);
    af::array rhs              = af::randu(lineLength, lines, f32);
    materialize(lhs);
    materialize(rhs);

    Work work = [lhs, rhs]() {
        return af::sum(reductionProducer(lhs, rhs), 0);
    };
    printMetrics("reduce_jit_short_dim0", options,
                 measure(work, options.iterations, stream));
}

void runGappedJitReduction(const Options &options, cudaStream_t stream) {
    if (!selected(options, "reduce_jit_gapped")) { return; }

    Options reduction = options;
    reduction.size    = std::max<dim_t>(256, options.size);
    af::deviceGC();
    af::array lhsBase = af::randu(reduction.size * 2, reduction.size, f32);
    af::array rhsBase = af::randu(reduction.size * 2, reduction.size, f32);
    materialize(lhsBase);
    materialize(rhsBase);
    af::array lhs = lhsBase(af::seq(0, reduction.size * 2 - 2, 2), af::span);
    af::array rhs = rhsBase(af::seq(0, reduction.size * 2 - 2, 2), af::span);

    Work work = [lhs, rhs]() {
        return af::sum(reductionProducer(lhs, rhs), 0);
    };
    printMetrics("reduce_jit_gapped", reduction,
                 measure(work, reduction.iterations, stream));
}

void runMatmul(const Options &options, cudaStream_t stream) {
    if (!selected(options, "matmul")) { return; }

    af::deviceGC();
    af::array lhs = af::randu(options.size, options.size, f32);
    af::array rhs = af::randu(options.size, options.size, f32);
    materialize(lhs);
    materialize(rhs);

    Work work = [lhs, rhs]() { return af::matmul(lhs, rhs); };
    printMetrics("matmul", options, measure(work, options.iterations, stream));
}

void runBatchedMatmul(const Options &options, cudaStream_t stream) {
    if (!selected(options, "matmul_batched_8")) { return; }

    af::deviceGC();
    const dim_t batches = std::max<dim_t>(2, options.size);
    af::array lhs       = af::randu(8, 8, batches, f32);
    af::array rhs       = af::randu(8, 8, batches, f32);
    materialize(lhs);
    materialize(rhs);

    Work work = [lhs, rhs]() { return af::matmul(lhs, rhs); };
    printMetrics("matmul_batched_8", options,
                 measure(work, options.iterations, stream));
}

void runBroadcastBatchedMatmul(const Options &options, cudaStream_t stream) {
    if (!selected(options, "matmul_batched_d2_broadcast_8")) { return; }

    af::deviceGC();
    constexpr dim_t batchDim2 = 8;
    const dim_t batchDim3 =
        std::max<dim_t>(2, (options.size + batchDim2 - 1) / batchDim2);
    af::array lhs = af::randu(8, 8, 1, batchDim3, f32);
    af::array rhs = af::randu(8, 8, batchDim2, batchDim3, f32);
    materialize(lhs);
    materialize(rhs);

    Work work = [lhs, rhs]() { return af::matmul(lhs, rhs); };
    printMetrics("matmul_batched_d2_broadcast_8", options,
                 measure(work, options.iterations, stream));
}

void runComputeBoundBatchedMatmul(const Options &options, cudaStream_t stream) {
    if (!selected(options, "matmul_batched_256")) { return; }

    af::deviceGC();
    const dim_t batches =
        std::max<dim_t>(2, std::min<dim_t>(16, options.size / 256));
    af::array lhs = af::randu(256, 256, batches, f32);
    af::array rhs = af::randu(256, 256, batches, f32);
    materialize(lhs);
    materialize(rhs);

    Work work = [lhs, rhs]() { return af::matmul(lhs, rhs); };
    printMetrics("matmul_batched_256", options,
                 measure(work, options.iterations, stream));
}

void runSort(const Options &options, cudaStream_t stream) {
    if (!selected(options, "sort_batched")) { return; }

    af::deviceGC();
    const dim_t batches = std::max<dim_t>(1, options.size / 4);
    af::array input     = af::randu(options.size, batches, f32);
    materialize(input);

    Work work = [input]() { return af::sort(input, 0, true); };
    printMetrics("sort_batched", options,
                 measure(work, options.iterations, stream));
}

void runShortSegmentSorts(const Options &options, cudaStream_t stream) {
    constexpr dim_t lineLength = 32;
    const dim_t batches        = std::max<dim_t>(32, options.size * 2);

    if (selected(options, "sort_values_32")) {
        af::deviceGC();
        af::array input = af::randu(lineLength, batches, f32);
        materialize(input);

        Work work = [input]() { return af::sort(input, 0, true); };
        printMetrics("sort_values_32", options,
                     measure(work, options.iterations, stream));
    }

    if (selected(options, "sort_values_32_dim1")) {
        af::deviceGC();
        af::array input = af::randu(batches, lineLength, f32);
        materialize(input);

        Work work = [input]() { return af::sort(input, 1, true); };
        printMetrics("sort_values_32_dim1", options,
                     measure(work, options.iterations, stream));
    }

    if (selected(options, "sort_index_32")) {
        af::deviceGC();
        af::array input = af::randu(lineLength, batches, f32);
        materialize(input);

        Work work = [input]() {
            af::array keys;
            af::array indices;
            af::sort(keys, indices, input, 0, true);
            return keys;
        };
        printMetrics("sort_index_32", options,
                     measure(work, options.iterations, stream));
    }

    if (selected(options, "sort_by_key_32")) {
        af::deviceGC();
        af::array keys   = af::randu(lineLength, batches, f32);
        af::array values = af::randu(lineLength, batches, c64);
        materialize(keys);
        materialize(values);

        Work work = [keys, values]() {
            af::array sortedKeys;
            af::array sortedValues;
            af::sort(sortedKeys, sortedValues, keys, values, 0, true);
            return sortedKeys;
        };
        printMetrics("sort_by_key_32", options,
                     measure(work, options.iterations, stream));
    }
}

void runMediumSegmentSorts(const Options &options, cudaStream_t stream) {
    constexpr dim_t lineLength = 256;
    const dim_t batches          = std::max<dim_t>(32, options.size / 2);
    const dim_t segmentedBatches =
        std::max<dim_t>(32, std::min<dim_t>(128, options.size / 2));

    if (selected(options, "sort_values_256_segmented")) {
        af::deviceGC();
        af::array input = af::randu(lineLength, segmentedBatches, f32);
        materialize(input);

        Work work = [input]() { return af::sort(input, 0, true); };
        printMetrics("sort_values_256_segmented", options,
                     measure(work, options.iterations, stream));
    }

    if (selected(options, "sort_index_256_segmented")) {
        af::deviceGC();
        af::array input = af::randu(lineLength, segmentedBatches, f32);
        materialize(input);

        Work work = [input]() {
            af::array keys;
            af::array indices;
            af::sort(keys, indices, input, 0, true);
            return keys;
        };
        printMetrics("sort_index_256_segmented", options,
                     measure(work, options.iterations, stream));
    }

    if (selected(options, "sort_by_key_256_segmented")) {
        af::deviceGC();
        af::array keys   = af::randu(lineLength, segmentedBatches, f32);
        af::array values = af::randu(lineLength, segmentedBatches, c64);
        materialize(keys);
        materialize(values);

        Work work = [keys, values]() {
            af::array sortedKeys;
            af::array sortedValues;
            af::sort(sortedKeys, sortedValues, keys, values, 0, true);
            return sortedKeys;
        };
        printMetrics("sort_by_key_256_segmented", options,
                     measure(work, options.iterations, stream));
    }

    if (selected(options, "sort_index_256")) {
        af::deviceGC();
        af::array input = af::randu(lineLength, batches, f32);
        materialize(input);

        Work work = [input]() {
            af::array keys;
            af::array indices;
            af::sort(keys, indices, input, 0, true);
            return keys;
        };
        printMetrics("sort_index_256", options,
                     measure(work, options.iterations, stream));
    }

    if (selected(options, "sort_by_key_256")) {
        af::deviceGC();
        af::array keys   = af::randu(lineLength, batches, f32);
        af::array values = af::randu(lineLength, batches, c64);
        materialize(keys);
        materialize(values);

        Work work = [keys, values]() {
            af::array sortedKeys;
            af::array sortedValues;
            af::sort(sortedKeys, sortedValues, keys, values, 0, true);
            return sortedKeys;
        };
        printMetrics("sort_by_key_256", options,
                     measure(work, options.iterations, stream));
    }

    if (selected(options, "sort_index_256_gapped_1024")) {
        af::deviceGC();
        constexpr dim_t regressionBatches = 1024;
        af::array base  = af::randu(regressionBatches * 2, lineLength, f32);
        af::array input =
            base(af::seq(0, regressionBatches * 2 - 2, 2), af::span);
        materialize(input);

        Work work = [input]() {
            af::array keys;
            af::array indices;
            af::sort(keys, indices, input, 1, false);
            return keys;
        };
        printMetrics("sort_index_256_gapped_1024", options,
                     measure(work, options.iterations, stream));
    }
}

void runLongSegmentSort(const Options &options, cudaStream_t stream) {
    if (!selected(options, "sort_values_100000")) { return; }

    af::deviceGC();
    constexpr dim_t lineLength = 100000;
    constexpr dim_t batches    = 11;
    af::array input             = af::randu(lineLength, batches, f32);
    materialize(input);

    Work work = [input]() { return af::sort(input, 0, true); };
    printMetrics("sort_values_100000", options,
                 measure(work, options.iterations, stream));
}

void runIterativeSort(const Options &options, cudaStream_t stream) {
    if (!selected(options, "sort_values_iterative_10")) { return; }

    af::deviceGC();
    constexpr dim_t batches = 10;
    const dim_t lineLength  = std::max<dim_t>(128, options.size);
    af::array input         = af::randu(lineLength, batches, f32);
    materialize(input);

    Work work = [input]() { return af::sort(input, 0, true); };
    printMetrics("sort_values_iterative_10", options,
                 measure(work, options.iterations, stream));
}

void runConvolution(const Options &options, cudaStream_t stream) {
    if (!selected(options, "convolve2_7x7")) { return; }

    af::deviceGC();
    af::array signal = af::randu(options.size, options.size, f32);
    af::array filter = af::constant(1.0f, 7, 7, f32) / 49.0f;
    materialize(signal);
    materialize(filter);

    Work work = [signal, filter]() {
        return af::convolve2(signal, filter, AF_CONV_DEFAULT, AF_CONV_SPATIAL);
    };
    printMetrics("convolve2_7x7", options,
                 measure(work, options.iterations, stream));
}

void runOneDimensionalConvolution(const Options &options, cudaStream_t stream) {
    if (!selected(options, "convolve1_9")) { return; }

    af::deviceGC();
    af::array signal = af::randu(options.size * options.size, f32);
    af::array filter = af::constant(1.0f, 9, f32) / 9.0f;
    materialize(signal);
    materialize(filter);

    Work work = [signal, filter]() {
        return af::convolve1(signal, filter, AF_CONV_DEFAULT, AF_CONV_SPATIAL);
    };
    printMetrics("convolve1_9", options,
                 measure(work, options.iterations, stream));
}

void runSeparableConvolution(const Options &options, cudaStream_t stream) {
    if (!selected(options, "convolve2_separable_7x7")) { return; }

    af::deviceGC();
    af::array signal = af::randu(options.size, options.size, f32);
    af::array column = af::constant(1.0f, 7, f32) / 7.0f;
    af::array row    = af::constant(1.0f, 7, f32) / 7.0f;
    materialize(signal);
    materialize(column);
    materialize(row);

    Work work = [signal, column, row]() {
        return af::convolve(column, row, signal, AF_CONV_DEFAULT);
    };
    printMetrics("convolve2_separable_7x7", options,
                 measure(work, options.iterations, stream));
}

void runCudnnForward(const Options &options, cudaStream_t stream) {
    if (!selected(options, "cudnn_forward_3x3")) { return; }

    const Options neural = neuralOptions(options);
    af::deviceGC();

    // Initialize the optional cuDNN plugin and handle with a different key so
    // first_call_ms below isolates selection for the measured descriptor.
    af::array primeSignal = af::randu(8, 8, 1, 1, f32);
    af::array primeFilter = af::randu(1, 1, 1, 1, f32);
    materialize(af::convolve2NN(primeSignal, primeFilter, af::dim4(1, 1),
                                af::dim4(0, 0, 1, 1), af::dim4(1, 1)));

    af::array signal = af::randu(neural.size, neural.size, 8, 2, f32);
    af::array filter = af::randu(3, 3, 8, 16, f32);
    materialize(signal);
    materialize(filter);

    Work work = [&signal, &filter]() {
        return af::convolve2NN(signal, filter, af::dim4(1, 1), af::dim4(1, 1),
                               af::dim4(1, 1));
    };
    printMetrics("cudnn_forward_3x3", neural,
                 measure(work, neural.iterations, stream));
}

void runCudnnBackwardFilter(const Options &options, cudaStream_t stream) {
    if (!selected(options, "cudnn_backward_filter_3x3")) { return; }

    const Options neural = neuralOptions(options);
    af::deviceGC();

    af::array primeSignal = af::randu(8, 8, 1, 1, f32);
    af::array primeFilter = af::randu(1, 1, 1, 1, f32);
    af::array primeOutput =
        af::convolve2NN(primeSignal, primeFilter, af::dim4(1, 1),
                        af::dim4(0, 0, 1, 1), af::dim4(1, 1));
    af::array primeGradient = af::randu(primeOutput.dims(), f32);
    materialize(af::convolve2GradientNN(
        primeGradient, primeSignal, primeFilter, primeOutput, af::dim4(1, 1),
        af::dim4(0, 0, 1, 1), af::dim4(1, 1), AF_CONV_GRADIENT_FILTER));

    const af::dim4 stride(1, 1);
    const af::dim4 padding(1, 1);
    const af::dim4 dilation(1, 1);
    af::array signal = af::randu(neural.size, neural.size, 8, 2, f32);
    af::array filter = af::randu(3, 3, 8, 16, f32);
    af::array output =
        af::convolve2NN(signal, filter, stride, padding, dilation);
    af::array incomingGradient = af::randu(output.dims(), f32);
    materialize(signal);
    materialize(filter);
    materialize(output);
    materialize(incomingGradient);

    Work work = [&incomingGradient, &signal, &filter, &output, &stride,
                 &padding, &dilation]() {
        return af::convolve2GradientNN(incomingGradient, signal, filter, output,
                                       stride, padding, dilation,
                                       AF_CONV_GRADIENT_FILTER);
    };
    printMetrics("cudnn_backward_filter_3x3", neural,
                 measure(work, neural.iterations, stream));
}

void runVolumeConvolution(const Options &options, cudaStream_t stream) {
    if (!selected(options, "convolve3_5x5_c64")) { return; }

    const Options volume = volumeOptions(options);
    af::deviceGC();
    af::array signal = af::randu(volume.size, volume.size, volume.size, c64);
    af::array filter = af::randu(5, 5, 5, c64);
    materialize(signal);
    materialize(filter);

    Work work = [signal, filter]() {
        return af::convolve3(signal, filter, AF_CONV_DEFAULT, AF_CONV_SPATIAL);
    };
    printMetrics("convolve3_5x5_c64", volume,
                 measure(work, volume.iterations, stream));
}

void runMorphology(const Options &options, cudaStream_t stream) {
    if (!selected(options, "dilate_7x7")) { return; }

    af::deviceGC();
    af::array input = af::randu(options.size, options.size, f32);
    af::array mask  = af::constant(1.0f, 7, 7, f32);
    materialize(input);
    materialize(mask);

    Work work = [input, mask]() { return af::dilate(input, mask); };
    printMetrics("dilate_7x7", options,
                 measure(work, options.iterations, stream));
}

void runVolumeMorphology(const Options &options, cudaStream_t stream) {
    if (!selected(options, "dilate3_7x7_f64")) { return; }

    const Options volume = volumeOptions(options);
    af::deviceGC();
    af::array input = af::randu(volume.size, volume.size, volume.size, f64);
    af::array mask  = af::constant(1.0, 7, 7, 7, f64);
    materialize(input);
    materialize(mask);

    Work work = [input, mask]() { return af::dilate3(input, mask); };
    printMetrics("dilate3_7x7_f64", volume,
                 measure(work, volume.iterations, stream));
}

void runTransform(const Options &options, cudaStream_t stream) {
    if (!selected(options, "transform_bilinear")) { return; }

    af::deviceGC();
    af::array input               = af::randu(options.size, options.size, f32);
    const float transformValues[] = {1.0f, 0.0f, 0.25f, 0.0f, 1.0f, 0.5f};
    af::array transform(3, 2, transformValues);
    materialize(input);
    materialize(transform);

    Work work = [input, transform, options]() {
        return af::transform(input, transform, options.size, options.size,
                             AF_INTERP_BILINEAR, false);
    };
    printMetrics("transform_bilinear", options,
                 measure(work, options.iterations, stream));
}

void runPerspectiveTransform(const Options &options, cudaStream_t stream) {
    if (!selected(options, "transform_perspective_bilinear")) { return; }

    af::deviceGC();
    af::array input               = af::randu(options.size, options.size, f32);
    const float transformValues[] = {1.0f, 0.0f,  0.25f,    0.0f, 1.0f,
                                     0.5f, 1e-5f, -5.0e-6f, 1.0f};
    af::array transform(3, 3, transformValues);
    materialize(input);
    materialize(transform);

    Work work = [input, transform, options]() {
        return af::transform(input, transform, options.size, options.size,
                             AF_INTERP_BILINEAR, false);
    };
    printMetrics("transform_perspective_bilinear", options,
                 measure(work, options.iterations, stream));
}

Options parseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--device" && i + 1 < argc) {
            options.device = std::atoi(argv[++i]);
        } else if (argument == "--iterations" && i + 1 < argc) {
            options.iterations = std::atoi(argv[++i]);
        } else if (argument == "--size" && i + 1 < argc) {
            options.size = static_cast<dim_t>(std::atoll(argv[++i]));
        } else if (argument == "--case" && i + 1 < argc) {
            options.selectedCase = argv[++i];
        } else if (argument == "--help") {
            std::cout << "Usage: cuda_backend_cuda [--device N] [--size N] "
                         "[--iterations N] [--case NAME]\n"
                      << "Cases: all, jit_contiguous, jit_gapped, reduce_dim0, "
                         "reduce_jit_dim0, reduce_jit_dim0_materialized, "
                         "reduce_jit_all, reduce_jit_all_materialized, "
                         "reduce_jit_reuse, reduce_jit_reuse_materialized, "
                         "reduce_jit_all_f64_affine_256, "
                         "reduce_jit_all_f64_affine_256_materialized, "
                         "reduce_jit_all_f64_complex_256, "
                         "reduce_jit_all_f64_complex_256_materialized, "
                         "reduce_jit_all_f64_deep_1024, "
                         "reduce_jit_all_f64_deep_1024_materialized, "
                         "reduce_jit_all_reuse4, "
                         "reduce_jit_all_reuse4_materialized, "
                         "reduce_jit_dim0_complex_32768x2, "
                         "reduce_jit_dim0_complex_32768x2_materialized, "
                         "reduce_jit_short_dim0, reduce_jit_gapped, "
                         "matmul, matmul_batched_8, "
                         "matmul_batched_d2_broadcast_8, matmul_batched_256, "
                         "sort_batched, sort_values_32, sort_values_32_dim1, "
                         "sort_index_32, sort_by_key_32, "
                         "sort_values_256_segmented, "
                         "sort_index_256_segmented, "
                         "sort_by_key_256_segmented, sort_index_256, "
                         "sort_by_key_256, sort_index_256_gapped_1024, "
                         "sort_values_100000, sort_values_iterative_10, "
                         "convolve1_9, convolve2_7x7, "
                         "convolve2_separable_7x7, dilate_7x7, "
                         "cudnn_forward_3x3, cudnn_backward_filter_3x3, "
                         "convolve3_5x5_c64, dilate3_7x7_f64, "
                         "transform_bilinear, "
                         "transform_perspective_bilinear\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown or incomplete argument: " +
                                        argument);
        }
    }

    if (options.iterations < 1) {
        throw std::invalid_argument("--iterations must be positive");
    }
    if (options.size < 32) {
        throw std::invalid_argument("--size must be at least 32");
    }
    return options;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parseOptions(argc, argv);
        af::setDevice(options.device);

        const int nativeDevice = afcu::getNativeId(options.device);
        checkCuda(cudaSetDevice(nativeDevice), "cudaSetDevice");
        cudaStream_t stream = afcu::getStream(options.device);

        char name[256]     = {};
        char platform[256] = {};
        char toolkit[256]  = {};
        char compute[256]  = {};
        af::deviceInfo(name, platform, toolkit, compute);

        std::cout << "# device=" << options.device
                  << ",native_device=" << nativeDevice << ",name=" << name
                  << ",platform=" << platform << ",toolkit=" << toolkit
                  << ",compute=" << compute << '\n';
        std::cout
            << "case,size,iterations,first_call_ms,warm_device_ms,"
               "warm_wall_ms,enqueue_us,pool_delta_bytes,"
               "pool_delta_buffers,locked_delta_bytes,locked_delta_buffers\n";

        runJitContiguous(options, stream);
        runJitGapped(options, stream);
        runReduction(options, stream);
        runJitReductions(options, stream);
        runJitReductionRegressions(options, stream);
        runShortJitReduction(options, stream);
        runGappedJitReduction(options, stream);
        runMatmul(options, stream);
        runBatchedMatmul(options, stream);
        runBroadcastBatchedMatmul(options, stream);
        runComputeBoundBatchedMatmul(options, stream);
        runSort(options, stream);
        runShortSegmentSorts(options, stream);
        runMediumSegmentSorts(options, stream);
        runLongSegmentSort(options, stream);
        runIterativeSort(options, stream);
        runOneDimensionalConvolution(options, stream);
        runConvolution(options, stream);
        runSeparableConvolution(options, stream);
        runCudnnForward(options, stream);
        runCudnnBackwardFilter(options, stream);
        runVolumeConvolution(options, stream);
        runMorphology(options, stream);
        runVolumeMorphology(options, stream);
        runTransform(options, stream);
        runPerspectiveTransform(options, stream);
        af::sync();
        return 0;
    } catch (const af::exception &error) {
        std::cerr << "ArrayFire error: " << error.what() << '\n';
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
    }
    return 1;
}
