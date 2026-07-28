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
                         "matmul, sort_batched, convolve2_7x7, dilate_7x7, "
                         "convolve3_5x5_c64, dilate3_7x7_f64, "
                         "transform_bilinear\n";
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
        runMatmul(options, stream);
        runSort(options, stream);
        runConvolution(options, stream);
        runVolumeConvolution(options, stream);
        runMorphology(options, stream);
        runVolumeMorphology(options, stream);
        runTransform(options, stream);
        af::sync();
        return 0;
    } catch (const af::exception &error) {
        std::cerr << "ArrayFire error: " << error.what() << '\n';
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
    }
    return 1;
}
