/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int run(const std::string &cacheDirectory, int device) {
    const af_err cacheError =
        af_set_kernel_cache_directory(cacheDirectory.c_str(), true);
    if (cacheError != AF_SUCCESS) {
        std::cerr << "af_set_kernel_cache_directory failed with "
                  << static_cast<int>(cacheError) << '\n';
        return 2;
    }

    af::setDevice(device);

    const dim_t elementCount = 257;
    std::vector<float> signalValues(elementCount);
    for (dim_t i = 0; i < elementCount; ++i) {
        signalValues[static_cast<size_t>(i)] =
            static_cast<float>((i * 17 + 3) % 101) / 16.0f;
    }
    const float filterValues[] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f};

    af::array signal(elementCount, signalValues.data());
    af::array filter(9, filterValues);
    af::array output =
        af::convolve1(signal, filter, AF_CONV_DEFAULT, AF_CONV_SPATIAL);
    output.eval();
    af::sync();

    if (output.dims() != af::dim4(elementCount)) {
        std::cerr << "unexpected output dimensions: " << output.dims() << '\n';
        return 3;
    }

    std::vector<float> outputValues(elementCount);
    output.host(outputValues.data());
    for (dim_t i = 0; i < elementCount; ++i) {
        const size_t index = static_cast<size_t>(i);
        if (std::abs(outputValues[index] - signalValues[index]) > 1.0e-6f) {
            std::cerr << "output mismatch at " << i << ": expected "
                      << signalValues[index] << ", received "
                      << outputValues[index] << '\n';
            return 4;
        }
    }

    std::cout << "cuda-kernel-cache-helper-ok\n";
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: " << argv[0] << " CACHE_DIRECTORY [DEVICE]\n";
        return 1;
    }

    const int device = argc == 3 ? std::atoi(argv[2]) : 0;
    try {
        return run(argv[1], device);
    } catch (const af::exception &error) {
        std::cerr << error.what() << '\n';
        return 5;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 6;
    }
}
