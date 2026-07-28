/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/Version.hpp>

#include <cstddef>

namespace arrayfire {
namespace cuda {

inline common::Version cudnnVersionComponents(std::size_t version) {
    // cuDNN 9 expanded the minor and patch fields:
    //   pre-9: major * 1000 + minor * 100 + patch
    //   9+:    major * 10000 + minor * 100 + patch
    if (version >= 90000) {
        return {static_cast<int>(version / 10000),
                static_cast<int>((version % 10000) / 100),
                static_cast<int>(version % 100)};
    }
    return {static_cast<int>(version / 1000),
            static_cast<int>((version % 1000) / 100),
            static_cast<int>(version % 100)};
}

inline common::Version cudnnCudaRuntimeVersionComponents(std::size_t version) {
    return {static_cast<int>(version / 1000),
            static_cast<int>((version % 1000) / 10),
            static_cast<int>(version % 10)};
}

}  // namespace cuda
}  // namespace arrayfire
