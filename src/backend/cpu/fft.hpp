/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

#include <cstddef>
#include <shared_mutex>

namespace af {
class dim4;
}

namespace arrayfire {
namespace cpu {

std::shared_mutex &fftwMutex();

void setFFTPlanCacheSize(size_t numPlans);

// Execute immediately; callers that are not already on the CPU worker queue
// should use the Array overload below.
template<typename T>
void fft_inplace(Param<T> in, const af::dim4 dataDims, const int rank,
                 const bool direction);

template<typename T>
void fft_inplace(Array<T> &in, const int rank, const bool direction);

template<typename Tc, typename Tr>
Array<Tc> fft_r2c(const Array<Tr> &in, const int rank);

template<typename Tr, typename Tc>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims, const int rank);
}  // namespace cpu
}  // namespace arrayfire
