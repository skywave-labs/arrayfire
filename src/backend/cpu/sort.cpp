/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <copy.hpp>
#include <kernel/sort.hpp>
#include <queue.hpp>
#include <sort.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> sort(const Array<T>& in, const unsigned dim, bool isAscending) {
    Array<T> out = copyArray<T>(in);
    switch (dim) {
        case 0:
        case 1:
        case 2:
        case 3:
            getQueue().enqueue(kernel::sortDim<T>, out, dim, isAscending);
            break;
        default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
    }
    return out;
}

#define INSTANTIATE(T)                                                \
    template Array<T> sort<T>(const Array<T>& in, const unsigned dim, \
                              bool isAscending);

INSTANTIATE(float)
INSTANTIATE(double)
// INSTANTIATE(cfloat)
// INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)

}  // namespace cpu
}  // namespace arrayfire
