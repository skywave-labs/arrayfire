/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <kernel/sort_by_key.hpp>
#include <queue.hpp>
#include <range.hpp>
#include <sort_index.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
void sort_index(Array<T> &okey, Array<uint> &oval, const Array<T> &in,
                const uint dim, bool isAscending) {
    // okey is values, oval is indices
    okey = copyArray<T>(in);
    oval = range<uint>(in.dims(), dim);

    switch (dim) {
        case 0:
        case 1:
        case 2:
        case 3:
            getQueue().enqueue(kernel::sortByKeyBatched<T, uint>, okey, oval,
                               dim, isAscending);
            break;
        default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
    }
}

#define INSTANTIATE(T)                                              \
    template void sort_index<T>(Array<T> & val, Array<uint> & idx,  \
                                const Array<T> &in, const uint dim, \
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
