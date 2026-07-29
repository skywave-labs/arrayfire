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
#include <err_cuda.hpp>
#include <kernel/segmented_sort.hpp>
#include <kernel/sort_by_key.hpp>
#include <math.hpp>
#include <reorder.hpp>
#include <sort_by_key.hpp>
#include <stdexcept>

namespace arrayfire {
namespace cuda {
template<typename Tk, typename Tv>
void sort_by_key(Array<Tk> &okey, Array<Tv> &oval, const Array<Tk> &ikey,
                 const Array<Tv> &ival, const uint dim, bool isAscending) {
    if (kernel::useSegmentedSort(ikey.dims(), dim, true)) {
        {
            Array<Tk> orderedKeys =
                dim == 0
                    ? (ikey.isReady() && ikey.isLinear() ? ikey
                                                         : copyArray<Tk>(ikey))
                    : reorder<Tk>(ikey, kernel::sortPreorder(dim));
            Array<Tv> orderedValues =
                dim == 0
                    ? (ival.isReady() && ival.isLinear() ? ival
                                                         : copyArray<Tv>(ival))
                    : reorder<Tv>(ival, kernel::sortPreorder(dim));

            okey                = createEmptyArray<Tk>(orderedKeys.dims());
            oval                = createEmptyArray<Tv>(orderedValues.dims());
            Array<uint> indices = createEmptyArray<uint>(orderedKeys.dims());
            const int elements  = static_cast<int>(orderedKeys.elements());
            const int segmentLength = static_cast<int>(orderedKeys.dims()[0]);

            kernel::segmentedSortPairs(okey.get(), indices.get(),
                                       orderedKeys.get(), elements,
                                       segmentLength, isAscending);
            kernel::gatherSegmented(oval.get(), orderedValues.get(),
                                    indices.get(), elements, segmentLength);
        }

        if (dim != 0) {
            okey = reorder<Tk>(okey, kernel::sortPostorder(dim));
            oval = reorder<Tv>(oval, kernel::sortPostorder(dim));
        }
        return;
    }

    okey = copyArray<Tk>(ikey);
    oval = copyArray<Tv>(ival);

    switch (dim) {
        case 0: kernel::sort0ByKey<Tk, Tv>(okey, oval, isAscending); break;
        case 1:
        case 2:
        case 3:
            kernel::sortByKeyBatched<Tk, Tv>(okey, oval, dim, isAscending);
            break;
        default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
    }

    if (dim != 0) {
        af::dim4 preorderDims = okey.dims();
        af::dim4 reorderDims(0, 1, 2, 3);
        reorderDims[dim] = 0;
        preorderDims[0]  = okey.dims()[dim];
        for (int i = 1; i <= (int)dim; i++) {
            reorderDims[i - 1] = i;
            preorderDims[i]    = okey.dims()[i - 1];
        }

        okey.setDataDims(preorderDims);
        oval.setDataDims(preorderDims);

        okey = reorder<Tk>(okey, reorderDims);
        oval = reorder<Tv>(oval, reorderDims);
    }
}

#define INSTANTIATE(Tk, Tv)                                        \
    template void sort_by_key<Tk, Tv>(                             \
        Array<Tk> & okey, Array<Tv> & oval, const Array<Tk> &ikey, \
        const Array<Tv> &ival, const uint dim, bool);

#define INSTANTIATE1(Tk)     \
    INSTANTIATE(Tk, float)   \
    INSTANTIATE(Tk, double)  \
    INSTANTIATE(Tk, cfloat)  \
    INSTANTIATE(Tk, cdouble) \
    INSTANTIATE(Tk, int)     \
    INSTANTIATE(Tk, uint)    \
    INSTANTIATE(Tk, short)   \
    INSTANTIATE(Tk, ushort)  \
    INSTANTIATE(Tk, char)    \
    INSTANTIATE(Tk, schar)   \
    INSTANTIATE(Tk, uchar)   \
    INSTANTIATE(Tk, intl)    \
    INSTANTIATE(Tk, uintl)

INSTANTIATE1(float)
INSTANTIATE1(double)
INSTANTIATE1(int)
INSTANTIATE1(uint)
INSTANTIATE1(short)
INSTANTIATE1(ushort)
INSTANTIATE1(char)
INSTANTIATE1(schar)
INSTANTIATE1(uchar)
INSTANTIATE1(intl)
INSTANTIATE1(uintl)

}  // namespace cuda
}  // namespace arrayfire
