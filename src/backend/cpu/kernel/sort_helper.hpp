/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_cpu.hpp>
#include <tuple>

namespace arrayfire {
namespace cpu {
namespace kernel {
template<typename Tk, typename Tv>
using IndexPair = std::tuple<Tk, Tv>;

template<typename Tk, typename Tv, bool isAscending>
struct IPCompare {
    bool operator()(const IndexPair<Tk, Tv> &lhs,
                    const IndexPair<Tk, Tv> &rhs) const {
        const Tk &lhsVal = std::get<0>(lhs);
        const Tk &rhsVal = std::get<0>(rhs);
        if (isAscending)
            return (lhsVal < rhsVal);
        else
            return (lhsVal > rhsVal);
    }
};

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
