/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr size_t blockElements       = 1 << 16;
constexpr size_t minParallelElements = 1 << 17;

int configureReassociatedReduceAll(const char *value) {
#if defined(_WIN32)
    return _putenv_s("AF_CPU_REDUCE_ALL_REASSOCIATE", value);
#else
    return setenv("AF_CPU_REDUCE_ALL_REASSOCIATE", value, 1);
#endif
}

const int initialEnvironmentResult = configureReassociatedReduceAll("0");

void expectBitwiseEqual(const float expected, const float actual) {
    EXPECT_EQ(0, std::memcmp(&expected, &actual, sizeof(float)));
}

}  // namespace

TEST(ReduceAllReassociateCache, StartsAtFirstLargeEligibleReduction) {
    SKIP_IF_FAST_MATH_ENABLED();
    ASSERT_EQ(0, initialEnvironmentResult);

    const float one = 1.0F;
    expectBitwiseEqual(one, af::sum<float>(af::array(1, &one)));

    ASSERT_EQ(0, configureReassociatedReduceAll("1"));
    std::vector<float> values(minParallelElements, 0.0F);
    const float large =
        std::ldexp(1.0F, std::numeric_limits<float>::digits + 1);
    values[blockElements - 1] = large;
    values[blockElements]     = -large;
    values[blockElements + 1] = one;
    const af::array input(static_cast<dim_t>(values.size()), values.data());

    // The serial left fold is one. The fixed two-block fold is zero because
    // the second block loses the trailing one before its -large is merged
    // with the first block's +large.
    expectBitwiseEqual(0.0F, af::sum<float>(input));

    ASSERT_EQ(0, configureReassociatedReduceAll("0"));
    expectBitwiseEqual(0.0F, af::sum<float>(input));
}
