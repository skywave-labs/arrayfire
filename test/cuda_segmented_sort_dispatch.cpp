/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cuda/kernel/segmented_sort_dispatch.hpp>
#include <gtest/gtest.h>

namespace {

using af::dim4;
using arrayfire::cuda::kernel::useSegmentedSort;

TEST(CudaSegmentedSortDispatch, PreservesExistingLowerThresholds) {
    EXPECT_FALSE(useSegmentedSort(dim4(256, 10), 0, false));
    EXPECT_TRUE(useSegmentedSort(dim4(256, 11), 0, false));

    EXPECT_FALSE(useSegmentedSort(dim4(256, 4), 0, true));
    EXPECT_TRUE(useSegmentedSort(dim4(256, 5), 0, true));
}

TEST(CudaSegmentedSortDispatch, UsesMeasuredSafeEnvelope) {
    EXPECT_TRUE(useSegmentedSort(dim4(32, 128), 0, false));
    EXPECT_TRUE(useSegmentedSort(dim4(64, 128), 0, true));
    EXPECT_TRUE(useSegmentedSort(dim4(256, 128), 0, false));
    EXPECT_TRUE(useSegmentedSort(dim4(256, 128), 0, true));
    EXPECT_TRUE(useSegmentedSort(dim4(4096, 128), 0, false));
    EXPECT_TRUE(useSegmentedSort(dim4(4096, 128), 0, true));

    EXPECT_FALSE(useSegmentedSort(dim4(32, 129), 0, false));
    EXPECT_FALSE(useSegmentedSort(dim4(256, 129), 0, true));
    EXPECT_FALSE(useSegmentedSort(dim4(4097, 11), 0, false));
    EXPECT_FALSE(useSegmentedSort(dim4(4097, 5), 0, true));
}

TEST(CudaSegmentedSortDispatch, RejectsMeasuredRegressionShapes) {
    EXPECT_FALSE(useSegmentedSort(dim4(32, 4096), 0, false));
    EXPECT_FALSE(useSegmentedSort(dim4(32, 4096), 0, true));
    EXPECT_FALSE(useSegmentedSort(dim4(256, 1024), 0, true));
    EXPECT_FALSE(useSegmentedSort(dim4(1024, 256), 1, true));
    EXPECT_FALSE(useSegmentedSort(dim4(100000, 11), 0, false));
    EXPECT_FALSE(useSegmentedSort(dim4(99999, 5), 0, true));
}

TEST(CudaSegmentedSortDispatch, RejectsInvalidSortDimensionsAndLines) {
    EXPECT_FALSE(useSegmentedSort(dim4(1, 128), 0, false));
    EXPECT_FALSE(useSegmentedSort(dim4(256, 64), 4, false));
}

}  // namespace
