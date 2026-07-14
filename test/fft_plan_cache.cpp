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
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using af::array;
using af::cfloat;
using af::dim4;
using af::dtype;
using af::fft;
using af::fft2InPlace;
using af::fft3InPlace;
using af::fftC2R;
using af::fftInPlace;
using af::fftR2C;
using af::ifft2InPlace;
using af::ifft3InPlace;
using af::ifftInPlace;
using af::randu;
using af::seq;
using af::span;
using std::vector;

namespace {

class FFTPlanCacheTest : public ::testing::Test {
   protected:
    void TearDown() override {
        af::sync();
        af::setFFTPlanCacheSize(5);
    }
};

double inverseScale(const dim4 &dims, const int rank) {
    double elements = 1.0;
    for (int i = 0; i < rank; ++i) { elements *= dims[i]; }
    return 1.0 / elements;
}

array complexRoundTrip(const array &input, const int rank) {
    array result = input.copy();
    switch (rank) {
        case 1:
            fftInPlace(result);
            ifftInPlace(result);
            break;
        case 2:
            fft2InPlace(result);
            ifft2InPlace(result);
            break;
        case 3:
            fft3InPlace(result);
            ifft3InPlace(result);
            break;
        default: throw std::logic_error("invalid FFT rank");
    }
    return result;
}

template<int rank>
void checkRealRoundTrip(const dim4 &dims, const dtype type,
                        const double tolerance) {
    array input          = randu(dims, type);
    const double forward = inverseScale(dims, rank);
    array spectrum       = fftR2C<rank>(input, forward);
    array output         = fftC2R<rank>(spectrum, (dims[0] & 1) != 0, 1.0);
    ASSERT_ARRAYS_NEAR(input, output, tolerance);
}

void checkInPlaceView(array view, const double tolerance) {
    array expected = fft(view);
    expected.eval();
    af::sync();

    fftInPlace(view);
    ASSERT_ARRAYS_NEAR(expected, view, tolerance);
}

class Barrier {
   public:
    explicit Barrier(const size_t count) : remaining_(count) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (--remaining_ == 0) {
            condition_.notify_all();
        } else {
            condition_.wait(lock, [&]() { return remaining_ == 0; });
        }
    }

   private:
    size_t remaining_;
    std::mutex mutex_;
    std::condition_variable condition_;
};

}  // namespace

TEST_F(FFTPlanCacheTest, DisabledCache) {
    ASSERT_SUCCESS(af_set_fft_plan_cache_size(0));
    af::setFFTPlanCacheSize(0);

    const array complexInput = randu(dim4(31, 5), c32);
    for (int i = 0; i < 3; ++i) {
        const array output = complexRoundTrip(complexInput, 1);
        ASSERT_ARRAYS_NEAR(complexInput, output, 1e-3);
    }

    checkRealRoundTrip<1>(dim4(31, 4), f32, 1e-3);
    checkRealRoundTrip<2>(dim4(15, 7, 2), f64, 1e-10);
}

TEST_F(FFTPlanCacheTest, SamePlanDifferentBuffers) {
    const array first  = randu(dim4(64, 5), c32);
    const array second = randu(dim4(64, 5), c32);

    af::setFFTPlanCacheSize(0);
    array firstExpected  = fft(first);
    array secondExpected = fft(second);
    firstExpected.eval();
    secondExpected.eval();
    af::sync();

    af::setFFTPlanCacheSize(5);
    array firstActual  = first.copy();
    array secondActual = second.copy();
    fftInPlace(firstActual);
    fftInPlace(secondActual);

    ASSERT_ARRAYS_NEAR(firstExpected, firstActual, 1e-4);
    ASSERT_ARRAYS_NEAR(secondExpected, secondActual, 1e-4);

    const dim4 realDims(33, 7, 2);
    const array realFirst      = randu(realDims, f32);
    const array realSecond     = randu(realDims, f32);
    const double forward       = inverseScale(realDims, 2);
    const array firstSpectrum  = fftR2C<2>(realFirst, forward);
    const array secondSpectrum = fftR2C<2>(realSecond, forward);
    const array firstReal      = fftC2R<2>(firstSpectrum, true, 1.0);
    const array secondReal     = fftC2R<2>(secondSpectrum, true, 1.0);

    ASSERT_ARRAYS_NEAR(realFirst, firstReal, 1e-3);
    ASSERT_ARRAYS_NEAR(realSecond, secondReal, 1e-3);
}

TEST_F(FFTPlanCacheTest, EvictionResizeAndBatchedKeys) {
    af::setFFTPlanCacheSize(1);

    const array first = randu(dim4(31, 3), c32);
    ASSERT_ARRAYS_NEAR(first, complexRoundTrip(first, 1), 1e-3);

    const array second = randu(dim4(37, 2), c64);
    ASSERT_ARRAYS_NEAR(second, complexRoundTrip(second, 1), 1e-10);

    checkRealRoundTrip<2>(dim4(15, 7, 3, 2), f32, 1e-3);
    ASSERT_ARRAYS_NEAR(first, complexRoundTrip(first, 1), 1e-3);

    af::setFFTPlanCacheSize(4);
    const array batchTwo   = randu(dim4(13, 7, 2), c32);
    const array batchThree = randu(dim4(13, 7, 3), c32);
    ASSERT_ARRAYS_NEAR(batchTwo, complexRoundTrip(batchTwo, 2), 2e-3);
    ASSERT_ARRAYS_NEAR(batchThree, complexRoundTrip(batchThree, 2), 2e-3);

    const array batched = randu(dim4(13, 7, 3, 2), c32);
    ASSERT_ARRAYS_NEAR(batched, complexRoundTrip(batched, 2), 2e-3);
    ASSERT_ARRAYS_NEAR(batched, complexRoundTrip(batched, 3), 2e-3);

    af::setFFTPlanCacheSize(1);
    af::setFFTPlanCacheSize(0);
    ASSERT_ARRAYS_NEAR(first, complexRoundTrip(first, 1), 1e-3);
}

TEST_F(FFTPlanCacheTest, AlignmentAndEmbeddingKeys) {
    af::setFFTPlanCacheSize(8);

    array alignedParent  = randu(dim4(65, 3), c32);
    array offsetParent   = randu(dim4(65, 3), c32);
    array differentEmbed = randu(dim4(66, 3), c32);
    array alignedView    = alignedParent(seq(0, 63), span);
    array offsetView     = offsetParent(seq(1, 64), span);
    array otherEmbedView = differentEmbed(seq(0, 63), span);

    checkInPlaceView(alignedView, 1e-4);
    checkInPlaceView(offsetView, 1e-4);
    checkInPlaceView(otherEmbedView, 1e-4);
}

TEST_F(FFTPlanCacheTest, C2RStridedViewUsesCopiedDimensions) {
    af::setFFTPlanCacheSize(5);

    array parent    = randu(dim4(9, 8, 3, 2), c32);
    array view      = parent(seq(1, 6), seq(1, 5), span, span);
    array preserved = view.copy();
    array compact   = view.copy();
    preserved.eval();
    compact.eval();

    array expected = fftC2R<2>(compact, false, 1.0);
    array actual   = fftC2R<2>(view, false, 1.0);

    ASSERT_ARRAYS_NEAR(expected, actual, 1e-3);
    ASSERT_ARRAYS_EQ(preserved, view);
}

TEST_F(FFTPlanCacheTest, ConcurrentColdMissReuseAndResize) {
    constexpr size_t workerCount = 6;
    constexpr int iterations     = 6;
    const dim4 dims(31, 17);
    const size_t elements = static_cast<size_t>(dims.elements());

    vector<array> inputs;
    vector<vector<cfloat>> expected(workerCount, vector<cfloat>(elements));
    vector<vector<cfloat>> output(workerCount, vector<cfloat>(elements));
    vector<std::exception_ptr> errors(workerCount + 1);
    inputs.reserve(workerCount);
    for (size_t i = 0; i < workerCount; ++i) {
        inputs.emplace_back(randu(dims, c32));
        inputs.back().eval();
    }
    af::sync();
    for (size_t i = 0; i < workerCount; ++i) {
        inputs[i].host(expected[i].data());
    }

    af::setFFTPlanCacheSize(0);
    af::setFFTPlanCacheSize(2);
    Barrier barrier(workerCount + 1);
    vector<std::thread> workers;
    workers.reserve(workerCount);

    for (size_t i = 0; i < workerCount; ++i) {
        workers.emplace_back([&, i]() {
            try {
                af::setDevice(0);
                array value = inputs[i].copy();
                barrier.wait();
                for (int j = 0; j < iterations; ++j) {
                    fft2InPlace(value);
                    ifft2InPlace(value);
                }
                value.host(output[i].data());
            } catch (...) { errors[i] = std::current_exception(); }
        });
    }

    std::thread resizer([&]() {
        try {
            barrier.wait();
            constexpr size_t sizes[] = {0, 1, 2, 5};
            for (int i = 0; i < 64; ++i) {
                af::setFFTPlanCacheSize(sizes[i % 4]);
            }
        } catch (...) { errors[workerCount] = std::current_exception(); }
    });

    for (auto &worker : workers) { worker.join(); }
    resizer.join();

    for (size_t i = 0; i < errors.size(); ++i) {
        if (errors[i]) {
            try {
                std::rethrow_exception(errors[i]);
            } catch (const std::exception &error) {
                FAIL() << "worker " << i << " failed: " << error.what();
            } catch (...) { FAIL() << "worker " << i << " failed"; }
        }
    }

    for (size_t i = 0; i < workerCount; ++i) {
        for (size_t j = 0; j < elements; ++j) {
            const float realError = expected[i][j].real - output[i][j].real;
            const float imagError = expected[i][j].imag - output[i][j].imag;
            ASSERT_NEAR(std::hypot(realError, imagError), 0.0, 3e-3)
                << "worker " << i << ", element " << j;
        }
    }
}
