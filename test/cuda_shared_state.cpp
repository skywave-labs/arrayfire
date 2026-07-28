/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <cuda/cudnn_algorithm_cache.hpp>
#include <cuda/cudnn_version.hpp>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Operation = std::function<af::array(bool)>;

struct CollidingIntHash {
    size_t operator()(int) const { return 0; }
};

using TestAlgorithmCache =
    arrayfire::cuda::detail::CudnnAlgorithmCache<int, size_t, CollidingIntHash,
                                                 3>;
using TestConvolutionKey =
    arrayfire::cuda::detail::CudnnConvolutionAlgorithmKey;
using TestConvolutionKeyCache = arrayfire::cuda::detail::CudnnAlgorithmCache<
    TestConvolutionKey, size_t,
    arrayfire::cuda::detail::CudnnConvolutionAlgorithmKeyHash, 16>;

class StartGate {
   public:
    explicit StartGate(size_t participants)
        : participants_(participants), waiting_(0), open_(false) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++waiting_;
        if (waiting_ == participants_) {
            open_ = true;
            condition_.notify_all();
        } else {
            condition_.wait(lock, [this] { return open_; });
        }
    }

   private:
    const size_t participants_;
    size_t waiting_;
    bool open_;
    std::mutex mutex_;
    std::condition_variable condition_;
};

TEST(CUDNNAlgorithmCache, ReusesValuesAndEvictsInFifoOrder) {
    TestAlgorithmCache cache;
    size_t value = 0;

    EXPECT_FALSE(cache.find(1, &value));
    EXPECT_EQ(11U, cache.insertOrGet(1, 11));
    EXPECT_EQ(22U, cache.insertOrGet(2, 22));
    EXPECT_EQ(33U, cache.insertOrGet(3, 33));
    EXPECT_EQ(3U, cache.size());

    EXPECT_TRUE(cache.find(1, &value));
    EXPECT_EQ(11U, value);
    EXPECT_EQ(11U, cache.insertOrGet(1, 99));

    EXPECT_EQ(44U, cache.insertOrGet(4, 44));
    EXPECT_EQ(TestAlgorithmCache::capacity(), cache.size());
    EXPECT_FALSE(cache.find(1, &value));
    EXPECT_TRUE(cache.find(4, &value));
    EXPECT_EQ(44U, value);

    cache.erase(4);
    EXPECT_FALSE(cache.find(4, &value));
    EXPECT_EQ(2U, cache.size());
}

TEST(CUDNNAlgorithmCache, SeparatesEveryConvolutionDescriptorField) {
    TestConvolutionKey base =
        arrayfire::cuda::detail::makeCudnnConvolutionAlgorithmKey(
            af::dim4(32, 24, 2, 2), af::dim4(3, 5, 2, 4),
            af::dim4(16, 24, 4, 2), af::dim4(2, 1), af::dim4(1, 2),
            af::dim4(1, 1), 0);

    std::vector<TestConvolutionKey> keys(8, base);
    keys[1].input[0] += 1;
    keys[2].filter[1] += 1;
    keys[3].output[2] += 1;
    keys[4].stride[0] += 1;
    keys[5].padding[1] += 1;
    keys[6].dilation[0] += 1;
    keys[7].dataType += 1;

    TestConvolutionKeyCache cache;
    size_t selections = 0;
    for (size_t index = 0; index < keys.size(); ++index) {
        const auto result = cache.getOrCreate(keys[index], [&]() {
            ++selections;
            return index + 1;
        });
        EXPECT_EQ(index + 1, result.value);
        EXPECT_EQ(TestConvolutionKeyCache::Lookup::Miss, result.lookup);
    }

    EXPECT_EQ(keys.size(), selections);
    EXPECT_EQ(keys.size(), cache.size());
    for (size_t index = 0; index < keys.size(); ++index) {
        const auto result =
            cache.getOrCreate(keys[index], []() { return 999U; });
        EXPECT_EQ(index + 1, result.value);
        EXPECT_EQ(TestConvolutionKeyCache::Lookup::Hit, result.lookup);
    }
}

TEST(CUDNNAlgorithmCache, ConcurrentInsertionUsesOneStableWinner) {
    TestAlgorithmCache cache;
    using Lookup             = TestAlgorithmCache::Lookup;
    const size_t threadCount = 8;
    StartGate gate(threadCount);
    std::atomic<size_t> selections(0);
    std::vector<size_t> values(threadCount);
    std::vector<Lookup> lookups(threadCount, Lookup::Hit);
    std::vector<std::thread> threads;

    for (size_t thread = 0; thread < threadCount; ++thread) {
        threads.emplace_back([&, thread]() {
            gate.wait();
            const auto result = cache.getOrCreate(7, [&, thread]() {
                ++selections;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                return thread + 1;
            });
            values[thread]    = result.value;
            lookups[thread]   = result.lookup;
        });
    }
    for (std::thread &thread : threads) { thread.join(); }

    size_t winner = 0;
    ASSERT_TRUE(cache.find(7, &winner));
    EXPECT_EQ(1U, cache.size());
    EXPECT_EQ(1U, selections);
    for (size_t value : values) { EXPECT_EQ(winner, value); }

    const size_t misses =
        std::count(lookups.begin(), lookups.end(), Lookup::Miss);
    const size_t waits =
        std::count(lookups.begin(), lookups.end(), Lookup::Wait);
    EXPECT_EQ(1U, misses);
    EXPECT_EQ(threadCount,
              misses + waits +
                  std::count(lookups.begin(), lookups.end(), Lookup::Hit));
}

TEST(CUDNNAlgorithmCache, SelectsDifferentKeysConcurrently) {
    TestAlgorithmCache cache;
    StartGate gate(2);
    std::mutex selectorMutex;
    std::condition_variable selectorCondition;
    size_t selectorsEntered = 0;
    std::vector<std::string> errors(2);
    std::vector<std::thread> threads;

    for (size_t thread = 0; thread < 2; ++thread) {
        threads.emplace_back([&, thread]() {
            gate.wait();
            try {
                const auto result =
                    cache.getOrCreate(static_cast<int>(thread), [&, thread]() {
                        std::unique_lock<std::mutex> lock(selectorMutex);
                        ++selectorsEntered;
                        selectorCondition.notify_all();
                        if (!selectorCondition.wait_for(
                                lock, std::chrono::seconds(2),
                                [&]() { return selectorsEntered == 2; })) {
                            throw std::runtime_error(
                                "selectors executed serially");
                        }
                        return thread + 1;
                    });
                if (result.value != thread + 1) {
                    errors[thread] = "wrong selected value";
                }
            } catch (const std::exception &error) {
                errors[thread] = error.what();
            }
        });
    }
    for (std::thread &thread : threads) { thread.join(); }

    EXPECT_EQ(2U, selectorsEntered);
    for (const std::string &error : errors) { EXPECT_TRUE(error.empty()); }
}

TEST(CUDNNAlgorithmCache, FailedSelectionIsRetried) {
    TestAlgorithmCache cache;
    size_t attempts = 0;

    EXPECT_THROW(
        cache.getOrCreate(9,
                          [&]() -> size_t {
                              ++attempts;
                              throw std::runtime_error("selection failed");
                          }),
        std::runtime_error);
    EXPECT_EQ(0U, cache.size());

    const auto result = cache.getOrCreate(9, [&]() {
        ++attempts;
        return 99U;
    });
    EXPECT_EQ(99U, result.value);
    EXPECT_EQ(2U, attempts);
    EXPECT_EQ(1U, cache.size());
}

TEST(CUDNNVersion, ParsesLegacyAndVersionNineEncodings) {
    const arrayfire::common::Version legacy =
        arrayfire::cuda::cudnnVersionComponents(8907);
    EXPECT_EQ(8, legacy.major());
    EXPECT_EQ(9, legacy.minor());
    EXPECT_EQ(7, legacy.patch());

    const arrayfire::common::Version versionNine =
        arrayfire::cuda::cudnnVersionComponents(90501);
    EXPECT_EQ(9, versionNine.major());
    EXPECT_EQ(5, versionNine.minor());
    EXPECT_EQ(1, versionNine.patch());

    const arrayfire::common::Version cudaRuntime =
        arrayfire::cuda::cudnnCudaRuntimeVersionComponents(12081);
    EXPECT_EQ(12, cudaRuntime.major());
    EXPECT_EQ(8, cudaRuntime.minor());
    EXPECT_EQ(1, cudaRuntime.patch());
}

std::vector<float> host(const af::array &value) {
    std::vector<float> output(value.elements());
    value.host(output.data());
    return output;
}

bool differs(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    if (lhs.size() != rhs.size()) { return true; }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i] && !(std::isnan(lhs[i]) && std::isnan(rhs[i]))) {
            return true;
        }
    }
    return false;
}

std::string mismatch(const std::vector<float> &gold,
                     const std::vector<float> &actual, float tolerance) {
    if (gold.size() != actual.size()) { return "output size changed"; }
    for (size_t i = 0; i < gold.size(); ++i) {
        const bool bothNaN = std::isnan(gold[i]) && std::isnan(actual[i]);
        const bool equal   = gold[i] == actual[i] || bothNaN;
        if (!equal && (!std::isfinite(gold[i]) || !std::isfinite(actual[i]) ||
                       std::abs(gold[i] - actual[i]) > tolerance)) {
            std::ostringstream stream;
            stream << "element " << i << " expected " << gold[i] << " got "
                   << actual[i];
            return stream.str();
        }
    }
    return std::string();
}

void verifyConcurrent(const Operation &operation, float tolerance,
                      size_t iterations = 64, size_t threadCount = 8) {
    const std::vector<float> gold0 = host(operation(false));
    const std::vector<float> gold1 = host(operation(true));
    ASSERT_TRUE(differs(gold0, gold1))
        << "The two parameter sets must produce distinct reference outputs";

    StartGate gate(threadCount);
    std::vector<std::string> errors(threadCount);
    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    for (size_t threadId = 0; threadId < threadCount; ++threadId) {
        threads.emplace_back([&, threadId]() {
            gate.wait();
            try {
                af::setDevice(0);
                const bool variant             = (threadId % 2) != 0;
                const std::vector<float> &gold = variant ? gold1 : gold0;
                for (size_t iteration = 0; iteration < iterations;
                     ++iteration) {
                    const std::string error =
                        mismatch(gold, host(operation(variant)), tolerance);
                    if (!error.empty()) {
                        std::ostringstream stream;
                        stream << "iteration " << iteration << ": " << error;
                        errors[threadId] = stream.str();
                        break;
                    }
                }
            } catch (const std::exception &error) {
                errors[threadId] = error.what();
            }
        });
    }

    for (std::thread &thread : threads) { thread.join(); }
    for (size_t threadId = 0; threadId < threadCount; ++threadId) {
        EXPECT_TRUE(errors[threadId].empty())
            << "thread " << threadId << ": " << errors[threadId];
    }
}

#if defined(AF_TEST_WITH_CUDNN)
std::vector<float> runConvolveNN() {
    af::setDevice(0);
    af::array signal = af::range(af::dim4(8, 8, 1, 1), 0, f32);
    af::array filter = af::constant(1.0f, 3, 3, 1, 1, f32);
    af::array output = af::convolve2NN(signal, filter, af::dim4(1, 1),
                                       af::dim4(1, 1), af::dim4(1, 1));
    output.eval();
    af::sync();
    return host(output);
}

TEST(CUDNNHandle, SurvivesFirstCallingThreadExit) {
    std::vector<float> first;
    std::exception_ptr workerError;
    std::thread firstCaller([&]() {
        try {
            first = runConvolveNN();
        } catch (...) { workerError = std::current_exception(); }
    });
    firstCaller.join();
    if (workerError) { std::rethrow_exception(workerError); }

    const std::vector<float> second = runConvolveNN();
    EXPECT_EQ(first, second);
}

TEST(CUDNNAlgorithmCache, RepeatedForwardAndBackwardFilterShapesStayAccurate) {
    af::setDevice(0);
    const af::dim4 stride(2, 1);
    const af::dim4 padding(1, 2);
    const af::dim4 dilation(1, 1);
    af::array signal = af::randu(32, 24, 2, 2, f32);
    af::array filter = af::randu(3, 5, 2, 4, f32);

    af::array forward0 =
        af::convolve2NN(signal, filter, stride, padding, dilation);
    af::array forward1 = af::convolve2NN(signal.copy(), filter.copy(), stride,
                                         padding, dilation);
    forward0.eval();
    forward1.eval();
    af::sync();
    ASSERT_ARRAYS_NEAR(forward0, forward1, 1.0e-4);

    af::array incomingGradient = af::randu(forward0.dims(), f32);
    af::array filterGradient0  = af::convolve2GradientNN(
         incomingGradient, signal, filter, forward0, stride, padding, dilation,
         AF_CONV_GRADIENT_FILTER);
    af::array filterGradient1 = af::convolve2GradientNN(
        incomingGradient.copy(), signal.copy(), filter.copy(), forward1, stride,
        padding, dilation, AF_CONV_GRADIENT_FILTER);
    filterGradient0.eval();
    filterGradient1.eval();
    af::sync();
    ASSERT_ARRAYS_NEAR(filterGradient0, filterGradient1, 4.0e-3);
}
#endif

TEST(CUDASharedState, ConcurrentSpatialConvolutionUsesPerCallFilter) {
    af::setDevice(0);
    af::array signal = af::randu(96, 80, f32);

    std::vector<float> filter0Values(25, 1.0f);
    std::vector<float> filter1Values(25, 0.0f);
    for (int i = 0; i < 5; ++i) {
        filter1Values[2 + 5 * i] = 1.0f;
        filter1Values[i + 5 * 2] = 1.0f;
    }
    af::array filter0(5, 5, filter0Values.data());
    af::array filter1(5, 5, filter1Values.data());
    signal.eval();
    filter0.eval();
    filter1.eval();
    af::sync();

    Operation operation = [signal, filter0, filter1](bool variant) {
        const af::array &filter = variant ? filter1 : filter0;
        return af::convolve2(signal, filter, AF_CONV_DEFAULT, AF_CONV_SPATIAL);
    };
    verifyConcurrent(operation, 1.0e-5f);
}

TEST(CUDASharedState, ConcurrentOneDimensionalConvolutionUsesPerCallFilter) {
    af::setDevice(0);
    af::array signal = af::randu(2048, f32);

    std::vector<float> filter0Values(9, 1.0f);
    std::vector<float> filter1Values(9, 0.0f);
    filter1Values[2] = 1.0f;
    filter1Values[4] = 2.0f;
    filter1Values[6] = 1.0f;
    af::array filter0(9, filter0Values.data());
    af::array filter1(9, filter1Values.data());
    signal.eval();
    filter0.eval();
    filter1.eval();
    af::sync();

    Operation operation = [signal, filter0, filter1](bool variant) {
        const af::array &filter = variant ? filter1 : filter0;
        return af::convolve1(signal, filter, AF_CONV_DEFAULT, AF_CONV_SPATIAL);
    };
    verifyConcurrent(operation, 1.0e-5f);
}

TEST(CUDASharedState, ConcurrentThreeDimensionalConvolutionUsesPerCallFilter) {
    af::setDevice(0);
    af::array signal = af::randu(24, 20, 16, f32);

    std::vector<float> filter0Values(27, 1.0f);
    std::vector<float> filter1Values(27, 0.0f);
    for (int i = 0; i < 3; ++i) {
        filter1Values[1 + 3 * 1 + 9 * i] = 1.0f;
        filter1Values[1 + 3 * i + 9 * 1] = 1.0f;
        filter1Values[i + 3 * 1 + 9 * 1] = 1.0f;
    }
    af::array filter0(3, 3, 3, filter0Values.data());
    af::array filter1(3, 3, 3, filter1Values.data());
    signal.eval();
    filter0.eval();
    filter1.eval();
    af::sync();

    Operation operation = [signal, filter0, filter1](bool variant) {
        const af::array &filter = variant ? filter1 : filter0;
        return af::convolve3(signal, filter, AF_CONV_DEFAULT, AF_CONV_SPATIAL);
    };
    verifyConcurrent(operation, 1.0e-5f);
}

TEST(CUDASharedState, ConcurrentSeparableConvolutionUsesPerCallFilters) {
    af::setDevice(0);
    af::array signal = af::randu(96, 80, f32);

    const float filter0Values[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float column1Values[] = {0.0f, 1.0f, 2.0f, 1.0f, 0.0f};
    const float row1Values[]    = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    af::array column0(5, filter0Values);
    af::array row0(5, filter0Values);
    af::array column1(5, column1Values);
    af::array row1(5, row1Values);
    signal.eval();
    column0.eval();
    row0.eval();
    column1.eval();
    row1.eval();
    af::sync();

    Operation operation = [signal, column0, row0, column1, row1](bool variant) {
        const af::array &column = variant ? column1 : column0;
        const af::array &row    = variant ? row1 : row0;
        return af::convolve(column, row, signal, AF_CONV_DEFAULT);
    };
    verifyConcurrent(operation, 1.0e-5f);
}

TEST(CUDASharedState, ConcurrentMorphologyUsesPerCallMask) {
    af::setDevice(0);
    af::array input = af::randu(96, 80, f32);

    std::vector<float> mask0Values(25, 1.0f);
    std::vector<float> mask1Values(25, 0.0f);
    for (int i = 0; i < 5; ++i) {
        mask1Values[2 + 5 * i] = 1.0f;
        mask1Values[i + 5 * 2] = 1.0f;
    }
    af::array mask0(5, 5, mask0Values.data());
    af::array mask1(5, 5, mask1Values.data());
    input.eval();
    mask0.eval();
    mask1.eval();
    af::sync();

    Operation operation = [input, mask0, mask1](bool variant) {
        const af::array &mask = variant ? mask1 : mask0;
        return af::dilate(input, mask);
    };
    verifyConcurrent(operation, 0.0f);
}

TEST(CUDASharedState, ConcurrentVolumeMorphologyUsesPerCallMask) {
    af::setDevice(0);
    af::array input = af::randu(24, 20, 16, f32);

    std::vector<float> mask0Values(27, 1.0f);
    std::vector<float> mask1Values(27, 0.0f);
    for (int i = 0; i < 3; ++i) {
        mask1Values[1 + 3 * 1 + 9 * i] = 1.0f;
        mask1Values[1 + 3 * i + 9 * 1] = 1.0f;
        mask1Values[i + 3 * 1 + 9 * 1] = 1.0f;
    }
    af::array mask0(3, 3, 3, mask0Values.data());
    af::array mask1(3, 3, 3, mask1Values.data());
    input.eval();
    mask0.eval();
    mask1.eval();
    af::sync();

    Operation operation = [input, mask0, mask1](bool variant) {
        const af::array &mask = variant ? mask1 : mask0;
        return af::dilate3(input, mask);
    };
    verifyConcurrent(operation, 0.0f);
}

TEST(CUDASharedState, ConcurrentTransformUsesPerCallMatrix) {
    af::setDevice(0);
    af::array input                = af::randu(96, 80, f32);
    const float transform0Values[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    const float transform1Values[] = {1.0f, 0.0f, 4.0f, 0.0f, 1.0f, 3.0f};
    af::array transform0(3, 2, transform0Values);
    af::array transform1(3, 2, transform1Values);
    input.eval();
    transform0.eval();
    transform1.eval();
    af::sync();

    Operation operation = [input, transform0, transform1](bool variant) {
        const af::array &transform = variant ? transform1 : transform0;
        return af::transform(input, transform, input.dims(0), input.dims(1),
                             AF_INTERP_NEAREST, false);
    };
    verifyConcurrent(operation, 0.0f);
}

TEST(CUDASharedState, TransformSupportsMoreThanConstantMemoryBatch) {
    af::setDevice(0);
    const dim_t transformCount = 513;
    af::array input            = af::randu(8, 8, f32);
    std::vector<float> transforms(6 * transformCount);
    for (dim_t i = 0; i < transformCount; ++i) {
        transforms[6 * i + 0] = 1.0f;
        transforms[6 * i + 1] = 0.0f;
        transforms[6 * i + 2] = 0.0f;
        transforms[6 * i + 3] = 0.0f;
        transforms[6 * i + 4] = 1.0f;
        transforms[6 * i + 5] = 0.0f;
    }
    af::array transform(af::dim4(3, 2, transformCount), transforms.data());

    af::array output =
        af::transform(input, transform, 8, 8, AF_INTERP_NEAREST, false);
    output.eval();
    af::sync();

    EXPECT_EQ(af::dim4(8, 8, transformCount), output.dims());
    const std::vector<float> first = host(output(af::span, af::span, 0));
    const std::vector<float> last =
        host(output(af::span, af::span, transformCount - 1));
    EXPECT_EQ(first, last);
}

TEST(CUDASharedState, ConcurrentCannyUsesPerCallConvergenceFlag) {
    af::setDevice(0);
    af::array input0 = af::randu(256, 256, f32);
    af::array input1 = af::flip(af::randu(256, 256, f32), 0);
    input0.eval();
    input1.eval();
    af::sync();

    Operation operation = [input0, input1](bool variant) {
        const af::array &input = variant ? input1 : input0;
        return af::canny(input, AF_CANNY_THRESHOLD_MANUAL, 0.1f, 0.3f, 3, true)
            .as(f32);
    };
    verifyConcurrent(operation, 0.0f, 8);
}

TEST(CUDASharedState, ConcurrentFloodFillUsesPerCallConvergenceFlag) {
    af::setDevice(0);
    const dim_t size = 128;
    std::vector<float> fullValues(size * size, 1.0f);
    std::vector<float> halfValues(size * size, 1.0f);
    for (dim_t y = 0; y < size; ++y) {
        for (dim_t x = size / 2; x < size; ++x) {
            halfValues[x + y * size] = 10.0f;
        }
    }
    af::array input0(size, size, fullValues.data());
    af::array input1(size, size, halfValues.data());
    const std::vector<unsigned> seedx(1, 4);
    const std::vector<unsigned> seedy(1, 4);
    input0.eval();
    input1.eval();
    af::sync();

    Operation operation = [input0, input1, seedx, seedy](bool variant) {
        const af::array &input = variant ? input1 : input0;
        return af::confidenceCC(input, seedx.size(), seedx.data(), seedy.data(),
                                1, 2, 1, 255.0);
    };
    verifyConcurrent(operation, 0.0f, 8);
}

TEST(CUDASharedState, ConcurrentRegionsUsesPerCallConvergenceFlag) {
    af::setDevice(0);
    const dim_t size = 128;
    std::vector<char> connected(size * size, 1);
    std::vector<char> divided(size * size, 1);
    for (dim_t y = 0; y < size; ++y) { divided[size / 2 + y * size] = 0; }
    af::array input0(size, size, connected.data());
    af::array input1(size, size, divided.data());
    input0.eval();
    input1.eval();
    af::sync();

    Operation operation = [input0, input1](bool variant) {
        const af::array &input = variant ? input1 : input0;
        return af::regions(input, AF_CONNECTIVITY_4, f32);
    };
    verifyConcurrent(operation, 0.0f, 8);
}

}  // namespace
