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
#include <array>
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
    arrayfire::cuda::detail::CudnnConvolutionAlgorithmKeyHash, 32>;

thread_local bool trackNextHash = false;
std::mutex trackedHashMutex;
std::condition_variable trackedHashCondition;
size_t trackedHashEntrants = 0;

struct TrackingIntHash {
    size_t operator()(int) const {
        if (trackNextHash) {
            trackNextHash = false;
            {
                std::lock_guard<std::mutex> lock(trackedHashMutex);
                ++trackedHashEntrants;
            }
            trackedHashCondition.notify_all();
        }
        return 0;
    }
};

using FailureTrackingCache =
    arrayfire::cuda::detail::CudnnAlgorithmCache<int, size_t, TrackingIntHash,
                                                 3>;

void resetTrackedHashEntrants() {
    std::lock_guard<std::mutex> lock(trackedHashMutex);
    trackedHashEntrants = 0;
}

bool waitForTrackedHashEntrants(size_t expected) {
    std::unique_lock<std::mutex> lock(trackedHashMutex);
    return trackedHashCondition.wait_for(
        lock, std::chrono::seconds(10),
        [expected]() { return trackedHashEntrants == expected; });
}

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

    EXPECT_EQ((std::array<dim_t, 4>{{32, 24, 2, 2}}), base.input);
    EXPECT_EQ((std::array<dim_t, 4>{{3, 5, 2, 4}}), base.filter);
    EXPECT_EQ((std::array<dim_t, 4>{{16, 24, 4, 2}}), base.output);
    EXPECT_EQ((std::array<dim_t, 2>{{2, 1}}), base.stride);
    EXPECT_EQ((std::array<dim_t, 2>{{1, 2}}), base.padding);
    EXPECT_EQ((std::array<dim_t, 2>{{1, 1}}), base.dilation);
    EXPECT_EQ(0, base.dataType);

    std::vector<TestConvolutionKey> keys(1, base);
    for (size_t coordinate = 0; coordinate < base.input.size(); ++coordinate) {
        TestConvolutionKey key = base;
        ++key.input[coordinate];
        keys.push_back(key);
    }
    for (size_t coordinate = 0; coordinate < base.filter.size(); ++coordinate) {
        TestConvolutionKey key = base;
        ++key.filter[coordinate];
        keys.push_back(key);
    }
    for (size_t coordinate = 0; coordinate < base.output.size(); ++coordinate) {
        TestConvolutionKey key = base;
        ++key.output[coordinate];
        keys.push_back(key);
    }
    for (size_t coordinate = 0; coordinate < base.stride.size(); ++coordinate) {
        TestConvolutionKey key = base;
        ++key.stride[coordinate];
        keys.push_back(key);
    }
    for (size_t coordinate = 0; coordinate < base.padding.size();
         ++coordinate) {
        TestConvolutionKey key = base;
        ++key.padding[coordinate];
        keys.push_back(key);
    }
    for (size_t coordinate = 0; coordinate < base.dilation.size();
         ++coordinate) {
        TestConvolutionKey key = base;
        ++key.dilation[coordinate];
        keys.push_back(key);
    }
    TestConvolutionKey differentType = base;
    ++differentType.dataType;
    keys.push_back(differentType);
    ASSERT_EQ(20U, keys.size());

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

TEST(CUDNNAlgorithmCache, ErasePreservesFifoOrderAcrossRefill) {
    TestAlgorithmCache cache;
    size_t value = 0;

    EXPECT_EQ(11U, cache.insertOrGet(1, 11));
    EXPECT_EQ(22U, cache.insertOrGet(2, 22));
    EXPECT_EQ(33U, cache.insertOrGet(3, 33));

    cache.erase(2);
    cache.erase(2);
    EXPECT_EQ(44U, cache.insertOrGet(4, 44));
    EXPECT_EQ(TestAlgorithmCache::capacity(), cache.size());
    EXPECT_TRUE(cache.find(1, &value));
    EXPECT_FALSE(cache.find(2, &value));
    EXPECT_TRUE(cache.find(3, &value));
    EXPECT_TRUE(cache.find(4, &value));

    EXPECT_EQ(55U, cache.insertOrGet(5, 55));
    EXPECT_FALSE(cache.find(1, &value));
    EXPECT_TRUE(cache.find(3, &value));
    EXPECT_TRUE(cache.find(4, &value));
    EXPECT_TRUE(cache.find(5, &value));

    size_t selections = 0;
    const auto refill = cache.getOrCreate(2, [&]() {
        ++selections;
        return 22U;
    });
    EXPECT_EQ(22U, refill.value);
    EXPECT_EQ(TestAlgorithmCache::Lookup::Miss, refill.lookup);
    EXPECT_EQ(1U, selections);
    EXPECT_FALSE(cache.find(3, &value));

    const auto hit = cache.getOrCreate(2, [&]() {
        ++selections;
        return 999U;
    });
    EXPECT_EQ(22U, hit.value);
    EXPECT_EQ(TestAlgorithmCache::Lookup::Hit, hit.lookup);
    EXPECT_EQ(1U, selections);
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

TEST(CUDNNAlgorithmCache,
     ConcurrentSelectionFailurePropagatesAndLaterRetryIsCached) {
    FailureTrackingCache cache;
    EXPECT_EQ(111U, cache.insertOrGet(-1, 111));

    using Lookup             = FailureTrackingCache::Lookup;
    const size_t threadCount = 8;
    StartGate gate(threadCount);
    std::atomic<size_t> failedSelections(0);
    std::vector<std::string> errors(threadCount);
    std::vector<std::thread> threads;
    resetTrackedHashEntrants();

    for (size_t thread = 0; thread < threadCount; ++thread) {
        threads.emplace_back([&, thread]() {
            gate.wait();
            trackNextHash = true;
            try {
                cache.getOrCreate(7, [&]() -> size_t {
                    ++failedSelections;
                    if (!waitForTrackedHashEntrants(threadCount)) {
                        throw std::runtime_error(
                            "not every caller reached the cache");
                    }
                    throw std::runtime_error("selection failed");
                });
                errors[thread] = "selection unexpectedly succeeded";
            } catch (const std::exception &error) {
                errors[thread] = error.what();
            }
        });
    }
    for (std::thread &thread : threads) { thread.join(); }

    EXPECT_EQ(1U, failedSelections);
    for (const std::string &error : errors) {
        EXPECT_EQ("selection failed", error);
    }
    EXPECT_EQ(1U, cache.size());
    size_t value = 0;
    EXPECT_FALSE(cache.find(7, &value));

    size_t retrySelections = 0;
    const auto retry       = cache.getOrCreate(7, [&]() {
        ++retrySelections;
        return 77U;
    });
    EXPECT_EQ(77U, retry.value);
    EXPECT_EQ(Lookup::Miss, retry.lookup);
    EXPECT_EQ(1U, retrySelections);

    const auto hit = cache.getOrCreate(7, [&]() {
        ++retrySelections;
        return 999U;
    });
    EXPECT_EQ(77U, hit.value);
    EXPECT_EQ(Lookup::Hit, hit.lookup);
    EXPECT_EQ(1U, retrySelections);
    EXPECT_EQ(2U, cache.size());
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

void verifyTransformBatchBeyondLegacyConstantMemory(bool perspective) {
    af::setDevice(0);
    const dim_t width            = 11;
    const dim_t height           = 9;
    const dim_t coefficientCount = perspective ? 9 : 6;
    const dim_t transformCount   = perspective ? 342 : 513;

    std::vector<float> inputValues(width * height);
    for (dim_t y = 0; y < height; ++y) {
        for (dim_t x = 0; x < width; ++x) {
            inputValues[x + y * width] = static_cast<float>(1 + x + y * width);
        }
    }
    af::array input(width, height, inputValues.data());

    std::vector<float> transforms(coefficientCount * transformCount);
    for (dim_t index = 0; index < transformCount; ++index) {
        const size_t offset = static_cast<size_t>(coefficientCount * index);
        const float translateX =
            static_cast<float>(static_cast<int>(index % 5) - 2);
        const float translateY =
            static_cast<float>(static_cast<int>((index / 5) % 5) - 2);
        transforms[offset + 0] = 1.0f;
        transforms[offset + 1] = 0.0f;
        transforms[offset + 2] = translateX;
        transforms[offset + 3] = 0.0f;
        transforms[offset + 4] = 1.0f;
        transforms[offset + 5] = translateY;
        if (perspective) {
            transforms[offset + 6] =
                0.001f * static_cast<float>(static_cast<int>(index % 3) - 1);
            transforms[offset + 7] =
                0.001f *
                static_cast<float>(static_cast<int>((index / 3) % 3) - 1);
            transforms[offset + 8] = 1.0f;
        }
    }
    af::array transform(af::dim4(3, perspective ? 3 : 2, transformCount),
                        transforms.data());

    af::array output = af::transform(input, transform, width, height,
                                     AF_INTERP_NEAREST, false);
    output.eval();
    af::sync();

    ASSERT_EQ(af::dim4(width, height, transformCount), output.dims());
    const std::vector<dim_t> selectedIndices =
        perspective ? std::vector<dim_t>{0, 1, 170, 340, 341}
                    : std::vector<dim_t>{0, 1, 255, 511, 512};
    std::vector<std::vector<float>> selectedOutputs;
    selectedOutputs.reserve(selectedIndices.size());
    for (dim_t index : selectedIndices) {
        const size_t offset = static_cast<size_t>(coefficientCount * index);
        af::array singleTransform(af::dim4(3, perspective ? 3 : 2),
                                  transforms.data() + offset);
        const std::vector<float> reference = host(af::transform(
            input, singleTransform, width, height, AF_INTERP_NEAREST, false));
        const std::vector<float> batched =
            host(output(af::span, af::span, index));
        const std::string error = mismatch(reference, batched, 0.0f);
        EXPECT_TRUE(error.empty())
            << "transform index " << index << ": " << error;
        selectedOutputs.push_back(batched);
    }

    for (size_t lhs = 0; lhs < selectedOutputs.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < selectedOutputs.size(); ++rhs) {
            EXPECT_TRUE(differs(selectedOutputs[lhs], selectedOutputs[rhs]))
                << "transform indices " << selectedIndices[lhs] << " and "
                << selectedIndices[rhs]
                << " must produce observably distinct outputs";
        }
    }
}

#if defined(AF_TEST_WITH_CUDNN)
using ArrayOperation = std::function<af::array()>;

size_t index4(const af::dim4 &dims, dim_t x, dim_t y, dim_t z, dim_t w) {
    return static_cast<size_t>(x + dims[0] * (y + dims[1] * (z + dims[2] * w)));
}

std::vector<float> hostConvolve2NN(
    const std::vector<float> &signal, const af::dim4 &signalDims,
    const std::vector<float> &filter, const af::dim4 &filterDims,
    const af::dim4 &stride, const af::dim4 &padding, const af::dim4 &dilation) {
    const dim_t outputWidth = 1 + (signalDims[0] + 2 * padding[0] -
                                   (((filterDims[0] - 1) * dilation[0]) + 1)) /
                                      stride[0];
    const dim_t outputHeight = 1 + (signalDims[1] + 2 * padding[1] -
                                    (((filterDims[1] - 1) * dilation[1]) + 1)) /
                                       stride[1];
    const af::dim4 outputDims(outputWidth, outputHeight, filterDims[3],
                              signalDims[3]);
    std::vector<float> output(outputDims.elements(), 0.0f);

    for (dim_t batch = 0; batch < signalDims[3]; ++batch) {
        for (dim_t outputChannel = 0; outputChannel < filterDims[3];
             ++outputChannel) {
            for (dim_t outputY = 0; outputY < outputHeight; ++outputY) {
                for (dim_t outputX = 0; outputX < outputWidth; ++outputX) {
                    double sum = 0.0;
                    for (dim_t inputChannel = 0; inputChannel < signalDims[2];
                         ++inputChannel) {
                        for (dim_t filterY = 0; filterY < filterDims[1];
                             ++filterY) {
                            const dim_t inputY = outputY * stride[1] +
                                                 filterY * dilation[1] -
                                                 padding[1];
                            if (inputY < 0 || inputY >= signalDims[1]) {
                                continue;
                            }
                            for (dim_t filterX = 0; filterX < filterDims[0];
                                 ++filterX) {
                                const dim_t inputX = outputX * stride[0] +
                                                     filterX * dilation[0] -
                                                     padding[0];
                                if (inputX < 0 || inputX >= signalDims[0]) {
                                    continue;
                                }
                                const dim_t flippedX =
                                    filterDims[0] - 1 - filterX;
                                const dim_t flippedY =
                                    filterDims[1] - 1 - filterY;
                                sum += static_cast<double>(signal[index4(
                                           signalDims, inputX, inputY,
                                           inputChannel, batch)]) *
                                       filter[index4(filterDims, flippedX,
                                                     flippedY, inputChannel,
                                                     outputChannel)];
                            }
                        }
                    }
                    output[index4(outputDims, outputX, outputY, outputChannel,
                                  batch)] = static_cast<float>(sum);
                }
            }
        }
    }
    return output;
}

std::vector<float> hostConvolve2FilterGradientNN(
    const std::vector<float> &incomingGradient,
    const af::dim4 &incomingGradientDims, const std::vector<float> &signal,
    const af::dim4 &signalDims, const af::dim4 &filterDims,
    const af::dim4 &stride, const af::dim4 &padding, const af::dim4 &dilation) {
    std::vector<float> output(filterDims.elements(), 0.0f);
    for (dim_t outputChannel = 0; outputChannel < filterDims[3];
         ++outputChannel) {
        for (dim_t inputChannel = 0; inputChannel < filterDims[2];
             ++inputChannel) {
            for (dim_t filterY = 0; filterY < filterDims[1]; ++filterY) {
                for (dim_t filterX = 0; filterX < filterDims[0]; ++filterX) {
                    double sum                   = 0.0;
                    const dim_t unwrappedFilterX = filterDims[0] - 1 - filterX;
                    const dim_t unwrappedFilterY = filterDims[1] - 1 - filterY;
                    for (dim_t batch = 0; batch < signalDims[3]; ++batch) {
                        for (dim_t outputY = 0;
                             outputY < incomingGradientDims[1]; ++outputY) {
                            const dim_t inputY =
                                outputY * stride[1] +
                                unwrappedFilterY * dilation[1] - padding[1];
                            if (inputY < 0 || inputY >= signalDims[1]) {
                                continue;
                            }
                            for (dim_t outputX = 0;
                                 outputX < incomingGradientDims[0]; ++outputX) {
                                const dim_t inputX =
                                    outputX * stride[0] +
                                    unwrappedFilterX * dilation[0] - padding[0];
                                if (inputX < 0 || inputX >= signalDims[0]) {
                                    continue;
                                }
                                sum += static_cast<double>(signal[index4(
                                           signalDims, inputX, inputY,
                                           inputChannel, batch)]) *
                                       incomingGradient[index4(
                                           incomingGradientDims, outputX,
                                           outputY, outputChannel, batch)];
                            }
                        }
                    }
                    output[index4(filterDims, filterX, filterY, inputChannel,
                                  outputChannel)] = static_cast<float>(sum);
                }
            }
        }
    }
    return output;
}

void verifyColdConcurrentVariants(const Operation &operation,
                                  const std::vector<float> &gold0,
                                  const std::vector<float> &gold1,
                                  float tolerance, size_t threadCount = 4) {
    ASSERT_TRUE(differs(gold0, gold1))
        << "The same-key variants must have distinct host references";
    StartGate gate(threadCount);
    std::vector<std::vector<float>> actual(threadCount);
    std::vector<std::string> errors(threadCount);
    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    for (size_t thread = 0; thread < threadCount; ++thread) {
        threads.emplace_back([&, thread]() {
            gate.wait();
            try {
                af::setDevice(0);
                actual[thread] = host(operation((thread % 2) != 0));
            } catch (const std::exception &error) {
                errors[thread] = error.what();
            }
        });
    }
    for (std::thread &thread : threads) { thread.join(); }
    for (size_t thread = 0; thread < threadCount; ++thread) {
        ASSERT_TRUE(errors[thread].empty())
            << "thread " << thread << ": " << errors[thread];
    }

    for (size_t thread = 0; thread < threadCount; ++thread) {
        const std::vector<float> &gold = (thread % 2) != 0 ? gold1 : gold0;
        EXPECT_TRUE(mismatch(gold, actual[thread], tolerance).empty())
            << "thread " << thread;
    }
}

void verifyColdConcurrentOperations(
    const std::vector<ArrayOperation> &operations,
    const std::vector<std::vector<float>> &gold, float tolerance) {
    ASSERT_GE(operations.size(), 2U);
    ASSERT_EQ(operations.size(), gold.size());
    StartGate gate(operations.size());
    std::vector<std::vector<float>> actual(operations.size());
    std::vector<std::string> errors(operations.size());
    std::vector<std::thread> threads;
    threads.reserve(operations.size());

    for (size_t thread = 0; thread < operations.size(); ++thread) {
        threads.emplace_back([&, thread]() {
            gate.wait();
            try {
                af::setDevice(0);
                actual[thread] = host(operations[thread]());
            } catch (const std::exception &error) {
                errors[thread] = error.what();
            }
        });
    }
    for (std::thread &thread : threads) { thread.join(); }

    af::setDevice(0);
    for (size_t thread = 0; thread < operations.size(); ++thread) {
        ASSERT_TRUE(errors[thread].empty())
            << "thread " << thread << ": " << errors[thread];
        EXPECT_TRUE(mismatch(gold[thread], actual[thread], tolerance).empty())
            << "thread " << thread;
    }
}

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

TEST(CUDNNAlgorithmCache, ConcurrentSameKeyForwardSelectionStaysAccurate) {
    af::setDevice(0);
    const af::dim4 signalDims(19, 17, 2, 1);
    const af::dim4 filterDims(3, 3, 2, 3);
    std::vector<float> signalValues(signalDims.elements());
    for (size_t index = 0; index < signalValues.size(); ++index) {
        signalValues[index] = static_cast<float>((index % 31) + 1) / 32.0f;
    }
    const std::vector<float> filter0Values(filterDims.elements(), 1.0f / 16.0f);
    const std::vector<float> filter1Values(filterDims.elements(),
                                           -1.0f / 32.0f);
    af::array signal(signalDims, signalValues.data());
    af::array filter0(filterDims, filter0Values.data());
    af::array filter1(filterDims, filter1Values.data());
    signal.eval();
    filter0.eval();
    filter1.eval();
    af::sync();
    const af::dim4 stride(1, 1);
    const af::dim4 padding(1, 1);
    const af::dim4 dilation(1, 1);

    Operation operation = [signal, filter0, filter1, stride, padding,
                           dilation](bool variant) {
        const af::array &filter = variant ? filter1 : filter0;
        return af::convolve2NN(signal, filter, stride, padding, dilation);
    };
    const std::vector<float> gold0 =
        hostConvolve2NN(signalValues, signalDims, filter0Values, filterDims,
                        stride, padding, dilation);
    const std::vector<float> gold1 =
        hostConvolve2NN(signalValues, signalDims, filter1Values, filterDims,
                        stride, padding, dilation);
    verifyColdConcurrentVariants(operation, gold0, gold1, 1.0e-4f);
}

TEST(CUDNNAlgorithmCache, ConcurrentDifferentKeyForwardSelectionsStayAccurate) {
    af::setDevice(0);
    const af::dim4 stride(1, 1);
    const af::dim4 padding(1, 1);
    const af::dim4 dilation(1, 1);
    std::vector<ArrayOperation> operations;
    std::vector<std::vector<float>> gold;
    for (dim_t variant = 0; variant < 3; ++variant) {
        const dim_t width  = 23 + variant;
        const dim_t height = 12 + variant;
        const af::dim4 signalDims(width, height, 2, 1);
        const af::dim4 filterDims(3, 3, 2, 3);
        const std::vector<float> signalValues(
            signalDims.elements(), static_cast<float>(variant + 1) / 32.0f);
        const std::vector<float> filterValues(
            filterDims.elements(), static_cast<float>(variant + 2) / 64.0f);
        af::array signal(signalDims, signalValues.data());
        af::array filter(filterDims, filterValues.data());
        signal.eval();
        filter.eval();
        gold.push_back(hostConvolve2NN(signalValues, signalDims, filterValues,
                                       filterDims, stride, padding, dilation));
        operations.emplace_back([signal, filter, stride, padding, dilation]() {
            return af::convolve2NN(signal, filter, stride, padding, dilation);
        });
    }
    af::sync();
    verifyColdConcurrentOperations(operations, gold, 1.0e-4f);
}

TEST(CUDNNAlgorithmCache,
     ConcurrentSameKeyBackwardFilterSelectionStaysAccurate) {
    af::setDevice(0);
    const af::dim4 signalDims(18, 15, 2, 1);
    const af::dim4 filterDims(3, 3, 2, 3);
    const af::dim4 gradientDims(18, 15, 3, 1);
    std::vector<float> signalValues(signalDims.elements());
    for (size_t index = 0; index < signalValues.size(); ++index) {
        signalValues[index] = static_cast<float>((index % 29) + 1) / 64.0f;
    }
    const std::vector<float> filterValues(filterDims.elements(), 1.0f / 16.0f);
    const std::vector<float> convolvedValues(gradientDims.elements(), 0.0f);
    const std::vector<float> gradient0Values(gradientDims.elements(),
                                             1.0f / 1024.0f);
    const std::vector<float> gradient1Values(gradientDims.elements(),
                                             -1.0f / 2048.0f);
    af::array signal(signalDims, signalValues.data());
    af::array filter(filterDims, filterValues.data());
    af::array convolvedOutput(gradientDims, convolvedValues.data());
    af::array gradient0(gradientDims, gradient0Values.data());
    af::array gradient1(gradientDims, gradient1Values.data());
    signal.eval();
    filter.eval();
    convolvedOutput.eval();
    gradient0.eval();
    gradient1.eval();
    af::sync();
    const af::dim4 stride(1, 1);
    const af::dim4 padding(1, 1);
    const af::dim4 dilation(1, 1);

    Operation operation = [signal, filter, convolvedOutput, gradient0,
                           gradient1, stride, padding, dilation](bool variant) {
        const af::array &incomingGradient = variant ? gradient1 : gradient0;
        return af::convolve2GradientNN(incomingGradient, signal, filter,
                                       convolvedOutput, stride, padding,
                                       dilation, AF_CONV_GRADIENT_FILTER);
    };
    const std::vector<float> gold0 = hostConvolve2FilterGradientNN(
        gradient0Values, gradientDims, signalValues, signalDims, filterDims,
        stride, padding, dilation);
    const std::vector<float> gold1 = hostConvolve2FilterGradientNN(
        gradient1Values, gradientDims, signalValues, signalDims, filterDims,
        stride, padding, dilation);
    verifyColdConcurrentVariants(operation, gold0, gold1, 4.0e-3f);
}

TEST(CUDNNAlgorithmCache,
     ConcurrentDifferentKeyBackwardFilterSelectionsStayAccurate) {
    af::setDevice(0);
    const af::dim4 stride(1, 1);
    const af::dim4 padding(1, 1);
    const af::dim4 dilation(1, 1);
    std::vector<ArrayOperation> operations;
    std::vector<std::vector<float>> gold;
    for (dim_t variant = 0; variant < 3; ++variant) {
        const dim_t width  = 27 + variant;
        const dim_t height = 14 + variant;
        const af::dim4 signalDims(width, height, 2, 1);
        const af::dim4 filterDims(3, 3, 2, 3);
        const af::dim4 gradientDims(width, height, 3, 1);
        const std::vector<float> signalValues(
            signalDims.elements(), static_cast<float>(variant + 1) / 64.0f);
        const std::vector<float> filterValues(filterDims.elements(),
                                              1.0f / 16.0f);
        const std::vector<float> convolvedValues(gradientDims.elements(), 0.0f);
        const std::vector<float> incomingGradientValues(
            gradientDims.elements(), static_cast<float>(variant + 1) / 1024.0f);
        af::array signal(signalDims, signalValues.data());
        af::array filter(filterDims, filterValues.data());
        af::array convolvedOutput(gradientDims, convolvedValues.data());
        af::array incomingGradient(gradientDims, incomingGradientValues.data());
        signal.eval();
        filter.eval();
        convolvedOutput.eval();
        incomingGradient.eval();
        gold.push_back(hostConvolve2FilterGradientNN(
            incomingGradientValues, gradientDims, signalValues, signalDims,
            filterDims, stride, padding, dilation));
        operations.emplace_back([signal, filter, convolvedOutput,
                                 incomingGradient, stride, padding,
                                 dilation]() {
            return af::convolve2GradientNN(incomingGradient, signal, filter,
                                           convolvedOutput, stride, padding,
                                           dilation, AF_CONV_GRADIENT_FILTER);
        });
    }
    af::sync();
    verifyColdConcurrentOperations(operations, gold, 4.0e-3f);
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
    verifyTransformBatchBeyondLegacyConstantMemory(false);
}

TEST(CUDASharedState, PerspectiveTransformSupportsMoreThanConstantMemoryBatch) {
    verifyTransformBatchBeyondLegacyConstantMemory(true);
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
