/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/dim4.hpp>

#include <array>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace arrayfire {
namespace cuda {
namespace detail {

struct CudnnConvolutionAlgorithmKey {
    std::array<dim_t, 4> input{};
    std::array<dim_t, 4> filter{};
    std::array<dim_t, 4> output{};
    std::array<dim_t, 2> stride{};
    std::array<dim_t, 2> padding{};
    std::array<dim_t, 2> dilation{};
    int dataType = 0;

    bool operator==(const CudnnConvolutionAlgorithmKey &other) const {
        return input == other.input && filter == other.filter &&
               output == other.output && stride == other.stride &&
               padding == other.padding && dilation == other.dilation &&
               dataType == other.dataType;
    }
};

// These descriptors are always packed NCHW, use CUDNN_CONVOLUTION with one
// group and default math, and select the fastest algorithm without an explicit
// workspace limit. If any of those policies become configurable, add them to
// this structural key.
struct CudnnConvolutionAlgorithmKeyHash {
    std::size_t operator()(const CudnnConvolutionAlgorithmKey &key) const {
        std::size_t seed = 0;
        for (dim_t value : key.input) { combine(&seed, value); }
        for (dim_t value : key.filter) { combine(&seed, value); }
        for (dim_t value : key.output) { combine(&seed, value); }
        for (dim_t value : key.stride) { combine(&seed, value); }
        for (dim_t value : key.padding) { combine(&seed, value); }
        for (dim_t value : key.dilation) { combine(&seed, value); }
        combine(&seed, key.dataType);
        return seed;
    }

   private:
    template<typename T>
    static void combine(std::size_t *seed, const T &value) {
        *seed ^=
            std::hash<T>()(value) + 0x9e3779b9U + (*seed << 6U) + (*seed >> 2U);
    }
};

inline CudnnConvolutionAlgorithmKey makeCudnnConvolutionAlgorithmKey(
    const af::dim4 &input, const af::dim4 &filter, const af::dim4 &output,
    const af::dim4 &stride, const af::dim4 &padding, const af::dim4 &dilation,
    int dataType) {
    CudnnConvolutionAlgorithmKey key;
    for (int i = 0; i < 4; ++i) {
        key.input[i]  = input[i];
        key.filter[i] = filter[i];
        key.output[i] = output[i];
    }
    for (int i = 0; i < 2; ++i) {
        key.stride[i]   = stride[i];
        key.padding[i]  = padding[i];
        key.dilation[i] = dilation[i];
    }
    key.dataType = dataType;
    return key;
}

// A small FIFO cache is used instead of an LRU cache so reads do not mutate
// cache state. Algorithm selection is expensive while cached values are tiny,
// so a bounded number of shape-specific entries gives predictable memory use.
template<typename Key, typename Value, typename Hash, std::size_t Capacity>
class CudnnAlgorithmCache {
   public:
    enum class Lookup { Hit, Miss, Wait };

    struct Result {
        Value value;
        Lookup lookup;
    };

    bool find(const Key &key, Value *value) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto cached = entries_.find(key);
        if (cached == entries_.end()) { return false; }

        *value = cached->second;
        return true;
    }

    template<typename Selector>
    Result getOrCreate(const Key &key, Selector selector) {
        std::shared_ptr<Pending> pending;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const auto cached = entries_.find(key);
            if (cached != entries_.end()) {
                return {cached->second, Lookup::Hit};
            }

            const auto selecting = pending_.find(key);
            if (selecting != pending_.end()) {
                pending = selecting->second;
                pending->condition.wait(lock,
                                        [&pending] { return pending->ready; });
                if (pending->error) { std::rethrow_exception(pending->error); }
                return {pending->value, Lookup::Wait};
            }

            pending = std::make_shared<Pending>();
            pending_.emplace(key, pending);
        }

        try {
            const Value selected = selector();
            Value value;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                value          = insertOrGetUnlocked(key, selected);
                pending->value = value;
                pending->ready = true;
                pending_.erase(key);
            }
            pending->condition.notify_all();
            return {value, Lookup::Miss};
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                pending->error = std::current_exception();
                pending->ready = true;
                pending_.erase(key);
            }
            pending->condition.notify_all();
            throw;
        }
    }

    Value insertOrGet(const Key &key, const Value &value) {
        std::lock_guard<std::mutex> lock(mutex_);
        return insertOrGetUnlocked(key, value);
    }

    void erase(const Key &key) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto cached = entries_.find(key);
        if (cached == entries_.end()) { return; }

        for (auto entry = insertionOrder_.begin();
             entry != insertionOrder_.end(); ++entry) {
            if (*entry == key) {
                insertionOrder_.erase(entry);
                break;
            }
        }
        entries_.erase(cached);
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

    static constexpr std::size_t capacity() { return Capacity; }

   private:
    struct Pending {
        Value value{};
        std::exception_ptr error;
        bool ready = false;
        std::condition_variable condition;
    };

    Value insertOrGetUnlocked(const Key &key, const Value &value) {
        const auto cached = entries_.find(key);
        if (cached != entries_.end()) { return cached->second; }
        if (Capacity == 0) { return value; }

        if (entries_.size() >= Capacity) {
            entries_.erase(insertionOrder_.front());
            insertionOrder_.pop_front();
        }

        insertionOrder_.push_back(key);
        try {
            entries_.emplace(key, value);
        } catch (...) {
            insertionOrder_.pop_back();
            throw;
        }
        return value;
    }

    mutable std::mutex mutex_;
    std::unordered_map<Key, Value, Hash> entries_;
    std::deque<Key> insertionOrder_;
    std::unordered_map<Key, std::shared_ptr<Pending>, Hash> pending_;
};

}  // namespace detail
}  // namespace cuda
}  // namespace arrayfire
