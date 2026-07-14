/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <parallel.hpp>

#include <common/util.hpp>

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using arrayfire::common::getEnvVar;
using std::function;
using std::lock_guard;
using std::mutex;
using std::queue;
using std::shared_ptr;
using std::string;
using std::thread;
using std::unique_lock;
using std::vector;

namespace arrayfire {
namespace cpu {
namespace {

thread_local bool is_pool_worker = false;

size_t configuredThreadCount() {
    static const size_t thread_count = []() {
        const size_t hardware_threads =
            std::max<size_t>(1, thread::hardware_concurrency());
        const string env_var = getEnvVar("AF_CPU_NUM_THREADS");
        if (env_var.empty()) { return hardware_threads; }

        try {
            size_t parsed_chars       = 0;
            const size_t user_threads = std::stoul(env_var, &parsed_chars);
            if (parsed_chars == env_var.size() && user_threads > 0) {
                return std::min(user_threads, hardware_threads);
            }
        } catch (const std::exception &) {}
        return hardware_threads;
    }();
    return thread_count;
}

class ThreadPool {
   public:
    explicit ThreadPool(size_t thread_count) {
        workers.reserve(thread_count);
        for (size_t i = 0; i < thread_count; ++i) {
            try {
                workers.emplace_back([this]() { workerLoop(); });
            } catch (const std::exception &) {
                // Continue with the workers that the system allowed us to
                // create. The calling thread always remains available.
                break;
            }
        }
    }

    ThreadPool(const ThreadPool &)            = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    ~ThreadPool() {
        {
            lock_guard<mutex> lock(queue_mutex);
            stopping = true;
        }
        work_available.notify_all();
        for (thread &worker : workers) { worker.join(); }
    }

    size_t size() const { return workers.size(); }

    void enqueue(function<void()> function) {
        {
            lock_guard<mutex> lock(queue_mutex);
            work.push(std::move(function));
        }
        work_available.notify_one();
    }

   private:
    void workerLoop() {
        is_pool_worker = true;
        while (true) {
            function<void()> function;
            {
                unique_lock<mutex> lock(queue_mutex);
                work_available.wait(
                    lock, [this]() { return stopping || !work.empty(); });
                if (stopping && work.empty()) { return; }
                function = std::move(work.front());
                work.pop();
            }
            function();
        }
    }

    vector<thread> workers;
    queue<function<void()>> work;
    mutex queue_mutex;
    std::condition_variable work_available;
    bool stopping{false};
};

ThreadPool &getThreadPool() {
    // DeviceManager intentionally lives until process exit. Keep its compute
    // workers alive for the same lifetime so an asynchronous CPU queue cannot
    // race static destruction.
    static ThreadPool *thread_pool =
        new ThreadPool(configuredThreadCount() - 1);
    return *thread_pool;
}

struct ParallelState {
    explicit ParallelState(function<void(size_t)> function)
        : function_(std::move(function)) {}

    void addTask() {
        lock_guard<mutex> lock(mutex_);
        ++remaining;
    }

    void run(size_t task) {
        try {
            function_(task);
        } catch (...) { recordException(); }
    }

    void recordException() {
        lock_guard<mutex> lock(mutex_);
        if (!exception) { exception = std::current_exception(); }
    }

    void complete() {
        lock_guard<mutex> lock(mutex_);
        if (--remaining == 0) { complete_condition.notify_one(); }
    }

    void wait() {
        unique_lock<mutex> lock(mutex_);
        complete_condition.wait(lock, [this]() { return remaining == 0; });
    }

    function<void(size_t)> function_;
    mutex mutex_;
    std::condition_variable complete_condition;
    size_t remaining{0};
    std::exception_ptr exception;
};

}  // namespace

size_t getParallelThreadCount() { return configuredThreadCount(); }

void parallelFor(size_t task_count, std::function<void(size_t)> function) {
    if (task_count == 0) { return; }
    if (task_count == 1 || is_pool_worker) {
        for (size_t task = 0; task < task_count; ++task) { function(task); }
        return;
    }

    ThreadPool &thread_pool = getThreadPool();
    if (thread_pool.size() == 0) {
        for (size_t task = 0; task < task_count; ++task) { function(task); }
        return;
    }

    auto state = std::make_shared<ParallelState>(std::move(function));
    size_t first_local_task = task_count;
    for (size_t task = 1; task < task_count; ++task) {
        state->addTask();
        try {
            thread_pool.enqueue([task, state]() {
                state->run(task);
                state->complete();
            });
        } catch (...) {
            state->complete();
            first_local_task = task;
            break;
        }
    }

    state->run(0);
    for (size_t task = first_local_task; task < task_count; ++task) {
        state->run(task);
    }

    state->wait();
    if (state->exception) { std::rethrow_exception(state->exception); }
}

void parallelForRange(size_t item_count, size_t min_items_per_task,
                      std::function<void(size_t begin, size_t end)> function) {
    if (item_count == 0) { return; }

    const size_t grain        = std::max<size_t>(1, min_items_per_task);
    const size_t useful_tasks = std::max<size_t>(1, item_count / grain);
    const size_t task_count =
        std::max<size_t>(1, std::min(getParallelThreadCount(), useful_tasks));

    parallelFor(task_count, [&](size_t task) {
        const size_t items_per_task = item_count / task_count;
        const size_t remainder      = item_count % task_count;
        const size_t begin = task * items_per_task + std::min(task, remainder);
        const size_t end = begin + items_per_task + (task < remainder ? 1 : 0);
        function(begin, end);
    });
}

}  // namespace cpu
}  // namespace arrayfire
