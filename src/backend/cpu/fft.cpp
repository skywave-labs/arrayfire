/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fft.hpp>

#include <Array.hpp>
#include <copy.hpp>
#include <err_cpu.hpp>
#include <fft_threads.hpp>
#include <fftw3.h>
#include <parallel.hpp>
#include <platform.hpp>
#include <types.hpp>
#include <af/dim4.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <utility>
#include <vector>

using af::dim4;
using std::array;

namespace arrayfire {
namespace cpu {

template<typename T>
struct fftw_transform;

#define TRANSFORM(PRE, TY)                                             \
    template<>                                                         \
    struct fftw_transform<TY> {                                        \
        typedef PRE##_plan plan_t;                                     \
        typedef PRE##_complex ctype_t;                                 \
                                                                       \
        template<typename... Args>                                     \
        plan_t create(Args... args) {                                  \
            return PRE##_plan_many_dft(args...);                       \
        }                                                              \
        void execute(plan_t plan, TY *in, TY *out) {                   \
            PRE##_execute_dft(plan, reinterpret_cast<ctype_t *>(in),   \
                              reinterpret_cast<ctype_t *>(out));       \
        }                                                              \
        void destroy(plan_t plan) { return PRE##_destroy_plan(plan); } \
    };

TRANSFORM(fftwf, cfloat)
TRANSFORM(fftw, cdouble)

template<typename To, typename Ti>
struct fftw_real_transform;

#define TRANSFORM_R2C(PRE, To, Ti)                                             \
    template<>                                                                 \
    struct fftw_real_transform<To, Ti> {                                       \
        typedef PRE##_plan plan_t;                                             \
        typedef PRE##_complex ctype_t;                                         \
                                                                               \
        template<typename... Args>                                             \
        plan_t create(Args... args) {                                          \
            return PRE##_plan_many_dft_r2c(args...);                           \
        }                                                                      \
        void execute(plan_t plan, Ti *in, To *out) {                           \
            PRE##_execute_dft_r2c(plan, in, reinterpret_cast<ctype_t *>(out)); \
        }                                                                      \
        void destroy(plan_t plan) { return PRE##_destroy_plan(plan); }         \
    };

#define TRANSFORM_C2R(PRE, To, Ti)                                             \
    template<>                                                                 \
    struct fftw_real_transform<To, Ti> {                                       \
        typedef PRE##_plan plan_t;                                             \
        typedef PRE##_complex ctype_t;                                         \
                                                                               \
        template<typename... Args>                                             \
        plan_t create(Args... args) {                                          \
            return PRE##_plan_many_dft_c2r(args...);                           \
        }                                                                      \
        void execute(plan_t plan, Ti *in, To *out) {                           \
            PRE##_execute_dft_c2r(plan, reinterpret_cast<ctype_t *>(in), out); \
        }                                                                      \
        void destroy(plan_t plan) { return PRE##_destroy_plan(plan); }         \
    };

TRANSFORM_R2C(fftwf, cfloat, float)
TRANSFORM_R2C(fftw, cdouble, double)
TRANSFORM_C2R(fftwf, float, cfloat)
TRANSFORM_C2R(fftw, double, cdouble)

inline array<int, AF_MAX_DIMS> computeDims(const int rank, const dim4 &idims) {
    array<int, AF_MAX_DIMS> retVal = {};
    for (int i = 0; i < rank; i++) { retVal[i] = idims[(rank - 1) - i]; }
    return retVal;
}

// FFTW only guarantees that its execute routines are thread-safe. Planning
// and destruction therefore take an exclusive lock, while new-array
// executions take a shared lock so the same cached plan can still be used by
// independent callers concurrently. Keep the mutex alive for the process
// lifetime because the CPU worker queue is also process-lifetime state.
std::shared_mutex &fftwMutex() {
    static auto *mutex = new std::shared_mutex();
    return *mutex;
}

namespace {

enum class FFTPrecision : std::uint8_t { SINGLE, DOUBLE };
enum class FFTTransformKind : std::uint8_t { C2C, R2C, C2R };

struct FFTPlanKey {
    FFTPrecision precision;
    FFTTransformKind kind;
    int rank;
    array<int, AF_MAX_DIMS> dims;
    int batch;
    int threadCount;
    array<int, AF_MAX_DIMS> inputEmbed;
    array<int, AF_MAX_DIMS> outputEmbed;
    int inputStride;
    int inputDistance;
    int outputStride;
    int outputDistance;
    int direction;
    unsigned flags;
    int inputAlignment;
    int outputAlignment;
    bool inPlace;

    bool operator==(const FFTPlanKey &other) const {
        return precision == other.precision && kind == other.kind &&
               rank == other.rank && dims == other.dims &&
               batch == other.batch && threadCount == other.threadCount &&
               inputEmbed == other.inputEmbed &&
               outputEmbed == other.outputEmbed &&
               inputStride == other.inputStride &&
               inputDistance == other.inputDistance &&
               outputStride == other.outputStride &&
               outputDistance == other.outputDistance &&
               direction == other.direction && flags == other.flags &&
               inputAlignment == other.inputAlignment &&
               outputAlignment == other.outputAlignment &&
               inPlace == other.inPlace;
    }
};

class FFTPlan {
   public:
    using DestroyFunction = void (*)(void *);

    FFTPlan(void *plan, DestroyFunction destroy)
        : plan_(plan), destroy_(destroy) {}

    ~FFTPlan() {
        if (plan_ != nullptr) {
            std::unique_lock<std::shared_mutex> lock(fftwMutex());
            destroy_(plan_);
        }
    }

    template<typename Plan>
    Plan get() const {
        return reinterpret_cast<Plan>(plan_);
    }

   private:
    void *plan_;
    DestroyFunction destroy_;
};

class FFTPlanCache {
    using PlanPtr = std::shared_ptr<FFTPlan>;

    struct Entry {
        FFTPlanKey key;
        PlanPtr plan;
    };

   public:
    template<typename Creator>
    PlanPtr getOrCreate(const FFTPlanKey &key, Creator &&create,
                        FFTPlan::DestroyFunction destroy) {
        PlanPtr discarded;
        std::unique_lock<std::mutex> cacheLock(mutex_);

        const auto match =
            std::find_if(cache_.begin(), cache_.end(),
                         [&](const Entry &entry) { return entry.key == key; });
        if (match != cache_.end()) {
            PlanPtr plan = match->plan;
            cache_.splice(cache_.begin(), cache_, match);
            return plan;
        }

        void *rawPlan = nullptr;
        {
            std::unique_lock<std::shared_mutex> fftwLock(fftwMutex());
            rawPlan = create();
        }
        if (rawPlan == nullptr) {
            AF_ERROR("FFTW plan creation failed", AF_ERR_INTERNAL);
        }

        PlanPtr plan;
        try {
            plan = std::make_shared<FFTPlan>(rawPlan, destroy);
        } catch (...) {
            std::unique_lock<std::shared_mutex> fftwLock(fftwMutex());
            destroy(rawPlan);
            throw;
        }
        if (maxSize_ > 0) {
            cache_.push_front(Entry{key, plan});
            if (cache_.size() > maxSize_) {
                discarded = std::move(cache_.back().plan);
                cache_.pop_back();
            }
        }

        cacheLock.unlock();
        discarded.reset();
        return plan;
    }

    void setMaxSize(size_t size) {
        std::vector<PlanPtr> discarded;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            maxSize_ = size;
            while (cache_.size() > maxSize_) {
                discarded.emplace_back(std::move(cache_.back().plan));
                cache_.pop_back();
            }
        }
    }

   private:
    size_t maxSize_ = 5;
    std::list<Entry> cache_;
    std::mutex mutex_;
};

FFTPlanCache &fftPlanCache() {
    // The CPU worker queue intentionally has process lifetime, so the plans it
    // may still reference must not be destroyed during static teardown.
    static auto *cache = new FFTPlanCache();
    return *cache;
}

template<typename T>
constexpr FFTPrecision fftPrecision() {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, cfloat>) {
        return FFTPrecision::SINGLE;
    } else {
        return FFTPrecision::DOUBLE;
    }
}

#ifdef AF_WITH_FFTW_THREADS

bool initializeFFTWThreads(const FFTPrecision precision) {
    if (getParallelThreadCount() <= 1) { return false; }

    if (precision == FFTPrecision::SINGLE) {
        static const bool initialized = fftwf_init_threads() != 0;
        return initialized;
    } else {
        static const bool initialized = fftw_init_threads() != 0;
        return initialized;
    }
}

int fftPlannerThreadCount(const FFTPrecision precision) {
    return precision == FFTPrecision::SINGLE ? fftwf_planner_nthreads()
                                             : fftw_planner_nthreads();
}

void setFFTPlannerThreadCount(const FFTPrecision precision,
                              const int threadCount) {
    if (precision == FFTPrecision::SINGLE) {
        fftwf_plan_with_nthreads(threadCount);
    } else {
        fftw_plan_with_nthreads(threadCount);
    }
}

int selectFFTThreadCount(const int rank, const array<int, AF_MAX_DIMS> &dims,
                         const int batch, const FFTTransformKind kind) {
    size_t logicalPoints = static_cast<size_t>(std::max(1, batch));
    for (int i = 0; i < rank; ++i) {
        const size_t dimension = static_cast<size_t>(std::max(1, dims[i]));
        if (logicalPoints > std::numeric_limits<size_t>::max() / dimension) {
            logicalPoints = std::numeric_limits<size_t>::max();
            break;
        }
        logicalPoints *= dimension;
    }

    // Below the existing large-transform gate, FFTW's C2R plans regress for
    // some 1D and batched factorizations. C2C and R2C use the gradual ramp.
    const bool useGradualRamp =
        kind == FFTTransformKind::C2C || kind == FFTTransformKind::R2C;
    const detail::FFTPlanThreadPolicy policy =
        useGradualRamp ? detail::FFTPlanThreadPolicy::GRADUAL
                       : detail::FFTPlanThreadPolicy::LARGE_ONLY;
    return static_cast<int>(detail::selectFFTPlanThreadCount(
        logicalPoints, getParallelThreadCount(), policy));
}

class FFTPlannerThreadGuard {
   public:
    FFTPlannerThreadGuard(const FFTPrecision precision, const int threadCount)
        : precision_(precision), active_(initializeFFTWThreads(precision)) {
        if (active_) {
            previousThreadCount_ = fftPlannerThreadCount(precision_);
            setFFTPlannerThreadCount(precision_, threadCount);
        }
    }

    ~FFTPlannerThreadGuard() {
        if (active_) {
            setFFTPlannerThreadCount(precision_, previousThreadCount_);
        }
    }

   private:
    FFTPrecision precision_;
    int previousThreadCount_ = 1;
    bool active_;
};

#endif

template<typename T>
int fftAlignment(const T *ptr) {
#ifdef USE_MKL
    // MKL's FFTW wrapper builds pointer-independent DFTI descriptors. Older
    // supported versions do not export fftw_alignment_of, so all buffers use
    // one alignment class and execution remains serialized below.
    UNUSED(ptr);
    return 0;
#else
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, cfloat>) {
        auto *scalar = reinterpret_cast<const float *>(ptr);
        return fftwf_alignment_of(const_cast<float *>(scalar));
    } else {
        auto *scalar = reinterpret_cast<const double *>(ptr);
        return fftw_alignment_of(const_cast<double *>(scalar));
    }
#endif
}

template<typename Ti, typename To>
FFTPlanKey makePlanKey(const FFTTransformKind kind, const int rank,
                       const array<int, AF_MAX_DIMS> &dims, const int batch,
                       const array<int, AF_MAX_DIMS> &inputEmbed,
                       const int inputStride, const int inputDistance,
                       const array<int, AF_MAX_DIMS> &outputEmbed,
                       const int outputStride, const int outputDistance,
                       const int direction, const unsigned flags,
                       const Ti *input, const To *output) {
    std::unique_lock<std::shared_mutex> fftwLock(fftwMutex());
    const FFTPrecision precision = fftPrecision<Ti>();
    int threadCount              = 1;
#ifdef AF_WITH_FFTW_THREADS
    // Initialize before the first FFTW operation of each precision. A failed
    // initialization can reset FFTW state, so it must not happen after plans
    // have entered the process-lifetime cache.
    if (initializeFFTWThreads(precision)) {
        threadCount = selectFFTThreadCount(rank, dims, batch, kind);
    }
#endif
    const int inputAlignment  = fftAlignment(input);
    const int outputAlignment = fftAlignment(output);
    return FFTPlanKey{
        precision,
        kind,
        rank,
        dims,
        batch,
        threadCount,
        inputEmbed,
        outputEmbed,
        inputStride,
        inputDistance,
        outputStride,
        outputDistance,
        direction,
        flags,
        inputAlignment,
        outputAlignment,
        static_cast<const void *>(input) == static_cast<const void *>(output)};
}

template<typename Transform>
void destroyFFTPlan(void *plan) {
    Transform transform;
    transform.destroy(reinterpret_cast<typename Transform::plan_t>(plan));
}

template<typename Transform, typename Creator>
std::shared_ptr<FFTPlan> getFFTPlan(const FFTPlanKey &key, Creator &&create) {
    return fftPlanCache().getOrCreate(
        key,
        [&]() {
#ifdef AF_WITH_FFTW_THREADS
            // FFTW's planner count is process-global and sticky. Install the
            // plan-specific value only while the exclusive planner lock is
            // held, then restore the host application's prior setting.
            FFTPlannerThreadGuard threadGuard(key.precision, key.threadCount);
#endif
            return reinterpret_cast<void *>(std::forward<Creator>(create)());
        },
        &destroyFFTPlan<Transform>);
}

template<typename Transform, typename... Args>
void executeFFTPlan(const std::shared_ptr<FFTPlan> &plan, Transform &transform,
                    Args... args) {
#ifdef USE_MKL
    // MKL's FFTW compatibility layer does not document concurrent execution
    // of the same descriptor, so keep its use conservative.
    std::unique_lock<std::shared_mutex> lock(fftwMutex());
#else
    std::shared_lock<std::shared_mutex> lock(fftwMutex());
#endif
    transform.execute(plan->template get<typename Transform::plan_t>(),
                      args...);
}

}  // namespace

void setFFTPlanCacheSize(size_t numPlans) {
    fftPlanCache().setMaxSize(numPlans);
}

template<typename T>
void fft_inplace(Param<T> in, const af::dim4 iDataDims, const int rank,
                 const bool direction) {
    const af::dim4 idims = in.dims();

    auto t_dims   = computeDims(rank, idims);
    auto in_embed = computeDims(rank, iDataDims);

    const af::dim4 istrides = in.strides();

    using ctype_t = typename fftw_transform<T>::ctype_t;
    fftw_transform<T> transform;

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= idims[i]; }

    const int fftwDirection  = direction ? FFTW_FORWARD : FFTW_BACKWARD;
    constexpr unsigned flags = FFTW_ESTIMATE;  // NOLINT(hicpp-signed-bitwise)
    const auto key = makePlanKey(FFTTransformKind::C2C, rank, t_dims, batch,
                                 in_embed, static_cast<int>(istrides[0]),
                                 static_cast<int>(istrides[rank]), in_embed,
                                 static_cast<int>(istrides[0]),
                                 static_cast<int>(istrides[rank]),
                                 fftwDirection, flags, in.get(), in.get());

    auto plan = getFFTPlan<fftw_transform<T>>(key, [&]() {
        return transform.create(
            rank, t_dims.data(), batch, reinterpret_cast<ctype_t *>(in.get()),
            in_embed.data(), static_cast<int>(istrides[0]),
            static_cast<int>(istrides[rank]),
            reinterpret_cast<ctype_t *>(in.get()), in_embed.data(),
            static_cast<int>(istrides[0]), static_cast<int>(istrides[rank]),
            fftwDirection, flags);
    });

    executeFFTPlan(plan, transform, in.get(), in.get());
}

template<typename T>
void fft_inplace(Array<T> &in, const int rank, const bool direction) {
    auto func = [=](Param<T> in, const af::dim4 iDataDims) {
        fft_inplace(in, iDataDims, rank, direction);
    };
    getQueue().enqueue(func, in, in.getDataDims());
}

template<typename Tc, typename Tr>
Array<Tc> fft_r2c(const Array<Tr> &in, const int rank) {
    dim4 odims    = in.dims();
    odims[0]      = odims[0] / 2 + 1;
    Array<Tc> out = createEmptyArray<Tc>(odims);

    auto func = [=](Param<Tc> out, const af::dim4 oDataDims, CParam<Tr> in,
                    const af::dim4 iDataDims) {
        af::dim4 idims = in.dims();

        auto t_dims    = computeDims(rank, idims);
        auto in_embed  = computeDims(rank, iDataDims);
        auto out_embed = computeDims(rank, oDataDims);

        const af::dim4 istrides = in.strides();
        const af::dim4 ostrides = out.strides();

        using ctype_t = typename fftw_real_transform<Tc, Tr>::ctype_t;
        fftw_real_transform<Tc, Tr> transform;

        int batch = 1;
        for (int i = rank; i < 4; i++) { batch *= idims[i]; }

        constexpr unsigned flags =
            FFTW_ESTIMATE;  // NOLINT(hicpp-signed-bitwise)
        const auto key = makePlanKey(
            FFTTransformKind::R2C, rank, t_dims, batch, in_embed,
            static_cast<int>(istrides[0]), static_cast<int>(istrides[rank]),
            out_embed, static_cast<int>(ostrides[0]),
            static_cast<int>(ostrides[rank]), 0, flags, in.get(), out.get());

        auto plan = getFFTPlan<fftw_real_transform<Tc, Tr>>(key, [&]() {
            return transform.create(
                rank, t_dims.data(), batch, const_cast<Tr *>(in.get()),
                in_embed.data(), static_cast<int>(istrides[0]),
                static_cast<int>(istrides[rank]),
                reinterpret_cast<ctype_t *>(out.get()), out_embed.data(),
                static_cast<int>(ostrides[0]), static_cast<int>(ostrides[rank]),
                flags);
        });

        executeFFTPlan(plan, transform, const_cast<Tr *>(in.get()), out.get());
    };

    getQueue().enqueue(func, out, out.getDataDims(), in, in.getDataDims());

    return out;
}

template<typename Tr, typename Tc>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims, const int rank) {
    Array<Tr> out = createEmptyArray<Tr>(odims);

    auto func = [=](Param<Tr> out, const af::dim4 oDataDims, CParam<Tc> in,
                    const af::dim4 iDataDims, const af::dim4 odims) {
        auto t_dims    = computeDims(rank, odims);
        auto in_embed  = computeDims(rank, iDataDims);
        auto out_embed = computeDims(rank, oDataDims);

        const af::dim4 istrides = in.strides();
        const af::dim4 ostrides = out.strides();

        using ctype_t = typename fftw_real_transform<Tr, Tc>::ctype_t;
        fftw_real_transform<Tr, Tc> transform;

        int batch = 1;
        for (int i = rank; i < 4; i++) { batch *= odims[i]; }

        // By default, fftw estimate flag is sufficient for most transforms.
        // However, complex to real transforms modify the input data memory
        // while performing the transformation. To avoid that, we need to pass
        // FFTW_PRESERVE_INPUT also. This flag however only works for 1D
        // transforms and for higher level transformations, a copy of input
        // data is passed onto the upstream FFTW calls.
        unsigned int flags = FFTW_ESTIMATE;  // NOLINT(hicpp-signed-bitwise)
        if (rank == 1) {
            flags |= FFTW_PRESERVE_INPUT;  // NOLINT(hicpp-signed-bitwise)
        }

        const auto key = makePlanKey(
            FFTTransformKind::C2R, rank, t_dims, batch, in_embed,
            static_cast<int>(istrides[0]), static_cast<int>(istrides[rank]),
            out_embed, static_cast<int>(ostrides[0]),
            static_cast<int>(ostrides[rank]), 0, flags, in.get(), out.get());

        auto plan = getFFTPlan<fftw_real_transform<Tr, Tc>>(key, [&]() {
            return transform.create(
                rank, t_dims.data(), batch,
                reinterpret_cast<ctype_t *>(const_cast<Tc *>(in.get())),
                in_embed.data(), static_cast<int>(istrides[0]),
                static_cast<int>(istrides[rank]), out.get(), out_embed.data(),
                static_cast<int>(ostrides[0]), static_cast<int>(ostrides[rank]),
                flags);
        });

        executeFFTPlan(plan, transform, const_cast<Tc *>(in.get()), out.get());
    };

#ifdef USE_MKL
    getQueue().enqueue(func, out, out.getDataDims(), in, in.getDataDims(),
                       odims);
#else
    if (rank > 1 || odims.ndims() > 1) {
        // FFTW does not have a input preserving algorithm for multidimensional
        // c2r FFTs
        Array<Tc> in_ = copyArray<Tc>(in);
        getQueue().enqueue(func, out, out.getDataDims(), in_, in_.getDataDims(),
                           odims);
    } else {
        getQueue().enqueue(func, out, out.getDataDims(), in, in.getDataDims(),
                           odims);
    }
#endif

    return out;
}

#define INSTANTIATE(T)                                                \
    template void fft_inplace<T>(Param<T>, const af::dim4, const int, \
                                 const bool);                         \
    template void fft_inplace<T>(Array<T> &, const int, const bool);

INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

#define INSTANTIATE_REAL(Tr, Tc)                                             \
    template Array<Tc> fft_r2c<Tc, Tr>(const Array<Tr> &, const int);        \
    template Array<Tr> fft_c2r<Tr, Tc>(const Array<Tc> &in, const dim4 &odi, \
                                       const int);

INSTANTIATE_REAL(float, cfloat)
INSTANTIATE_REAL(double, cdouble)

}  // namespace cpu
}  // namespace arrayfire
