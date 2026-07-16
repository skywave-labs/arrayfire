/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cpu_features.hpp>

#include <cstdint>

#if defined(AF_WITH_CPUID) &&                                        \
    (defined(__x86_64__) || defined(__amd64__) || defined(_M_X64) || \
     defined(__i386__) || defined(_M_IX86))
#define AF_CPU_X86_FEATURE_DETECTION
#endif

#if defined(AF_CPU_X86_FEATURE_DETECTION) && defined(_MSC_VER)
#include <intrin.h>
#elif defined(AF_CPU_X86_FEATURE_DETECTION) && \
    (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

namespace arrayfire {
namespace cpu {
namespace detail {
namespace {

#if defined(AF_CPU_X86_FEATURE_DETECTION) && \
    (defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__))

constexpr std::uint32_t OSXSAVE_BIT        = 1U << 27U;
constexpr std::uint32_t AVX_BIT            = 1U << 28U;
constexpr std::uint32_t F16C_BIT           = 1U << 29U;
constexpr std::uint32_t AVX2_BIT           = 1U << 5U;
constexpr std::uint64_t XMM_YMM_STATE_MASK = 0x6U;

#if defined(_MSC_VER)

bool detectAVX2Support() noexcept {
    int registers[4] = {};
    __cpuidex(registers, 0, 0);
    const int max_leaf = registers[0];
    if (max_leaf < 1) { return false; }

    __cpuidex(registers, 1, 0);
    const auto feature_ecx = static_cast<std::uint32_t>(registers[2]);
    if ((feature_ecx & (OSXSAVE_BIT | AVX_BIT)) != (OSXSAVE_BIT | AVX_BIT)) {
        return false;
    }

    const std::uint64_t enabled_state = _xgetbv(0);
    if ((enabled_state & XMM_YMM_STATE_MASK) != XMM_YMM_STATE_MASK) {
        return false;
    }

    if (max_leaf < 7) { return false; }
    __cpuidex(registers, 7, 0);
    return (static_cast<std::uint32_t>(registers[1]) & AVX2_BIT) != 0;
}

bool detectF16CSupport() noexcept {
    int registers[4] = {};
    __cpuidex(registers, 0, 0);
    if (registers[0] < 1) { return false; }

    __cpuidex(registers, 1, 0);
    const auto feature_ecx = static_cast<std::uint32_t>(registers[2]);
    const auto required    = OSXSAVE_BIT | AVX_BIT | F16C_BIT;
    if ((feature_ecx & required) != required) { return false; }

    const std::uint64_t enabled_state = _xgetbv(0);
    return (enabled_state & XMM_YMM_STATE_MASK) == XMM_YMM_STATE_MASK;
}

#elif defined(__GNUC__) || defined(__clang__)

std::uint64_t readXCR0() noexcept {
    std::uint32_t lower = 0;
    std::uint32_t upper = 0;
    asm volatile("xgetbv" : "=a"(lower), "=d"(upper) : "c"(0));
    return (static_cast<std::uint64_t>(upper) << 32U) | lower;
}

bool detectAVX2Support() noexcept {
    const unsigned int max_leaf = __get_cpuid_max(0, nullptr);
    if (max_leaf < 1) { return false; }

    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    if ((ecx & (OSXSAVE_BIT | AVX_BIT)) != (OSXSAVE_BIT | AVX_BIT)) {
        return false;
    }

    if ((readXCR0() & XMM_YMM_STATE_MASK) != XMM_YMM_STATE_MASK) {
        return false;
    }

    if (max_leaf < 7) { return false; }
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & AVX2_BIT) != 0;
}

bool detectF16CSupport() noexcept {
    if (__get_cpuid_max(0, nullptr) < 1) { return false; }

    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    const unsigned int required = OSXSAVE_BIT | AVX_BIT | F16C_BIT;
    if ((ecx & required) != required) { return false; }

    return (readXCR0() & XMM_YMM_STATE_MASK) == XMM_YMM_STATE_MASK;
}

#endif

#else

bool detectAVX2Support() noexcept { return false; }
bool detectF16CSupport() noexcept { return false; }

#endif

}  // namespace

bool isAVX2Supported() noexcept {
    static const bool supported = detectAVX2Support();
    return supported;
}

bool isF16CSupported() noexcept {
    static const bool supported = detectF16CSupport();
    return supported;
}

}  // namespace detail
}  // namespace cpu
}  // namespace arrayfire
