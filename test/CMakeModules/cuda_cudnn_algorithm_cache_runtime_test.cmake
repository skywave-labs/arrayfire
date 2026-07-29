# Copyright (c) 2026, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED TEST_EXECUTABLE OR TEST_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR
            "TEST_EXECUTABLE must name the CUDA shared-state test binary")
endif()
if(NOT EXISTS "${TEST_EXECUTABLE}")
    message(FATAL_ERROR
            "CUDA shared-state test binary does not exist: ${TEST_EXECUTABLE}")
endif()

set(
    cudnn_test_filter
    "CUDNNAlgorithmCache.ConcurrentSameKeyForwardSelectionStaysAccurate:CUDNNAlgorithmCache.ConcurrentSameKeyBackwardFilterSelectionStaysAccurate"
)
execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env "AF_TRACE=platform" "${TEST_EXECUTABLE}"
        "--gtest_filter=${cudnn_test_filter}"
    RESULT_VARIABLE test_result
    OUTPUT_VARIABLE test_output
    ERROR_VARIABLE test_error
    TIMEOUT 300)

set(test_log "${test_output}\n${test_error}")
if(test_log MATCHES "WARNING: Unable to load cuDNN:")
    message(STATUS "AF_TEST_SKIP_CUDNN_RUNTIME_UNAVAILABLE")
    return()
endif()

if(NOT test_result EQUAL 0)
    message(
        FATAL_ERROR
            "cuDNN algorithm-cache runtime test failed (${test_result})\nstdout:\n${test_output}\nstderr:\n${test_error}"
    )
endif()

string(REGEX MATCHALL
       "cuDNN forward algorithm cache miss on device "
       forward_cache_misses
       "${test_log}")
list(LENGTH forward_cache_misses forward_cache_miss_count)
if(NOT forward_cache_miss_count EQUAL 1)
    message(
        FATAL_ERROR
            "Expected exactly one cuDNN forward cache miss, found ${forward_cache_miss_count}\nstdout:\n${test_output}\nstderr:\n${test_error}"
    )
endif()
string(REGEX MATCHALL
       "cuDNN backward-filter algorithm cache miss on device "
       backward_cache_misses
       "${test_log}")
list(LENGTH backward_cache_misses backward_cache_miss_count)
if(NOT backward_cache_miss_count EQUAL 1)
    message(
        FATAL_ERROR
            "Expected exactly one cuDNN backward-filter cache miss, found ${backward_cache_miss_count}\nstdout:\n${test_output}\nstderr:\n${test_error}"
    )
endif()
