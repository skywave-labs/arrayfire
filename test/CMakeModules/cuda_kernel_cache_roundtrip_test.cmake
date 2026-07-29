# Copyright (c) 2026, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED HELPER_EXECUTABLE OR HELPER_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "HELPER_EXECUTABLE must name the cache test helper")
endif()
if(NOT EXISTS "${HELPER_EXECUTABLE}")
    message(FATAL_ERROR "Cache test helper does not exist: ${HELPER_EXECUTABLE}")
endif()

if(NOT DEFINED ROUNDTRIP_ROOT OR ROUNDTRIP_ROOT STREQUAL "")
    set(ROUNDTRIP_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
endif()
if(NOT DEFINED CUDA_DEVICE)
    set(CUDA_DEVICE 0)
endif()

get_filename_component(ROUNDTRIP_ROOT "${ROUNDTRIP_ROOT}" ABSOLUTE)
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef roundtrip_suffix)
set(cache_directory
    "${ROUNDTRIP_ROOT}/cuda-kernel-cache-roundtrip-${roundtrip_suffix}")
get_filename_component(cache_directory "${cache_directory}" ABSOLUTE)

# Keep recursive cleanup constrained to the uniquely named directory below the
# caller-provided test root.
string(FIND "${cache_directory}"
            "${ROUNDTRIP_ROOT}/cuda-kernel-cache-roundtrip-" cache_prefix)
if(NOT cache_prefix EQUAL 0)
    message(FATAL_ERROR "Refusing unsafe cache directory: ${cache_directory}")
endif()

function(fail_after_cleanup description output error_output)
    file(REMOVE_RECURSE "${cache_directory}")
    message(
        FATAL_ERROR
            "${description}\nstdout:\n${output}\nstderr:\n${error_output}")
endfunction()

file(MAKE_DIRECTORY "${cache_directory}")

# The first process compiles a named (non-JIT-expression) CUDA kernel and
# persists its cubin plus NVRTC mangled-name table.
execute_process(
    COMMAND "${HELPER_EXECUTABLE}" "${cache_directory}" "${CUDA_DEVICE}"
    RESULT_VARIABLE write_result
    OUTPUT_VARIABLE write_output
    ERROR_VARIABLE write_error)
if(NOT write_result EQUAL 0)
    fail_after_cleanup("CUDA cache writer failed (${write_result})"
                       "${write_output}" "${write_error}")
endif()

file(GLOB cache_files LIST_DIRECTORIES FALSE "${cache_directory}/KER*.bin")
list(LENGTH cache_files cache_file_count)
if(cache_file_count EQUAL 0)
    fail_after_cleanup("CUDA cache writer produced no kernel cache file"
                       "${write_output}" "${write_error}")
endif()

# A fresh process has an empty in-memory module cache. Logging verifies that
# its successful convolution came from the on-disk module rather than a
# transparent recompile.
execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env "AF_TRACE=jit" "${HELPER_EXECUTABLE}"
        "${cache_directory}" "${CUDA_DEVICE}"
    RESULT_VARIABLE read_result
    OUTPUT_VARIABLE read_output
    ERROR_VARIABLE read_error)
if(NOT read_result EQUAL 0)
    fail_after_cleanup("CUDA cache reader failed (${read_result})"
                       "${read_output}" "${read_error}")
endif()

set(read_log "${read_output}\n${read_error}")
if(NOT read_log MATCHES "loaded from")
    fail_after_cleanup(
        "CUDA cache reader did not report loading the persisted module"
        "${read_output}" "${read_error}")
endif()
if(read_log MATCHES "compile:")
    fail_after_cleanup(
        "CUDA cache reader recompiled a module instead of using the cache"
        "${read_output}" "${read_error}")
endif()
if(read_log MATCHES
   "Unable to open|Unable to read|Corrupt binary|cuModuleLoadData failed")
    fail_after_cleanup("CUDA cache reader reported a rejected cache entry"
                       "${read_output}" "${read_error}")
endif()

file(REMOVE_RECURSE "${cache_directory}")
