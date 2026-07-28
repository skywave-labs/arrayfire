# Copyright (c) 2026, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED TEST_CASE)
    foreach(test_spec
            "cuda_12_7,12.7"
            "cuda_12_8,12.8"
            "cuda_12_9,12.9"
            "cuda_13_0,13.0"
            "cuda_13_2,13.2")
        string(REPLACE "," ";" test_spec "${test_spec}")
        list(GET test_spec 0 test_case)
        list(GET test_spec 1 cuda_version)
        execute_process(
            COMMAND
                "${CMAKE_COMMAND}" "-DTEST_CASE=${test_case}"
                "-DCUDA_VERSION=${cuda_version}" -P
                "${CMAKE_CURRENT_LIST_FILE}"
            RESULT_VARIABLE test_result)
        if(NOT test_result EQUAL 0)
            message(
                FATAL_ERROR
                    "CUDA architecture selection test ${test_case} failed")
        endif()
    endforeach()
    return()
endif()

get_filename_component(ARRAYFIRE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.."
                       ABSOLUTE)
include("${ARRAYFIRE_SOURCE_DIR}/CMakeModules/select_compute_arch.cmake")

function(assert_list_contains list_name item)
    list(FIND ${list_name} "${item}" item_index)
    if(item_index EQUAL -1)
        message(FATAL_ERROR
                "${list_name} does not contain ${item}: ${${list_name}}")
    endif()
endfunction()

function(assert_list_excludes list_name item)
    list(FIND ${list_name} "${item}" item_index)
    if(NOT item_index EQUAL -1)
        message(FATAL_ERROR
                "${list_name} unexpectedly contains ${item}: ${${list_name}}")
    endif()
endfunction()

function(assert_equal actual expected description)
    if(NOT "${actual}" STREQUAL "${expected}")
        message(FATAL_ERROR
                "${description}: expected '${expected}', received '${actual}'")
    endif()
endfunction()

function(readable_flags_to_list output_variable readable_flags)
    string(REPLACE " " ";" flag_list "${readable_flags}")
    set(${output_variable} "${flag_list}" PARENT_SCOPE)
endfunction()

if(TEST_CASE STREQUAL "cuda_12_7")
    assert_list_excludes(CUDA_KNOWN_GPU_ARCHITECTURES "Blackwell")
    assert_list_contains(CUDA_COMMON_GPU_ARCHITECTURES "9.0+PTX")
    assert_equal("${CUDA_LIMIT_GPU_ARCHITECTURE}" "9.0"
                 "CUDA 12.7 architecture limit")
elseif(TEST_CASE STREQUAL "cuda_12_8")
    assert_list_contains(CUDA_KNOWN_GPU_ARCHITECTURES "Blackwell")
    foreach(architecture "10.0" "10.1" "12.0")
        assert_list_contains(CUDA_COMMON_GPU_ARCHITECTURES "${architecture}")
    endforeach()
    assert_list_excludes(CUDA_COMMON_GPU_ARCHITECTURES "10.3")
    assert_list_excludes(CUDA_COMMON_GPU_ARCHITECTURES "12.1")
    assert_list_contains(CUDA_COMMON_GPU_ARCHITECTURES "12.0+PTX")
    assert_equal("${CUDA_LIMIT_GPU_ARCHITECTURE}" "12.1"
                 "CUDA 12.8 architecture limit")

    CUDA_SELECT_NVCC_ARCH_FLAGS(flags 10.0 12.0+PTX)
    set(expected_flags
        -gencode arch=compute_100,code=sm_100
        -gencode arch=compute_120,code=sm_120
        -gencode arch=compute_120,code=compute_120)
    assert_equal("${flags}" "${expected_flags}" "CUDA 12.8 explicit flags")
    assert_equal("${flags_readable}" "sm_100 sm_120 compute_120"
                 "CUDA 12.8 readable flags")

    CUDA_SELECT_NVCC_ARCH_FLAGS(nested_flags "12.0(10.0)")
    set(expected_nested_flags -gencode arch=compute_100,code=sm_120)
    assert_equal("${nested_flags}" "${expected_nested_flags}"
                 "CUDA 12.8 explicit code/architecture flags")

    CUDA_SELECT_NVCC_ARCH_FLAGS(all_flags All)
    readable_flags_to_list(all_architectures "${all_flags_readable}")
    assert_list_contains(all_architectures "sm_87")

    set(CUDA_GPU_DETECT_OUTPUT "8.7")
    CUDA_SELECT_NVCC_ARCH_FLAGS(auto_flags Auto)
    assert_equal("${auto_flags_readable}" "sm_87"
                 "CUDA 12.8 Auto preserves SM87")

    set(CUDA_GPU_DETECT_OUTPUT "12.1")
    CUDA_DETECT_INSTALLED_GPUS(detected_architecture)
    string(STRIP "${detected_architecture}" detected_architecture)
    assert_equal("${detected_architecture}" "12.0+PTX"
                 "CUDA 12.8 unsupported architecture fallback")

    set(CUDA_GPU_DETECT_OUTPUT "10.3 11.0")
    CUDA_DETECT_INSTALLED_GPUS(detected_architecture)
    string(STRIP "${detected_architecture}" detected_architecture)
    assert_equal("${detected_architecture}" "10.1+PTX 10.1+PTX"
                 "CUDA 12.8 sparse architecture fallbacks")

    set(CUDA_GPU_DETECT_OUTPUT "8.8")
    CUDA_DETECT_INSTALLED_GPUS(detected_architecture)
    string(STRIP "${detected_architecture}" detected_architecture)
    assert_equal("${detected_architecture}" "8.7+PTX"
                 "CUDA 12.8 SM88 compatibility fallback")
elseif(TEST_CASE STREQUAL "cuda_12_9")
    foreach(architecture "10.0" "10.1" "10.3" "12.0" "12.1")
        assert_list_contains(CUDA_COMMON_GPU_ARCHITECTURES "${architecture}")
    endforeach()
    assert_list_contains(CUDA_COMMON_GPU_ARCHITECTURES "12.1+PTX")
    assert_equal("${CUDA_LIMIT_GPU_ARCHITECTURE}" "12.2"
                 "CUDA 12.9 architecture limit")

    CUDA_SELECT_NVCC_ARCH_FLAGS(flags Blackwell)
    assert_equal("${flags_readable}" "sm_100 sm_101 sm_103 sm_120 sm_121"
                 "CUDA 12.9 Blackwell flags")

    set(CUDA_GPU_DETECT_OUTPUT "11.0")
    CUDA_DETECT_INSTALLED_GPUS(detected_architecture)
    string(STRIP "${detected_architecture}" detected_architecture)
    assert_equal("${detected_architecture}" "10.3+PTX"
                 "CUDA 12.9 sparse architecture fallback")
elseif(TEST_CASE STREQUAL "cuda_13_0" OR TEST_CASE STREQUAL "cuda_13_2")
    foreach(architecture
            "8.8"
            "8.9"
            "9.0"
            "10.0"
            "10.3"
            "11.0"
            "12.0"
            "12.1")
        assert_list_contains(CUDA_COMMON_GPU_ARCHITECTURES "${architecture}")
    endforeach()
    foreach(architecture "5.0" "6.0" "7.0" "10.1")
        assert_list_excludes(CUDA_COMMON_GPU_ARCHITECTURES "${architecture}")
    endforeach()
    foreach(architecture "Kepler" "Maxwell" "Pascal" "Volta")
        assert_list_excludes(CUDA_KNOWN_GPU_ARCHITECTURES "${architecture}")
    endforeach()
    assert_list_contains(CUDA_COMMON_GPU_ARCHITECTURES "12.1+PTX")
    assert_equal("${CUDA_LIMIT_GPU_ARCHITECTURE}" "12.2"
                 "CUDA 13 architecture limit")

    CUDA_SELECT_NVCC_ARCH_FLAGS(flags Blackwell)
    assert_equal("${flags_readable}" "sm_100 sm_103 sm_110 sm_120 sm_121"
                 "CUDA 13 Blackwell flags")

    CUDA_SELECT_NVCC_ARCH_FLAGS(all_flags All)
    readable_flags_to_list(all_architectures "${all_flags_readable}")
    foreach(architecture
            "sm_75"
            "sm_80"
            "sm_86"
            "sm_87"
            "sm_88"
            "sm_89"
            "sm_90"
            "sm_100"
            "sm_103"
            "sm_110"
            "sm_120"
            "sm_121")
        assert_list_contains(all_architectures "${architecture}")
    endforeach()
    foreach(architecture "sm_50" "sm_61" "sm_70" "sm_101")
        assert_list_excludes(all_architectures "${architecture}")
    endforeach()

    set(CUDA_GPU_DETECT_OUTPUT "12.1")
    CUDA_DETECT_INSTALLED_GPUS(detected_architecture)
    string(STRIP "${detected_architecture}" detected_architecture)
    assert_equal("${detected_architecture}" "12.1"
                 "CUDA 13 native architecture detection")

    set(CUDA_GPU_DETECT_OUTPUT "10.1")
    CUDA_DETECT_INSTALLED_GPUS(detected_architecture)
    string(STRIP "${detected_architecture}" detected_architecture)
    assert_equal("${detected_architecture}" "10.0+PTX"
                 "CUDA 13 renamed architecture fallback")
else()
    message(FATAL_ERROR "Unknown TEST_CASE: ${TEST_CASE}")
endif()
