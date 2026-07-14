# - Find the FFTW library
#
# Usage:
#   FIND_PACKAGE(FFTW [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   FFTW_FOUND               ... true if fftw is found on the system
#   FFTW_LIBRARIES           ... full path to fftw library
#   FFTW_INCLUDES            ... fftw include directory
#   FFTW_THREADS_FOUND       ... true if the required FFTW thread APIs are found
#
# The following variables will be checked by the function
#   FFTW_USE_STATIC_LIBS    ... if true, only static libraries are found
#   FFTW_ROOT               ... if set, the libraries are exclusively searched
#                               under this path
#   FFTW_LIBRARY            ... fftw library to use
#   FFTW_INCLUDE_DIR        ... fftw include directory
#
#If environment variable FFTWDIR is specified, it has same effect as FFTW_ROOT

######## This FindFFTW.cmake file is a copy of the file from the eigen library
######## http://code.metager.de/source/xref/lib/eigen/cmake/FindFFTW.cmake

find_package(PkgConfig)
pkg_check_modules(PKG_FFTW "fftw3")

find_path( FFTW_INCLUDE_DIR
  NAMES "fftw3.h"
  PATHS ${FFTW_ROOT}
        ${CMAKE_SYSTEM_INCLUDE_PATH}
        ${CMAKE_SYSTEM_PREFIX_PATH}
        ${PKG_FFTW_INCLUDE_DIRS}
  PATH_SUFFIXES "include" "include/fftw"
  )

find_library( FFTW_LIBRARY
  NAMES "fftw3" "libfftw3-3" "fftw3-3"
  PATHS ${FFTW_ROOT}
        ${CMAKE_SYSTEM_PREFIX_PATH}
        ${PKG_FFTW_LIBRARY_DIRS}
  PATH_SUFFIXES "lib" "lib64"
)

find_library( FFTWF_LIBRARY
  NAMES "fftw3f" "libfftw3f-3" "fftw3f-3"
  PATHS ${FFTW_ROOT}
        ${CMAKE_SYSTEM_PREFIX_PATH}
        ${CMAKE_SYSTEM_LIBRARY_PATH}
        ${PKG_FFTW_LIBRARY_DIRS}
  PATH_SUFFIXES "lib" "lib64"
)

find_library( FFTW_THREADS_LIBRARY
  NAMES "fftw3_threads" "libfftw3_threads-3" "fftw3_threads-3"
  PATHS ${FFTW_ROOT}
        ${CMAKE_SYSTEM_PREFIX_PATH}
        ${CMAKE_SYSTEM_LIBRARY_PATH}
        ${PKG_FFTW_LIBRARY_DIRS}
  PATH_SUFFIXES "lib" "lib64"
)

find_library( FFTWF_THREADS_LIBRARY
  NAMES "fftw3f_threads" "libfftw3f_threads-3" "fftw3f_threads-3"
  PATHS ${FFTW_ROOT}
        ${CMAKE_SYSTEM_PREFIX_PATH}
        ${CMAKE_SYSTEM_LIBRARY_PATH}
        ${PKG_FFTW_LIBRARY_DIRS}
  PATH_SUFFIXES "lib" "lib64"
)

mark_as_advanced(FFTW_INCLUDE_DIR FFTW_LIBRARY FFTWF_LIBRARY
                 FFTW_THREADS_LIBRARY FFTWF_THREADS_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG
    FFTW_INCLUDE_DIR FFTW_LIBRARY FFTWF_LIBRARY)

if (FFTW_FOUND)
  add_library(FFTW::FFTW UNKNOWN IMPORTED)
  set_target_properties(FFTW::FFTW PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGE "C"
    IMPORTED_LOCATION "${FFTW_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}")

  add_library(FFTW::FFTWF UNKNOWN IMPORTED)
  set_target_properties(FFTW::FFTWF PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGE "C"
    IMPORTED_LOCATION "${FFTWF_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}")

  set(FFTW_THREADS_FOUND FALSE)
  if (FFTW_THREADS_LIBRARY AND FFTWF_THREADS_LIBRARY AND Threads_FOUND)
    include(CheckSymbolExists)
    include(CMakePushCheckState)

    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES "${FFTW_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES ${FFTW_THREADS_LIBRARY} ${FFTW_LIBRARY}
                                 ${CMAKE_THREAD_LIBS_INIT})
    if (UNIX)
      list(APPEND CMAKE_REQUIRED_LIBRARIES m)
    endif()
    check_symbol_exists(fftw_planner_nthreads "fftw3.h"
                        FFTW_PLANNER_NTHREADS_FOUND)
    cmake_pop_check_state()

    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES "${FFTW_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES ${FFTWF_THREADS_LIBRARY} ${FFTWF_LIBRARY}
                                 ${CMAKE_THREAD_LIBS_INIT})
    if (UNIX)
      list(APPEND CMAKE_REQUIRED_LIBRARIES m)
    endif()
    check_symbol_exists(fftwf_planner_nthreads "fftw3.h"
                        FFTWF_PLANNER_NTHREADS_FOUND)
    cmake_pop_check_state()
  endif()

  if (FFTW_PLANNER_NTHREADS_FOUND AND FFTWF_PLANNER_NTHREADS_FOUND)
    set(FFTW_THREADS_FOUND TRUE)

    add_library(FFTW::FFTW_THREADS UNKNOWN IMPORTED)
    set_target_properties(FFTW::FFTW_THREADS PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGE "C"
      IMPORTED_LOCATION "${FFTW_THREADS_LIBRARY}"
      INTERFACE_LINK_LIBRARIES "FFTW::FFTW;Threads::Threads")

    add_library(FFTW::FFTWF_THREADS UNKNOWN IMPORTED)
    set_target_properties(FFTW::FFTWF_THREADS PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGE "C"
      IMPORTED_LOCATION "${FFTWF_THREADS_LIBRARY}"
      INTERFACE_LINK_LIBRARIES "FFTW::FFTWF;Threads::Threads")
  endif()
endif (FFTW_FOUND)
