# ======================================================================== #
# Copyright 2018-2024 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.20)
  # honor the LANGUAGE source property (compile .cu files as C++ in DC_CPU mode)
  cmake_policy(SET CMP0119 NEW)
endif()

if (NOT (DEFINED CMAKE_CUDA_ARCHITECTURES))
  if (NOT CMAKE_VERSION VERSION_LESS "3.17")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(DeepComp VERSION 1.0.0 LANGUAGES C CXX)

# ------------------------------------------------------------------
# CUDA (original) or host/CPU (C++ + OpenMP + plain MPI) implementation.
# DC_CPU defaults to ON when no CUDA compiler is available.
# ------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  set(DC_CPU_DEFAULT OFF)
else()
  set(DC_CPU_DEFAULT ON)
endif()
option(DC_CPU "Build the host (CPU, OpenMP + host MPI) implementation instead of CUDA" ${DC_CPU_DEFAULT})
if (NOT DC_CPU)
  enable_language(CUDA)
endif()

find_package(MPI REQUIRED)

include(GNUInstallDirs)

configure_file(dcConfig.h.in ${PROJECT_BINARY_DIR}/dcConfig.h)

#add_subdirectory(../owl build_owl EXCLUDE_FROM_ALL)
if (DC_CPU)
  find_package(OpenMP REQUIRED)
  option(DC_BUILD_SHARED "Build libdeepCompositing as a shared library (CPU build)" ON)
  if (DC_BUILD_SHARED)
    set(DC_LIB_TYPE SHARED)
  else()
    set(DC_LIB_TYPE STATIC)
  endif()
  add_library(deepCompositing ${DC_LIB_TYPE}
    deepCompositing.h
    dc_host_math.h
    deepCompositing_cpu.cpp
    )
  target_link_libraries(deepCompositing
    PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    )
  target_include_directories(deepCompositing
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/deepCompositing>
    )
  set_target_properties(deepCompositing
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    )
else()
  add_library(deepCompositing STATIC
    deepCompositing.h
    deepCompositing.cu
    )
  target_link_libraries(deepCompositing
    PUBLIC
    MPI::MPI_CXX
    )
  target_include_directories(deepCompositing
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
  set_target_properties(deepCompositing
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_USE_STATIC_CUDA_RUNTIME ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  )
endif()

## Installation of the library (CPU build) ##

if (DC_CPU)
  install(TARGETS deepCompositing
    EXPORT deepCompositingTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  install(FILES
    deepCompositing.h
    dc_host_math.h
    mpi_helper.h
    ${PROJECT_BINARY_DIR}/dcConfig.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/deepCompositing
    )
  install(EXPORT deepCompositingTargets
    NAMESPACE deepCompositing::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/deepCompositing
    )
  file(WRITE ${PROJECT_BINARY_DIR}/deepCompositingConfig.cmake
"include(CMakeFindDependencyMacro)
find_dependency(MPI)
find_dependency(OpenMP)
include(\${CMAKE_CURRENT_LIST_DIR}/deepCompositingTargets.cmake)
")
  install(FILES ${PROJECT_BINARY_DIR}/deepCompositingConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/deepCompositing
    )
endif()


if ((${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

  add_subdirectory(samples)
endif()

option(DC_BUILD_ANARI "Build the 'ptc' ANARI compositing wrapper device (needs ANARI-SDK)" ON)
if (DC_BUILD_ANARI)
  add_subdirectory(anari)
endif()
