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

find_package(anari 0.8.0 REQUIRED)

## PTC device target ##

if (DC_CPU)
  project(anari_library_ptc LANGUAGES C CXX)
  # FrameWrapper.cu holds both implementations; in the CPU build it is
  # plain C++ (no kernels, see the DC_CPU sections in it)
  set_source_files_properties(FrameWrapper.cu PROPERTIES LANGUAGE CXX)
else()
  project(anari_library_ptc LANGUAGES C CXX CUDA)
endif()

project_add_library(MODULE PTCDevice.cpp PTCLibrary.cpp FrameWrapper.cu)
project_link_libraries(anari::helium deepCompositing)

if (NOT DC_CPU)
  set_source_files_properties(FrameWrapper.cu
    PROPERTIES COMPILE_FLAGS "--expt-relaxed-constexpr"
  )
endif()

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} EXPORT_MACRO_NAME "PTC_EXPORT")
project_include_directories(PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

## Installation ##

include(GNUInstallDirs)

if (DC_CPU)
  # find the installed libdeepCompositing.so next to the device
  set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
