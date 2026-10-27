# build interactive viewer IF (and only if) there is a clone of the
# owl repo checked out as a sister directory to this project
if (EXISTS ../../owl AND NOT DC_CPU)
  add_subdirectory(../../owl build_owl EXCLUDE_FROM_ALL)
  add_executable(testDeepViewer
    testDeepViewer.cpp
    testRenderer.cu
    )
  target_link_libraries(testDeepViewer
    deepCompositing
    owl_viewer
    stdc++fs # for experimental::filesystem
    )
  target_include_directories(testDeepViewer
    PUBLIC
    3rdParty/
    )
endif()




if (NOT (TARGET stb_image))
  add_subdirectory(3rdParty/stb_image EXCLUDE_FROM_ALL)
endif()

if (DC_CPU)
  # host version of the cube test renderer (OpenMP instead of a CUDA kernel)
  set(DC_TEST_RENDERER testRenderer_cpu.cpp)
else()
  set(DC_TEST_RENDERER testRenderer.cu)
endif()

add_executable(testDeepOffline
  testDeepOffline.cpp
  ${DC_TEST_RENDERER}
  )
target_link_libraries(testDeepOffline
  deepCompositing
  stb_image
  )
target_include_directories(testDeepOffline
  PUBLIC
  3rdParty/
  )





add_executable(dcGenTestData
  genTestData.cpp
  )
target_link_libraries(dcGenTestData
  deepCompositing
  stb_image
  )
target_include_directories(dcGenTestData
  PUBLIC
  3rdParty/
  )


# data-parallel ANARI rendering + deep compositing test (host build only)
if (DC_CPU AND DC_BUILD_ANARI)
  find_package(anari 0.8.0 REQUIRED)
  add_executable(dcAnariDataParallel
    dcAnariDataParallel.cpp
    )
  target_link_libraries(dcAnariDataParallel
    deepCompositing
    stb_image
    anari::anari
    )
  target_include_directories(dcAnariDataParallel
    PUBLIC
    3rdParty/
    )
endif()

if (DC_CPU)
  install(TARGETS testDeepOffline dcGenTestData
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  if (TARGET dcAnariDataParallel)
    install(TARGETS dcAnariDataParallel
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()
