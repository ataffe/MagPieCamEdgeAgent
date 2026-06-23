# Copyright © 2026 Alexander Taffe

cmake_minimum_required(VERSION 3.16)
project(byte_track_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")


# --- Camera Client ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)
pkg_check_modules(OPENCV    REQUIRED opencv4)

# --- ByteTrack core (no external dependencies; all matrices are <= 8x8) ---
add_library(bytetrack
        src/tracking/kalman_filter.cpp
        src/tracking/strack.cpp
        src/tracking/matching.cpp
        src/tracking/byte_tracker.cpp)
target_include_directories(bytetrack PUBLIC include)
set_target_properties(bytetrack PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The boost program_options dev package (symlink + cmake config) isn't installed,
# but the versioned runtime .so is. Link it directly by full path. NAMES carries
# the soname so we don't depend on a -dev symlink.
find_library(BOOST_PROGRAM_OPTIONS
        NAMES boost_program_options libboost_program_options.so.1.83.0
        PATHS /usr/lib/aarch64-linux-gnu)
if(NOT BOOST_PROGRAM_OPTIONS)
        message(FATAL_ERROR "libboost_program_options not found")
endif()

add_executable(track src/main_track.cpp
        src/parsers/imx500_yolo_parser.cpp
        src/server/mjpeg_server.cpp)
target_include_directories(track PRIVATE
        include
        /usr/include/rpicam-apps)
target_compile_options(track PRIVATE
        ${LIBCAMERA_CFLAGS} ${OPENCV_CFLAGS})
target_link_libraries(track PRIVATE
        bytetrack
        rpicam_app          # the rpicam-apps shared library
        ${LIBCAMERA_LIBRARIES}
        ${OPENCV_LIBRARIES}
        ${BOOST_PROGRAM_OPTIONS})

# --- Unit tests (GoogleTest) ---
# Off by default so the normal Pi build stays lean and offline. Enable with:
#   cmake -S . -B build-test -DBUILD_TESTING=ON && cmake --build build-test
#   ctest --test-dir build-test --output-on-failure
option(BUILD_TESTING "Build unit tests" OFF)
if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2)
    # Don't install GoogleTest
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(bytetrack_tests
            test/test_matching.cpp
            test/test_strack.cpp
            test/test_kalman_filter.cpp
            test/test_byte_tracker.cpp
            test/test_imx500_yolo_parser.cpp
            # The parser lives in the `track` exe, not the bytetrack lib, but it only
            # depends on byte_tracker.h (no libcamera), so compile it straight in.
            src/parsers/imx500_yolo_parser.cpp)
    target_link_libraries(bytetrack_tests PRIVATE bytetrack GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(bytetrack_tests)
endif()
