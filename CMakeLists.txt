# Copyright © 2026 Alexander Taffe

cmake_minimum_required(VERSION 3.16)
project(byte_track_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")


# --- presigned test
include(FetchContent)
# cpr
FetchContent_Declare(cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
        GIT_TAG 1.14.2)
# libpsl (a curl dependency pulled in by cpr 1.14) requires meson to build from
# source, which is not available on CI runners. Disable it — it is only needed
# for cookie domain validation, which this client does not use.
set(CPR_CURL_USE_LIBPSL OFF CACHE BOOL "" FORCE)
# Use the system's libcurl (requires libcurl4-openssl-dev) instead of building
# our own from source. opencv_imgcodecs pulls in the system libhdf5, which is
# linked against the system libcurl's *versioned* symbols (CURL_OPENSSL_4). A
# self-built curl exports unversioned symbols, so it can't satisfy hdf5's
# reference and the final link fails with undefined `curl_*@CURL_OPENSSL_4`.
set(CPR_USE_SYSTEM_CURL ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cpr)

# nlohmann/json
FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)

# spdlog
FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.15.1)
FetchContent_MakeAvailable(spdlog)

# --- Backend client (camera registration + presigned-URL upload to S3) ---
add_library(backend_client src/backend/backend_client.cpp)
target_include_directories(backend_client PUBLIC include)
target_link_libraries(backend_client PUBLIC cpr::cpr nlohmann_json::nlohmann_json spdlog::spdlog)
set_target_properties(backend_client PROPERTIES POSITION_INDEPENDENT_CODE ON)


# --- ByteTrack core (no external dependencies; all matrices are <= 8x8) ---
add_library(bytetrack
        src/tracking/kalman_filter.cpp
        src/tracking/strack.cpp
        src/tracking/matching.cpp
        src/tracking/byte_tracker.cpp)
target_include_directories(bytetrack PUBLIC include)
set_target_properties(bytetrack PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Camera Client ---
# The `track` executable needs the Raspberry Pi camera stack (libcamera, OpenCV,
# rpicam-apps), which only exists on the Pi. Keep it ON for the device build, but
# let CI / dev machines turn it OFF to build just the library and unit tests.
option(BUILD_CAMERA_APP "Build the libcamera-based camera client (track)" ON)
if(BUILD_CAMERA_APP)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBCAMERA REQUIRED libcamera)
    pkg_check_modules(OPENCV    REQUIRED opencv4)

    # The boost program_options dev package (symlink + cmake config) isn't installed,
    # but the versioned runtime .so is. Link it directly by full path. NAMES carries
    # the soname so we don't depend on a -dev symlink.
    find_library(BOOST_PROGRAM_OPTIONS
            NAMES boost_program_options libboost_program_options.so.1.83.0
            PATHS /usr/lib/aarch64-linux-gnu)
    if(NOT BOOST_PROGRAM_OPTIONS)
            message(FATAL_ERROR "libboost_program_options not found")
    endif()

    include(FetchContent)
    FetchContent_Declare(
            argparse
            GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    )
    FetchContent_MakeAvailable(argparse)

    add_executable(track src/main_track.cpp
            src/parsers/imx500_yolo_parser.cpp
            src/server/mjpeg_server.cpp)
    target_include_directories(track PRIVATE
            include
            /usr/include/rpicam-apps)
    target_compile_options(track PRIVATE
            ${LIBCAMERA_CFLAGS} ${OPENCV_CFLAGS})
    # Link only the OpenCV modules actually used (imgproc, imgcodecs) rather than
    # every module in opencv4.pc. Pulling in the full list drags in opencv_hdf,
    # which links the system libcurl (versioned symbols) — conflicting with the
    # unversioned libcurl built from source by cpr's FetchContent and breaking
    # the link with undefined `curl_*@CURL_OPENSSL_4` references.
    target_link_libraries(track PRIVATE
            bytetrack
            rpicam_app # the rpicam-apps shared library
            argparse
            backend_client
            spdlog::spdlog
            ${LIBCAMERA_LIBRARIES}
            opencv_core
            opencv_imgproc
            opencv_imgcodecs
            ${BOOST_PROGRAM_OPTIONS})
endif()

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
            test/test_backend_client.cpp
            # The parser lives in the `track` exe, not the bytetrack lib, but it only
            # depends on byte_tracker.h (no libcamera), so compile it straight in.
            src/parsers/imx500_yolo_parser.cpp)
    target_link_libraries(bytetrack_tests PRIVATE bytetrack backend_client GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(bytetrack_tests)
endif()
