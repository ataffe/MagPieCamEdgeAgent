cmake_minimum_required(VERSION 3.16)
project(byte_track_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

## --- ByteTrack core (no external dependencies; all matrices are <= 8x8) ---
#add_library(bytetrack
#    src/kalman_filter.cpp
#    src/strack.cpp
#    src/matching.cpp
#    src/byte_tracker.cpp)
#target_include_directories(bytetrack PUBLIC include)
#set_target_properties(bytetrack PROPERTIES POSITION_INDEPENDENT_CODE ON)
#
## --- Core smoke test ---
#add_executable(smoke src/main_smoke.cpp)
#target_link_libraries(smoke PRIVATE bytetrack)
#
## --- Optional: pybind11 Python extension (bytetrack_cpp.so) ---
## Install pybind11 first: pip install pybind11 --user
#find_package(Python3 COMPONENTS Interpreter Development)
#if(Python3_FOUND)
#    execute_process(
#        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
#        OUTPUT_VARIABLE pybind11_DIR
#        OUTPUT_STRIP_TRAILING_WHITESPACE
#        RESULT_VARIABLE _pybind11_rc
#    )
#    if(_pybind11_rc EQUAL 0)
#        find_package(pybind11 CONFIG QUIET)
#    endif()
#    if(pybind11_FOUND)
#        pybind11_add_module(bytetrack_cpp src/bytetrack_pybind.cpp)
#        target_link_libraries(bytetrack_cpp PRIVATE bytetrack)
#        message(STATUS "pybind11 found — bytetrack_cpp Python module will be built")
#    else()
#        message(STATUS "pybind11 not found — skipping bytetrack_cpp Python module (pip install pybind11 --user)")
#    endif()
#endif()


# --- Standalone camera app ---
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
