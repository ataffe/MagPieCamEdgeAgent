cmake_minimum_required(VERSION 3.16)
project(byte_track_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# --- ByteTrack core (no external dependencies; all matrices are <= 8x8) ---
add_library(bytetrack
    src/kalman_filter.cpp
    src/strack.cpp
    src/matching.cpp
    src/byte_tracker.cpp)
target_include_directories(bytetrack PUBLIC include)

# --- Core smoke test ---
add_executable(smoke src/main_smoke.cpp)
target_link_libraries(smoke PRIVATE bytetrack)

# The camera front-end target is added later, once librpicam-app-dev is installed.