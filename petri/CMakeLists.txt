cmake_minimum_required(VERSION 3.18)
project(petri LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU architecture for GTX 1650 Ti (Turing, SM 75)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Fetch Raylib
include(FetchContent)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        5.0
)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(raylib)

# Header include paths
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${raylib_SOURCE_DIR}/src
)

# Source files
file(GLOB_RECURSE SRC_FILES
    "src/*.cpp"
    "src/*.h"
    "src/*.cu"
)

add_executable(petri main.cpp ${SRC_FILES})

# Target properties for CUDA
set_target_properties(petri PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(petri PRIVATE raylib)

# Link system libraries needed by raylib on Linux
target_link_libraries(petri PRIVATE
    GL
    m
    pthread
    dl
    rt
    X11
)
