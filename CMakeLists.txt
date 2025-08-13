#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(nd-light-transport LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES "89")
set(CMAKE_CUDA_STANDARD 20)
#set(CMAKE_CXX_STANDARD 20) 

# --- ImGui Integration Starts Here ---

# Define the path to your ImGui directory.
# This assumes ImGui is a subfolder named 'imgui' in your project root.
# Adjust this path if your ImGui directory is located elsewhere.
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
set ( IMGUI_SOURCES
    "${IMGUI_DIR}/imconfig.h"
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_internal.h"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui.h"
    "${IMGUI_DIR}/imstb_rectpack.h"
    "${IMGUI_DIR}/imstb_textedit.h"
    "${IMGUI_DIR}/imstb_truetype.h"
)

# EXAMPLE: Using GLFW for windowing and OpenGL 3 for rendering.
list(APPEND IMGUI_SOURCES
    "${IMGUI_DIR}/backends/imgui_impl_glfw.h"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.h"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
    #"${IMGUI_DIR}/backends/imgui_impl_vulkan.h"
    #"${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
    "${IMGUI_DIR}/misc/cpp/imgui_stdlib.h"
    "${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp"
)

# Source
set ( EXTERNAL
)

set ( SRC_CPU
  main.cpp
)

set ( SRC_GPU
  main.cu
)

# Specific compiler flags below. We're not going to add options for all possible compilers, but if
# you're new to CMake (like we are), the following may be a helpful example if you're using a
# different compiler or want to set different compiler options.

message (STATUS "Compiler ID: " ${CMAKE_CXX_COMPILER_ID})
message (STATUS "Release flags: " ${CMAKE_CXX_FLAGS_RELEASE})
message (STATUS "Debug flags: " ${CMAKE_CXX_FLAGS_DEBUG})

# Executables
add_executable(nd-raytracer      ${SRC_GPU} ${IMGUI_SOURCES})

# --- AddressSanitizer Configuration ---
# Enable AddressSanitizer for Debug and RelWithDebInfo builds.
# It adds significant overhead, so it's typically not used for Release builds.
if (CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    # Check if the compiler supports AddressSanitizer
    # GCC and Clang typically support it.
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        message(STATUS "Enabling AddressSanitizer for ${CMAKE_BUILD_TYPE} build.")
        target_compile_options(nd-raytracer PUBLIC -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(nd-raytracer PUBLIC -fsanitize=address)
    else()
        message(WARNING "AddressSanitizer requested but not supported by compiler: ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -O0 --keep-device-functions")
endif()

# For VS Code IntelliSense and other tools to pick up compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
#target_compile_features(nd-raytracer PUBLIC cxx_std_20)
target_compile_options(nd-raytracer PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>" "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>" "$<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=64>")

target_include_directories(nd-raytracer PUBLIC
    "${IMGUI_DIR}"
    "${IMGUI_DIR}/backends"
)

find_package(glfw3 3.4 REQUIRED)
target_link_libraries(nd-raytracer glfw)

find_package(OpenGL REQUIRED)
target_link_libraries(nd-raytracer OpenGL::GL)

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
#set_target_properties( raytracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)