cmake_minimum_required(VERSION 3.16)
project(SpeechEnhancer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Optimization flags ────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ── ONNXRuntime ────────────────────────────────────────────────────────────
# Set ONNXRUNTIME_DIR to your ONNXRuntime installation, e.g.:
#   cmake .. -DONNXRUNTIME_DIR=/opt/onnxruntime
#
# Download from: https://github.com/microsoft/onnxruntime/releases

if(NOT DEFINED ONNXRUNTIME_DIR)
    # Try to find system-installed ONNXRuntime
    find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS /usr/local/include/onnxruntime /opt/onnxruntime/include
        PATH_SUFFIXES onnxruntime
    )
    find_library(ORT_LIB onnxruntime
        HINTS /usr/local/lib /opt/onnxruntime/lib
    )
else()
    set(ORT_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
    find_library(ORT_LIB onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib)
endif()

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
    message(FATAL_ERROR
        "ONNXRuntime not found. Download from:\n"
        "  https://github.com/microsoft/onnxruntime/releases\n"
        "Then set: cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime"
    )
endif()

message(STATUS "ONNXRuntime include: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNXRuntime lib:     ${ORT_LIB}")

# ── Executable ────────────────────────────────────────────────────────────
add_executable(speech_enhancer
    src/speech_enhancer.cpp
)

target_include_directories(speech_enhancer PRIVATE
    include/
    ${ORT_INCLUDE_DIR}
)

target_link_libraries(speech_enhancer PRIVATE
    ${ORT_LIB}
    m        # math
)

# ── Optional: link FFTW3 for fast FFT (recommended for production) ─────────
find_package(FFTW3 QUIET)
if(FFTW3_FOUND)
    message(STATUS "FFTW3 found — enabling fast FFT")
    target_link_libraries(speech_enhancer PRIVATE FFTW3::fftw3f)
    target_compile_definitions(speech_enhancer PRIVATE USE_FFTW)
else()
    message(STATUS "FFTW3 not found — using naive DFT (install fftw3 for production)")
endif()

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS speech_enhancer DESTINATION bin)

# ── Build info ────────────────────────────────────────────────────────────
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags:  ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
