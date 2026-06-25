cmake_minimum_required(VERSION 3.14)
project(DDFacetCpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# =============================================================================
# Dependências
# Ubuntu/WSL: sudo apt install libfftw3-dev
# =============================================================================

find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED
    HINTS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu)

find_library(FFTW3_LIB NAMES fftw3 REQUIRED
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib)

# planner thread-safe do FFTW (necessário ao paralelizar facetas com OpenMP)
find_library(FFTW3_THREADS_LIB NAMES fftw3_threads REQUIRED
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib)

# Paralelização: OpenMP (eixo intra-nó / facetas e visibilidades)
find_package(OpenMP REQUIRED)

# casacore: leitura de Measurement Sets reais (formato CASA)
find_path(CASA_INC casacore/tables/Tables.h HINTS /usr/include)
find_library(CASA_CASA   NAMES casa_casa   REQUIRED)
find_library(CASA_TABLES NAMES casa_tables REQUIRED)
find_library(CASA_MS     NAMES casa_ms     REQUIRED)

# FUTURO (Fase 3):
#   find_package(MPI REQUIRED)   # eixo multi-nó (MS)

# Include (casacore como SYSTEM para não emitir warnings dos headers dele)
include_directories(${CMAKE_SOURCE_DIR}/include ${FFTW3_INCLUDE_DIR})
include_directories(SYSTEM ${CASA_INC})

# Sources
add_executable(ddfacet_cpp
    src/main.cpp
    src/ddfacet.cpp
    src/fft.cpp
    src/ms_io.cpp
)

target_link_libraries(ddfacet_cpp
    ${FFTW3_THREADS_LIB} ${FFTW3_LIB} m OpenMP::OpenMP_CXX
    ${CASA_MS} ${CASA_TABLES} ${CASA_CASA})

# MPI opcional (eixo J / multi-nó):  cmake -DUSE_MPI=ON ..
option(USE_MPI "Compilar com MPI (eixo J multi-nó)" OFF)
if(USE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(ddfacet_cpp MPI::MPI_CXX)
    target_compile_definitions(ddfacet_cpp PRIVATE USE_MPI)
    message(STATUS "MPI        : ${MPI_CXX_COMPILER} (USE_MPI=ON)")
endif()

install(TARGETS ddfacet_cpp DESTINATION bin)

message(STATUS "")
message(STATUS "DDFacet C++ Configuration Summary")
message(STATUS "==================================")
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Std    : ${CMAKE_CXX_STANDARD}")
message(STATUS "FFTW3      : ${FFTW3_LIB}")
message(STATUS "OpenMP     : ${OpenMP_CXX_FLAGS}")
message(STATUS "casacore   : ${CASA_MS}")
message(STATUS "")
