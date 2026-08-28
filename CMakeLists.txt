cmake_minimum_required(VERSION 3.14)
project(ddfacet_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
add_compile_options(-Wall -Wextra)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

find_package(OpenMP REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
#  ddfacet_ompc — pipeline distribuído por canal (OpenMP Cluster)
#
#  Sem dependências externas: a FFT e a escrita FITS são próprias, porque o
#  container do OMPC não traz FFTW nem casacore. Compilado com g++ as regiões
#  `omp target` executam no host; no cluster, use o clang do container:
#
#    clang++ -O2 -std=c++17 -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu \
#            -Iinclude src/main_ompc.cpp -o ddfacet_ompc
#
#  (ver scripts/sorgan_ompc.sh)
# ─────────────────────────────────────────────────────────────────────────────
add_executable(ddfacet_ompc src/main_ompc.cpp)
target_include_directories(ddfacet_ompc PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ddfacet_ompc PRIVATE OpenMP::OpenMP_CXX)

# ─────────────────────────────────────────────────────────────────────────────
#  ms_export — Measurement Set → um arquivo .vis por canal
#
#  Roda no HOST e precisa de casacore; é o que alimenta o binário OMPC.
# ─────────────────────────────────────────────────────────────────────────────
find_path(CASACORE_INCLUDE_DIR casacore/tables/Tables/Table.h
          PATHS /usr/include /usr/local/include)
find_library(CASACORE_MS      casa_ms)
find_library(CASACORE_TABLES  casa_tables)
find_library(CASACORE_CASA    casa_casa)

if(CASACORE_INCLUDE_DIR AND CASACORE_MS AND CASACORE_TABLES AND CASACORE_CASA)
    add_executable(ms_export tools/ms_export.cpp src/ms_io.cpp)
    target_include_directories(ms_export PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CASACORE_INCLUDE_DIR}/casacore)
    target_link_libraries(ms_export PRIVATE
        ${CASACORE_MS} ${CASACORE_TABLES} ${CASACORE_CASA})
    message(STATUS "casacore   : ${CASACORE_INCLUDE_DIR} (ms_export habilitado)")
else()
    message(WARNING "casacore nao encontrado — ms_export nao sera compilado")
endif()

message(STATUS "build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP     : ${OpenMP_CXX_VERSION}")
