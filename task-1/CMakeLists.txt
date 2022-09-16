cmake_minimum_required(VERSION 3.10)

project(MpiIntegrate)

find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

find_package(OpenMP REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MpiIntegrate_SOURCE_DIR}/bin)

add_executable(MpiIntegrate main.cpp)

target_link_libraries(MpiIntegrate PRIVATE ${MPI_C_LIBRARIES})
target_link_libraries(MpiIntegrate PRIVATE OpenMP::OpenMP_CXX)
