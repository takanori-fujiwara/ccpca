cmake_minimum_required(VERSION 3.9)
project(ccpca)
set(PYBIND11_CPP_STANDARD -std=c++11)
set(CMAKE_CXX_FLAGS "-Wall -O3 -mtune=native -march=native")

set(Python_ADDITIONAL_VERSIONS 3.7)

find_package(PkgConfig REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})
message("Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message("Python3_LIBRARY_DIRS: ${Python3_LIBRARY_DIRS}")

find_package(pybind11 REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message("EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")

pybind11_add_module(cpca_cpp SHARED cpca.cpp cpca_wrap.cpp)
pybind11_add_module(ccpca_cpp SHARED cpca.cpp cpca_wrap.cpp ccpca.cpp ccpca_wrap.cpp)
