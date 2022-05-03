cmake_minimum_required(VERSION 3.10)
project(MYSFM)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-fPIC")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})

# Eigen (required)
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})
add_definitions(-DEIGEN_USE_NEW_STDVECTOR -DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET)

find_package(Ceres REQUIRED)

add_subdirectory(app)


