project(test_ceres)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fPIC")

include_directories("../..")

add_executable(test_ceres main.cpp)

