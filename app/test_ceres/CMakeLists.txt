project(test_ceres)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-fPIC")

include_directories("../..")

add_executable(test_ceres main.cpp)
target_link_libraries(test_ceres Ceres::ceres)

