project(test_features)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-fPIC")

include_directories("../..")

add_executable(test_features test_features.cpp)
target_link_libraries(test_features ${OpenCV_LIBS})

add_executable(test_matching test_matching.cpp)
target_link_libraries(test_matching ${OpenCV_LIBS})


