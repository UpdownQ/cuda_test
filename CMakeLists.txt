cmake_minimum_required(VERSION 3.10)
project(test_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL)
find_package(Eigen)


include_directories(include ${PCL_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} src/test_cuda.cu)
target_link_libraries(${PROJECT_NAME})

add_executable(matrix_math src/matrix_math.cu)
target_link_libraries(matrix_math ${PCL_LIBRARIES} ${EIGEN_LIBRARIES})
