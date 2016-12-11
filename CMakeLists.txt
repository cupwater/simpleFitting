project(4dface)
cmake_minimum_required(VERSION 2.8.10)

SET(CMAKE_BUILD_TYPE "Debug")  
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -ggdb -Wall -std=c++14")  
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wall")  


# Find dependencies:
find_package(OpenCV)  #REQUIRED core imgproc highgui objdetect features2d
find_package(Boost 1.48.0 COMPONENTS system filesystem program_options REQUIRED)
find_package(Eigen3 REQUIRED)
include(FindZLIB)


set(superviseddescent_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/superviseddescent/include")
set(glm_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/glm-0.9.7.0")
set(cereal_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/cereal-1.1.1/include")
set(matparse_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/matparse")

# Add header includes:
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${superviseddescent_INCLUDE_DIR})
include_directories(${glm_INCLUDE_DIR})
include_directories(${cereal_INCLUDE_DIR})
include_directories(${matparse_INCLUDE_DIR})
include_directories(SYSTEM ${ZLIB_INCLUDE_DIR})

add_executable(myFitting simpleFitting.cpp)
target_link_libraries(myFitting ${OpenCV_LIBS} ${ZLIB_LIBRARIES})
