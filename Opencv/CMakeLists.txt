# project(MYSWAP)

# add_executable(my_cmake_swap Helloworld.cpp swap.cpp)
# set(CMAKE_BUILD_TYPE DEBUG)

cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
add_executable( DisplayImage Template_match.cpp)
target_link_libraries( DisplayImage ${OpenCV_LIBS} )