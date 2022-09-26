cmake_minimum_required(VERSION 3.0)
project(Evaluation)

find_package(OpenCV REQUIRED )
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIR})
add_executable(Evaluation Evaluation.cpp src/imgCompare.cpp src/templateExtractor.cpp)
add_executable(EvaluationORB EvaluationORB.cc src/ORBextractor.cc)
target_link_libraries(Evaluation ${OpenCV_LIBS})
target_link_libraries(EvaluationORB ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so)

