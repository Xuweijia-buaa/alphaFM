cmake_minimum_required(VERSION 3.19)
project(alphaFM)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/Frame)
include_directories(src/FTRL)
include_directories(src/Lock)
include_directories(src/Mem)
include_directories(src/Sample)
include_directories(src/Utils)

add_executable(alphaFM
        src/Frame/pc_frame.cpp
        src/Utils/utils.cpp
        fm_train.cpp)
