cmake_minimum_required(VERSION 3.5)

project(Parallilos)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -std=c++11 -O3 -g3 -Wall -Wextra -Wconversion -Wpedantic -pedantic -Wdisabled-optimization -Wstrict-aliasing=2 -Wstrict-overflow=5 -Wmissing-include-dirs")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
set(PARALLILOS_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(Parallilos ${PARALLILOS_SOURCES_DIR}/examples.cpp ${PARALLILOS_SOURCES_DIR}/test.cpp)