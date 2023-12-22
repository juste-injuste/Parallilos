cmake_minimum_required(VERSION 3.5)

project(Parallilos)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -g3 -Wall -Wextra -Wconversion -Wpedantic -pedantic -Wcast-align -Wcast-qual -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs -Wredundant-decls -Wshadow -Wstrict-aliasing=2 -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused-parameter")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
set(PARALLILOS_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(Parallilos ${PARALLILOS_SOURCES_DIR}/examples.cpp)