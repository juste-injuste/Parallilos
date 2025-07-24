cmake_minimum_required(VERSION 3.5)
project(Library)

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type for single-config generators (e.g. Ninja, MinGW)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Default build type" FORCE)
endif()

# === Output to project root ===
set(FINAL_OUTPUT_DIR "${CMAKE_SOURCE_DIR}")

# Multi-config generators (e.g., MSVC)
if(CMAKE_CONFIGURATION_TYPES)
  foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${FINAL_OUTPUT_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${FINAL_OUTPUT_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${FINAL_OUTPUT_DIR})
  endforeach()
endif()

# Single-config generators (e.g., MinGW, Ninja, Make)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${FINAL_OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${FINAL_OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${FINAL_OUTPUT_DIR})

# === Compiler flags ===
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive- /W4 /WX /O2 /GL /Ob2 /Oi /Ot /fp:fast")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GF /Gy /Zi /Zc:__cplusplus /Zc:preprocessor /MP /wd4996 /sdl")
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)

  foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${cfg} upper_cfg)
    set(CMAKE_EXE_LINKER_FLAGS_${upper_cfg} "/LTCG /DEBUG /INCREMENTAL:NO")
  endforeach()
else() # GCC / Clang
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -march=native -Ofast")
  #  -DNDEBUG -g3 -fno-omit-frame-pointer")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -Wpedantic -pedantic")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align -Wcast-qual -Wformat=2 -Winit-self")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wlogical-op -Wmissing-include-dirs -Wredundant-decls")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Wstrict-aliasing=2 -Wstrict-overflow=5 -Wswitch-default")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wundef -Wno-unused-parameter -Wdeprecated")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference -Wduplicated-cond -Wduplicated-branches")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wlogical-not-parentheses -Wzero-as-null-pointer-constant")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuseless-cast -Wdouble-promotion -Wold-style-cast")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsuggest-override")
endif()

# === Include and source ===
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
set(LIBRARY_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples/)



add_executable(PARALLILOS  ${LIBRARY_SOURCES_DIR}/main.cpp)
