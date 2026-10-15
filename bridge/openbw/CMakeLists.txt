# Builds liburanium_bridge.so = thin BWAPI shim (C++) + Rust staticlib (spec 011).
# Invoked by tools/openbw/build_bridge.sh, which builds the Rust part first.
cmake_minimum_required(VERSION 3.16)
project(uranium_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BWAPI_DIR "${CMAKE_CURRENT_LIST_DIR}/../../third_party/bwapi" CACHE PATH "BWAPI (OpenBW fork) source dir")
set(BWAPI_BUILD_DIR "${BWAPI_DIR}/build" CACHE PATH "BWAPI build dir")
set(RUST_STATICLIB "" CACHE FILEPATH "Path to liburanium_bridge_openbw.a")
if(NOT RUST_STATICLIB)
  message(FATAL_ERROR "RUST_STATICLIB not set")
endif()

add_library(uranium_bridge SHARED shim/shim.cpp)
set_target_properties(uranium_bridge PROPERTIES PREFIX "lib" CXX_VISIBILITY_PRESET hidden)
target_compile_options(uranium_bridge PRIVATE -include cstdint -Wall -Wno-unused-parameter)
target_include_directories(uranium_bridge PRIVATE include "${BWAPI_DIR}/bwapi/include")
target_link_directories(uranium_bridge PRIVATE "${BWAPI_BUILD_DIR}/lib")
target_link_libraries(uranium_bridge PRIVATE BWAPILIB "${RUST_STATICLIB}" pthread dl m)
# Rust staticlib symbols must not leak: export only gameInit/newAIModule.
target_link_options(uranium_bridge PRIVATE "-Wl,--exclude-libs,ALL")
