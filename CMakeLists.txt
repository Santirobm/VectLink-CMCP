cmake_minimum_required(VERSION 3.16)
project(vectlink_cmcp VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(VECTLINK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Vect-Link" CACHE PATH "Path to Vect-Link repo")
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" CACHE PATH "Path to llama.cpp repo")
option(VCMCP_BUILD_KV_BRIDGE "Build llama.cpp + Vect-Link KV reuse bridge" ON)
option(VCMCP_BUILD_TESTS "Build C unit tests" ON)

find_library(VECTLINK_IMPORT_LIB
    NAMES vectlink libvectlink.dll.a libvectlink
    PATHS "${VECTLINK_DIR}/build"
    NO_DEFAULT_PATH
)
if(NOT VECTLINK_IMPORT_LIB)
    message(FATAL_ERROR "Could not find Vect-Link library in ${VECTLINK_DIR}/build")
endif()

if(VCMCP_BUILD_TESTS)
    enable_testing()
    add_executable(test_vl_cmcp_c_api tests/c/test_vl_cmcp_c_api.c)
    target_include_directories(test_vl_cmcp_c_api PRIVATE "${VECTLINK_DIR}/include")
    target_link_libraries(test_vl_cmcp_c_api PRIVATE "${VECTLINK_IMPORT_LIB}")
    if(WIN32)
        target_link_libraries(test_vl_cmcp_c_api PRIVATE psapi)
    endif()
    add_test(NAME test_vl_cmcp_c_api COMMAND test_vl_cmcp_c_api)
endif()

if(VCMCP_BUILD_KV_BRIDGE)
    find_library(LLAMA_IMPORT_LIB
        NAMES llama libllama.dll.a libllama
        PATHS "${LLAMA_DIR}/build/src" "${LLAMA_DIR}/build/bin" "${LLAMA_DIR}/build"
        NO_DEFAULT_PATH
    )
    if(NOT LLAMA_IMPORT_LIB)
        message(FATAL_ERROR "Could not find libllama in ${LLAMA_DIR}/build")
    endif()

    add_executable(vl_cmcp_kv_bridge csrc/vl_cmcp_kv_bridge.c)
    target_include_directories(vl_cmcp_kv_bridge PRIVATE
        "${VECTLINK_DIR}/include"
        "${LLAMA_DIR}/include"
        "${LLAMA_DIR}/ggml/include"
    )
    target_link_libraries(vl_cmcp_kv_bridge PRIVATE "${VECTLINK_IMPORT_LIB}" "${LLAMA_IMPORT_LIB}")
    if(WIN32)
        target_link_libraries(vl_cmcp_kv_bridge PRIVATE psapi)
    else()
        target_link_libraries(vl_cmcp_kv_bridge PRIVATE m)
    endif()
endif()
