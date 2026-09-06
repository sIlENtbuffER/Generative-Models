cmake_minimum_required(VERSION 3.18)
project(GenerativeModels C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB_RECURSE GENMODELS_SOURCES CONFIGURE_DEPENDS c_impl/src/*.c)
list(APPEND GENMODELS_SOURCES
c_impl/third_party/cjson/cJSON.c
c_impl/third_party/stb/stb_image_write.c
)

add_executable(genmodels_train_c ${GENMODELS_SOURCES})
target_include_directories(genmodels_train_c PRIVATE c_impl c_impl/include)
target_compile_options(genmodels_train_c PRIVATE -Wall -Wextra -Wpedantic $<$<CONFIG:Release>:-march=native>)
target_link_libraries(genmodels_train_c PRIVATE m)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES native)

    file(GLOB_RECURSE GENMODELS_CUDA_SOURCES CONFIGURE_DEPENDS cuda_impl/src/*.cu)
    file(GLOB GENMODELS_CUDA_C_SOURCES CONFIGURE_DEPENDS
    c_impl/src/core/*.c
    c_impl/src/data/*.c
    c_impl/src/tools/*.c
    )
    list(REMOVE_ITEM GENMODELS_CUDA_C_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/c_impl/src/core/rng.c)
    list(APPEND GENMODELS_CUDA_SOURCES ${GENMODELS_CUDA_C_SOURCES}
    c_impl/third_party/cjson/cJSON.c
    c_impl/third_party/stb/stb_image_write.c
    )

    add_executable(genmodels_train_cuda ${GENMODELS_CUDA_SOURCES})
    target_include_directories(genmodels_train_cuda PRIVATE cuda_impl cuda_impl/include c_impl c_impl/include)
    target_compile_options(genmodels_train_cuda PRIVATE -Wall -Wextra)
    set_target_properties(genmodels_train_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(genmodels_train_cuda PRIVATE m)
else()
    message(STATUS "No CUDA compiler found - skipping cuda build")
endif()
