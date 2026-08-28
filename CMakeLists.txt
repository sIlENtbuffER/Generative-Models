cmake_minimum_required(VERSION 3.18)
project(GenerativeModels C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB_RECURSE VAE_SOURCES CONFIGURE_DEPENDS c_impl/src/*.c)
list(APPEND VAE_SOURCES
c_impl/third_party/cjson/cJSON.c
c_impl/third_party/stb/stb_image_write.c
)

add_executable(vae_train_c ${VAE_SOURCES})
target_include_directories(vae_train_c PRIVATE c_impl c_impl/include)
target_compile_options(vae_train_c PRIVATE -Wall -Wextra -Wpedantic $<$<CONFIG:Release>:-march=native>)
target_link_libraries(vae_train_c PRIVATE m)
