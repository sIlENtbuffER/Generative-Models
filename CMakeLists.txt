cmake_minimum_required(VERSION 3.18)
project(GenerativeModels C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

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

file(GLOB_RECURSE VAE_CUDA_SOURCES CONFIGURE_DEPENDS cuda_impl/src/*.cu)
list(APPEND VAE_CUDA_SOURCES
c_impl/src/core/tensor.c
c_impl/src/data/data.c
c_impl/src/data/mnist.c
c_impl/src/tools/config.c
c_impl/src/tools/checkpoint.c
c_impl/third_party/cjson/cJSON.c
c_impl/third_party/stb/stb_image_write.c
)

add_executable(vae_train_cuda ${VAE_CUDA_SOURCES})
target_include_directories(vae_train_cuda PRIVATE cuda_impl cuda_impl/include c_impl c_impl/include)
target_compile_options(vae_train_cuda PRIVATE -Wall -Wextra)
set_target_properties(vae_train_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(vae_train_cuda PRIVATE m)
