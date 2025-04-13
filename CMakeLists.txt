cmake_minimum_required(VERSION 3.21.2)
project(cuda_image_filtering LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(ENABLE_LOGGING "Enable logging" ON)

cmake_policy(SET CMP0079 NEW)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wold-style-cast")

set(CMAKE_CUDA_ARCHITECTURES "native")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)

add_executable(main main.cu)
target_compile_features(main PRIVATE cxx_std_17)
set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)

target_link_libraries(main
    PRIVATE
    CUDA::cudart
    cuda_image_filtering_lib
)

install(TARGETS main DESTINATION bin)
