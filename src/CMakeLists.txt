add_library(cuda_image_filtering_lib STATIC
  convert.cu
  gaussian_blur.cu
  utils.cc
)

target_include_directories(cuda_image_filtering_lib
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(cuda_image_filtering_lib
  PUBLIC
  CUDA::cudart
)

target_compile_options(cuda_image_filtering_lib
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -pedantic>
)

target_compile_definitions(cuda_image_filtering_lib
  PRIVATE
  $<$<BOOL:${ENABLE_LOGGING}>:ENABLE_LOGGING>
)

set_target_properties(cuda_image_filtering_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RUNTIME_LIBRARY Shared
  POSITION_INDEPENDENT_CODE ON
)

install(TARGETS cuda_image_filtering_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
