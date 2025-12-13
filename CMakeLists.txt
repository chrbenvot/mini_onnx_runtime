cmake_minimum_required(VERSION 3.18)  # >=3.18 for CUDA targets support

project(OnnxEngine LANGUAGES CXX CUDA)

# -----------------------------
# 1. C++ Standard
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# -----------------------------
# 2. CUDA settings
# -----------------------------
# Set target architectures (compute capability)
set(CMAKE_CUDA_ARCHITECTURES 75 86)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>) # Enable relocatable device code

# -----------------------------
# 3. Find Libraries
# -----------------------------
find_package(Protobuf REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS cublas)

# -----------------------------
# 4. Include directories
# -----------------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# -----------------------------
# 5. Protobuf ONNX generation
# -----------------------------
set(ONNX_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx)
protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS ${ONNX_PROTO_DIR}/onnx.proto)
add_library(onnx_proto ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
target_link_libraries(onnx_proto PRIVATE ${PROTOBUF_LIBRARIES})

# -----------------------------
# 6. Compile options for CPU SIMD
# -----------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-mavx2 -mfma -O3)
elseif(MSVC)
    add_compile_options(/arch:AVX2 /O2)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr")

# Automatically collect all .cu files in src/operators
file(GLOB_RECURSE OPERATOR_CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/*.cu")
# -----------------------------
# 7. Engine target
# -----------------------------
add_executable(engine
    src/main.cpp
    ${OPERATOR_CU_FILES}

)
set_target_properties(engine PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(engine
    onnx_proto
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
)

target_include_directories(engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# -----------------------------
# 8. Unit test target
# -----------------------------
add_executable(test_tensor
    tests/test_tensor.cpp ${OPERATOR_CU_FILES}
)

target_link_libraries(test_tensor
    onnx_proto
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
)

target_include_directories(test_tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# cuda test target
add_executable(test_cuda tests/test_cuda.cpp ${OPERATOR_CU_FILES})

# CRITICAL: Tell CMake to use the CUDA compiler and link necessary libs
# We also need to include your custom headers and the CUDA headers
target_link_libraries(test_cuda onnx_proto  CUDA::cudart CUDA::cublas)
target_include_directories(test_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# cue gemm test target
add_executable(test_cuda_gemm tests/test_cuda_gemm.cpp src/operators/gemm.cu ${ONNX_PROTO_SRCS})
    
# Link CUDA Runtime and cuBLAS
target_link_libraries(test_cuda_gemm 
        ${PROTOBUF_LIBRARIES} 
        CUDA::cudart 
        CUDA::cublas
)
    
target_include_directories(test_cuda_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
