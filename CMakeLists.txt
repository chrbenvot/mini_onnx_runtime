cmake_minimum_required(VERSION 3.10)

# 1. Name your project
project(OnnxEngine)

# 2. Set C++ Standard (We need C++17 for modern features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3. Find External Libraries (Protobuf is required for ONNX)
find_package(Protobuf REQUIRED)

# 4. Include Directories (Where are your .h files?)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include          # Your headers (tensor.h)
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx # The onnx.proto file
    ${CMAKE_CURRENT_BINARY_DIR}                  # Where generated ONNX files land
    ${PROTOBUF_INCLUDE_DIRS}                     # Protobuf headers
)

# 5. Generate ONNX C++ files from onnx.proto
# This runs the 'protoc' command automatically for you
set(ONNX_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx)
set(ONNX_PROTO_FILE ${ONNX_PROTO_DIR}/onnx.proto)

protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS ${ONNX_PROTO_FILE})

# --------------------------------------------------------
# Target 1: The Main Engine Executable
# --------------------------------------------------------
add_executable(engine 
    src/main.cpp 
    ${ONNX_PROTO_SRCS} # Include the generated .pb.cc file
)

# Link the executable to the Protobuf library
target_link_libraries(engine ${PROTOBUF_LIBRARIES})

# --------------------------------------------------------
# Target 2: The Unit Test
# --------------------------------------------------------
add_executable(test_tensor tests/test_tensor.cpp)

# (Optional) Link libraries to test if needed later
# target_link_libraries(test_tensor ${PROTOBUF_LIBRARIES})
