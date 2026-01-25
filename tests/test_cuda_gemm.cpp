#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "tensor.h"
#include "engine.h"
#include "operators/gemm.h"

void test_simple_gemm() {
    std::cout << "--- Testing GEMM on GPU (cuBLAS) ---" << std::endl;

    // Initialize Engine (Creating cuBLAS Handle)
    InferenceEngine engine; 

    //  Prepare Data (A * B = Y)
    // Matrix A (2x2): [[1, 2], [3, 4]]
    Tensor A(DataType::FLOAT32, {2, 2});
    float* a_ptr = A.data<float>();
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f;
    a_ptr[2] = 3.0f; a_ptr[3] = 4.0f;

    // Matrix B (2x2 Identity): [[1, 0], [0, 1]]
    Tensor B(DataType::FLOAT32, {2, 2});
    float* b_ptr = B.data<float>();
    b_ptr[0] = 1.0f; b_ptr[1] = 0.0f;
    b_ptr[2] = 0.0f; b_ptr[3] = 1.0f;

    // Output Y
    Tensor Y(DataType::FLOAT32, {2, 2});

    //  Move to GPU
    std::cout << "  Moving data to GPU..." << std::endl;
    A.allocate_device_memory(); A.copy_to_device();
    B.allocate_device_memory(); B.copy_to_device();
    Y.allocate_device_memory(); // Allocate output space

    //  Run Operator
    std::cout << "  Running cuBLAS GEMM..." << std::endl;
    GemmOp op;
    std::vector<Tensor*> inputs = {&A, &B};
    std::vector<Tensor*> outputs = {&Y};
    onnx::NodeProto node; // Default attributes (alpha=1, beta=1, trans=0)

    // We need to manually access the handle from the engine for this raw test
    // Friend classes or a getter would be cleaner, but for testing we can cheat 
    // OR just create a local handle if we can't access private members.
    
    // For this test script ONLY, let's make a local handle to verify the OP logic
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    op.forward_gpu(inputs, outputs, node, handle);
    
    cudaDeviceSynchronize();

    //  Read Back
    std::cout << "  Reading result..." << std::endl;
    Y.copy_to_host();

    //  Verify: A * I = A
    // [[1, 2], [3, 4]]
    float* y_ptr = Y.data<float>();
    std::cout << "  Result: [" << y_ptr[0] << ", " << y_ptr[1] << "]" << std::endl;
    std::cout << "          [" << y_ptr[2] << ", " << y_ptr[3] << "]" << std::endl;

    assert(std::abs(y_ptr[0] - 1.0f) < 0.001);
    assert(std::abs(y_ptr[1] - 2.0f) < 0.001);
    assert(std::abs(y_ptr[2] - 3.0f) < 0.001);
    assert(std::abs(y_ptr[3] - 4.0f) < 0.001);

    // Cleanup
    cublasDestroy(handle);
    A.free_device_memory();
    B.free_device_memory();
    Y.free_device_memory();

    std::cout << "  -> Passed!" << std::endl;
}

int main() {
    test_simple_gemm();
    return 0;
}
