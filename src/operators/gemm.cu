#include "operators/gemm.h"
#include <cublas_v2.h>

void GemmOp::forward_gpu(const std::vector<Tensor*>& inputs, 
                         std::vector<Tensor*>& outputs, 
                         const onnx::NodeProto& node, 
                         cublasHandle_t& handle) {
    
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    const Tensor* C_bias = (inputs.size() > 2) ? inputs[2] : nullptr;
    Tensor* Y = outputs[0];

    // 1. Get Attributes
    float alpha = get_float_attribute(node, "alpha", 1.0f);
    float beta  = get_float_attribute(node, "beta", 1.0f);
    int transA = get_int_attribute(node, "transA", 0);
    int transB = get_int_attribute(node, "transB", 0);

    // 2. Dimensions (Standard Logic)
    
    int M = (transA == 0) ? A->shape()[0] : A->shape()[1];
    int K = (transA == 0) ? A->shape()[1] : A->shape()[0];
    int N = (transB == 0) ? B->shape()[1] : B->shape()[0];

    Y->reshape({M, N});
    Y->free_device_memory(); // Safety clear
    Y->allocate_device_memory();

    // 3. Pointers (Device Memory!)
    const float* d_A = (const float*)A->device_data();
    const float* d_B = (const float*)B->device_data();
    float* d_Y = (float*)Y->device_data();

    // 4. cuBLAS GEMM
    // We want Row-Major C = A * B.
    // cuBLAS is Col-Major. We trick it by computing C' = B' * A'
    // This means we treat pointers as if they are transposed.
    
    // Operation logic:
    // If standard C++ TransA=0, cuBLAS sees it as Transposed.
    // So we invoke CUBLAS_OP_N (No Transpose) if we WANT it transposed (to cancel out the implicit transpose).
    // This double-negative logic is confusing. 
    
    // SIMPLIFIED TRICK: 
    // standard_sgemm(M, N, K, A, B) -> cublas_sgemm(N, M, K, B, A)
    
    cublasOperation_t opA = (transA) ? CUBLAS_OP_T : CUBLAS_OP_N; 
    cublasOperation_t opB = (transB) ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Note args are swapped: B comes before A, N comes before M
    cublasSgemm(handle, 
                opB, opA, 
                N, M, K, 
                &alpha, 
                d_B, (transB ? K : N), // Lead dim of B
                d_A, (transA ? M : K), // Lead dim of A
                &beta, 
                d_Y, N);               // Lead dim of C
}
