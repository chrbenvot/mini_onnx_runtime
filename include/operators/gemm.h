#pragma once
#include "../operator.h"
#include "../simd_utils.h" 

class GemmOp : public Operator {
public:
    void forward(const std::vector<Tensor*>& inputs, 
                 std::vector<Tensor*>& outputs, 
                 const onnx::NodeProto& node) override {
        
        const Tensor* A = inputs[0];
        const Tensor* B = inputs[1];
        const Tensor* C = (inputs.size() > 2) ? inputs[2] : nullptr;
        Tensor* Y = outputs[0];

        // 1. Get Attributes
        float alpha = get_float_attribute(node, "alpha", 1.0f);
        float beta  = get_float_attribute(node, "beta", 1.0f);
        int64_t transA = get_int_attribute(node, "transA", 0);
        int64_t transB = get_int_attribute(node, "transB", 0);

        // 2. Dimensions
        int64_t M = (transA == 0) ? A->shape()[0] : A->shape()[1];
        int64_t K = (transA == 0) ? A->shape()[1] : A->shape()[0];
        int64_t N = (transB == 0) ? B->shape()[1] : B->shape()[0];

        Y->reshape({M, N});

        const float* a_data = A->data<float>();
        const float* b_data = B->data<float>();
        const float* c_data = (C) ? C->data<float>() : nullptr;
        float* y_data = Y->data<float>();

        // 3. Stride Calculation (Fallback Logic)
        int64_t a_stride_0 = A->shape()[1];
        int64_t b_stride_0 = B->shape()[1];

        // 4. The Loop
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                
                float sum = 0.0f;

                // --- OPTIMIZATION START ---
                // We check for the "Happy Path":
                // 1. TransA=0: We read Row 'm' of A. (Contiguous if stride=1)
                // 2. TransB=1: We read Row 'n' of raw B (which is Col 'n' of Transposed B). (Contiguous!)
                
                if (transA == 0 && transB == 1) {
                    // Fast Path: AVX Dot Product
                    const float* a_ptr = a_data + m * K; // Start of Row m
                    const float* b_ptr = b_data + n * K; // Start of Row n (in raw B)
                    
                    sum = dot_product_avx(a_ptr, b_ptr, K);
                } 
                else {
                    // Slow Path: Scalar Loop with complex strides
                    for (int k = 0; k < K; ++k) {
                        int64_t a_idx = (transA == 0) ? (m * a_stride_0 + k) : (k * a_stride_0 + m);
                        int64_t b_idx = (transB == 0) ? (k * b_stride_0 + n) : (n * b_stride_0 + k);
                        sum += a_data[a_idx] * b_data[b_idx];
                    }
                }
                // --- OPTIMIZATION END ---

                sum *= alpha;

                // Bias logic
                if (c_data) {
                    int64_t c_idx = 0;
                    if (C->size() == M * N) c_idx = m * N + n;
                    else if (C->size() == N) c_idx = n;
                    // else scalar 0
                    sum += beta * c_data[c_idx];
                }

                y_data[m * N + n] = sum;
            }
        }
    }
};
