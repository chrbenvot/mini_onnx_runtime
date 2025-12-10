#pragma once
#include "../operator.h"

class GemmOp : public Operator {
public:
    void forward(const std::vector<Tensor*>& inputs, 
                 std::vector<Tensor*>& outputs, 
                 const onnx::NodeProto& node) override {
        
        const Tensor* A = inputs[0];
        const Tensor* B = inputs[1];
        const Tensor* C = (inputs.size() > 2) ? inputs[2] : nullptr;
        Tensor* Y = outputs[0];

        //  Get Attributes
        float alpha = get_float_attribute(node, "alpha", 1.0f);
        float beta  = get_float_attribute(node, "beta", 1.0f);
        int64_t transA = get_int_attribute(node, "transA", 0);
        int64_t transB = get_int_attribute(node, "transB", 0);

        //  Determine Dimensions
        // A is [M, K] normally, or [K, M] if transposed
        int64_t M = (transA == 0) ? A->shape()[0] : A->shape()[1];
        int64_t K = (transA == 0) ? A->shape()[1] : A->shape()[0];
        
        // B is [K, N] normally, or [N, K] if transposed
        int64_t N = (transB == 0) ? B->shape()[1] : B->shape()[0];
        
        // Quick verify K matches
        int64_t K_check = (transB == 0) ? B->shape()[0] : B->shape()[1];
        if (K != K_check) {
            std::cerr << "Error: Gemm Dimension Mismatch K=" << K << " vs " << K_check << std::endl;
            return;
        }

        Y->reshape({M, N});

        const float* a_data = A->data<float>();
        const float* b_data = B->data<float>();
        const float* c_data = (C) ? C->data<float>() : nullptr;
        float* y_data = Y->data<float>();

        //  Stride Logic
        // We need to know how much to jump in the RAW buffer to get to the next element.
        // Raw Shape A: [Dim0, Dim1]
        int64_t a_stride_0 = A->shape()[1]; // Jump one row
        int64_t a_stride_1 = 1;             // Jump one col

        int64_t b_stride_0 = B->shape()[1];
        int64_t b_stride_1 = 1;

        //  The Loop
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                
                float sum = 0.0f;

                for (int k = 0; k < K; ++k) {
                    // Logic:
                    // If TransA=0: We want A[m, k]. Raw Index = m * stride0 + k * stride1
                    // If TransA=1: We want A[k, m]. Raw Index = k * stride0 + m * stride1
                    
                    int64_t a_idx = (transA == 0) 
                                  ? (m * a_stride_0 + k * a_stride_1)
                                  : (k * a_stride_0 + m * a_stride_1); // <--- Fixed Stride Logic

                    int64_t b_idx = (transB == 0)
                                  ? (k * b_stride_0 + n * b_stride_1)
                                  : (n * b_stride_0 + k * b_stride_1); // <--- Fixed Stride Logic
                    
                    sum += a_data[a_idx] * b_data[b_idx];
                }

                // Apply Alpha
                sum *= alpha;

                // Apply Beta * C
                if (c_data) {
                    // C can be [M, N] or broadcasted [1, N] or [M, 1] or [1] or [N]
                    // Simplifying assuming C is [M, N] or [N] (common bias)
                    int64_t c_idx = 0;
                    if (C->size() == M * N) {
                        c_idx = m * N + n;
                    } else if (C->size() == N) { // Standard Bias [N]
                        c_idx = n;
                    } else if (C->size() == 1) { // Scalar
                        c_idx = 0;
                    }
                    
                    sum += beta * c_data[c_idx];
                }

                y_data[m * N + n] = sum;
            }
        }
    }
};
