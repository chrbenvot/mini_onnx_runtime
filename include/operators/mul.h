#pragma once
#include "../operator.h"
#include <algorithm>
#include <vector>

class MulOp : public Operator {
public:
    void forward(const std::vector<Tensor*>& inputs, 
                 std::vector<Tensor*>& outputs, 
                 const onnx::NodeProto& node,std::vector<float>& workspace) override {
        
        const Tensor* A = inputs[0];
        const Tensor* B = inputs[1];
        Tensor* Y = outputs[0];
        // Fast Path with SIMD
        if (A->shape() == B->shape()) {
            Y->reshape(A->shape());
            mul_avx(A->data<float>(), B->data<float>(), Y->data<float>(), A->size());
            return;
        }

        // 1. Determine Output Shape (Max of dims)
        // We align dimensions to the right (NumPy style)
        int rank_a = A->shape().size();
        int rank_b = B->shape().size();
        int max_rank = std::max(rank_a, rank_b);
        
        std::vector<int64_t> out_shape(max_rank);
        std::vector<int64_t> pad_shape_a(max_rank);
        std::vector<int64_t> pad_shape_b(max_rank);

        // Align shapes to the right by padding with 1s
        for (int i = 0; i < max_rank; ++i) {
            // Index from the end (right side)
            int idx_out = max_rank - 1 - i;
            int idx_a   = rank_a - 1 - i;
            int idx_b   = rank_b - 1 - i;

            int64_t dim_a = (idx_a >= 0) ? A->shape()[idx_a] : 1;
            int64_t dim_b = (idx_b >= 0) ? B->shape()[idx_b] : 1;

            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                std::cerr << "Error: Incompatible broadcast shapes in AddOp." << std::endl;
                return;
            }

            out_shape[idx_out]   = std::max(dim_a, dim_b);
            pad_shape_a[idx_out] = dim_a;
            pad_shape_b[idx_out] = dim_b;
        }

        Y->reshape(out_shape);

        // 2. Compute "Virtual Strides"
        // This tells us how much to jump in the raw buffer for each output dimension.
        // If a dimension is 1, the stride is 0 (we repeat the same value).
        std::vector<int64_t> strides_out(max_rank);
        std::vector<int64_t> strides_a(max_rank);
        std::vector<int64_t> strides_b(max_rank);

        int64_t current_stride_out = 1;
        int64_t current_stride_a = 1;
        int64_t current_stride_b = 1;

        // Compute physical strides first (from right to left)
        for (int i = max_rank - 1; i >= 0; --i) {
            strides_out[i] = current_stride_out;
            
            // For A: If we padded it, or if dim is 1, stride is effectively 0 for broadcast
            int64_t dim_a = pad_shape_a[i];
            strides_a[i] = (dim_a == 1) ? 0 : current_stride_a; 
            // Only update physical stride if this dimension actually exists in A
            if (i >= (max_rank - rank_a)) current_stride_a *= A->shape()[i - (max_rank - rank_a)];

            // For B: Same logic
            int64_t dim_b = pad_shape_b[i];
            strides_b[i] = (dim_b == 1) ? 0 : current_stride_b;
            if (i >= (max_rank - rank_b)) current_stride_b *= B->shape()[i - (max_rank - rank_b)];

            current_stride_out *= out_shape[i];
        }

        const float* a_ptr = A->data<float>();
        const float* b_ptr = B->data<float>();
        float* y_ptr = Y->data<float>();
        int64_t total_elements = Y->size();

        // 3. The Loop (Flattened)
        // We iterate 0..total_elements and reconstruct coordinates
        for (int64_t i = 0; i < total_elements; ++i) {
            int64_t idx_a = 0;
            int64_t idx_b = 0;
            int64_t temp_i = i;

            // Map flat output index 'i' to input indices 'idx_a' and 'idx_b'
            for (int dim = 0; dim < max_rank; ++dim) {
                // Recover coordinate for this dimension
                int64_t coord = temp_i / strides_out[dim];
                temp_i %= strides_out[dim];

                idx_a += coord * strides_a[dim];
                idx_b += coord * strides_b[dim];
            }

            y_ptr[i] = a_ptr[idx_a] * b_ptr[idx_b];
        }
    }
};
