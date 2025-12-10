#pragma once
#include "../operator.h"
#include <algorithm>
#include <vector>
#include "../simd_utils.h" // Include SIMD helper

class AddOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node, std::vector<float> &workspace) override
    {

        const Tensor *A = inputs[0];
        const Tensor *B = inputs[1];
        Tensor *Y = outputs[0];

        // --- OPTIMIZATION: Fast Path for Element-Wise Add ---
        // If shapes are identical, no broadcasting is needed.
        // We can just blast through the arrays with AVX.
        if (A->shape() == B->shape()) {
            Y->reshape(A->shape());
            
            const float* a_ptr = A->data<float>();
            const float* b_ptr = B->data<float>();
            float* y_ptr = Y->data<float>();
            
            // Use the SIMD helper from simd_utils.h
            add_avx(a_ptr, b_ptr, y_ptr, A->size());
            return;
        }
        // ----------------------------------------------------

        // 1. Determine Output Shape (Max of dims)
        int rank_a = A->shape().size();
        int rank_b = B->shape().size();
        int max_rank = std::max(rank_a, rank_b);

        std::vector<int64_t> out_shape(max_rank);
        std::vector<int64_t> pad_shape_a(max_rank);
        std::vector<int64_t> pad_shape_b(max_rank);

        // Align shapes to the right by padding with 1s
        for (int i = 0; i < max_rank; ++i)
        {
            int idx_out = max_rank - 1 - i;
            int idx_a = rank_a - 1 - i;
            int idx_b = rank_b - 1 - i;

            int64_t dim_a = (idx_a >= 0) ? A->shape()[idx_a] : 1;
            int64_t dim_b = (idx_b >= 0) ? B->shape()[idx_b] : 1;

            if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
            {
                std::cerr << "Error: Incompatible broadcast shapes in AddOp." << std::endl;
                return;
            }

            out_shape[idx_out] = std::max(dim_a, dim_b);
            pad_shape_a[idx_out] = dim_a;
            pad_shape_b[idx_out] = dim_b;
        }

        Y->reshape(out_shape);

        // 2. Compute "Virtual Strides"
        std::vector<int64_t> strides_out(max_rank);
        std::vector<int64_t> strides_a(max_rank);
        std::vector<int64_t> strides_b(max_rank);

        int64_t current_stride_out = 1;
        int64_t current_stride_a = 1;
        int64_t current_stride_b = 1;

        for (int i = max_rank - 1; i >= 0; --i)
        {
            strides_out[i] = current_stride_out;

            int64_t dim_a = pad_shape_a[i];
            strides_a[i] = (dim_a == 1) ? 0 : current_stride_a;
            if (i >= (max_rank - rank_a))
                current_stride_a *= A->shape()[i - (max_rank - rank_a)];

            int64_t dim_b = pad_shape_b[i];
            strides_b[i] = (dim_b == 1) ? 0 : current_stride_b;
            if (i >= (max_rank - rank_b))
                current_stride_b *= B->shape()[i - (max_rank - rank_b)];

            current_stride_out *= out_shape[i];
        }

        const float *a_ptr = A->data<float>();
        const float *b_ptr = B->data<float>();
        float *y_ptr = Y->data<float>();
        int64_t total_elements = Y->size();

        // 3. The Loop (Flattened)
        for (int64_t i = 0; i < total_elements; ++i)
        {
            int64_t idx_a = 0;
            int64_t idx_b = 0;
            int64_t temp_i = i;

            for (int dim = 0; dim < max_rank; ++dim)
            {
                int64_t coord = temp_i / strides_out[dim];
                temp_i %= strides_out[dim];

                idx_a += coord * strides_a[dim];
                idx_b += coord * strides_b[dim];
            }

            y_ptr[i] = a_ptr[idx_a] + b_ptr[idx_b];
        }
    }
};
