#pragma once
#include "../operator.h"
#include <cmath>
#include <algorithm>
#include <iostream>

class UpsampleOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node,std::vector<float>& workspace) override
    {
        const Tensor *X = inputs[0];
        Tensor *Y = outputs[0];

        const auto &in_shape = X->shape(); // [N, C, H, W]
        int64_t N = in_shape[0];
        int64_t C = in_shape[1];
        int64_t H = in_shape[2];
        int64_t W = in_shape[3];

        // Default scales (Identity)
        float scale_h = 1.0f;
        float scale_w = 1.0f;

        // 1. Parse Scales 
        if (inputs.size() > 2 && inputs[2]->size() > 0)
        {
            // Case A: Scaling factors provided
            const Tensor *scales_tensor = inputs[2];
            const float *s_data = scales_tensor->data<float>();
            
            // Scales is usually [N_scale, C_scale, H_scale, W_scale]
            // We assume 4D input, so we look at indices 2 and 3
            if (scales_tensor->size() >= 4) {
                scale_h = s_data[2];
                scale_w = s_data[3];
            } else if (scales_tensor->size() == 2) {
                // Rare 2D case
                scale_h = s_data[0];
                scale_w = s_data[1];
            }
        } 
        else if (inputs.size() > 3 && inputs[3]->size() > 0)
        {
            // Case B: Target sizes provided (Input[3])
            const Tensor* sizes_tensor = inputs[3];
            const int64_t* size_data = sizes_tensor->data<int64_t>(); // Usually INT64
            // Sizes tensor shape matches Input rank (N,C,H,W)
            // We want the new H and W (indices 2 and 3)
            int64_t target_h = size_data[2];
            int64_t target_w = size_data[3];
            
            // Derive scale from target size
            scale_h = static_cast<float>(target_h) / static_cast<float>(H);
            scale_w = static_cast<float>(target_w) / static_cast<float>(W);
        }

        // 2. Output Calculation
        int64_t out_h = static_cast<int64_t>(H * scale_h);
        int64_t out_w = static_cast<int64_t>(W * scale_w);

        Y->reshape({N, C, out_h, out_w});

        const float *x_data = X->data<float>();
        float *y_data = Y->data<float>();

        // Pre-calculate inverse scales to replace division with multiplication (Optimization)
        float inv_scale_h = 1.0f / scale_h;
        float inv_scale_w = 1.0f / scale_w;

        // 3. Execution (Nearest Neighbor "Asymmetric")
        
        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int oy = 0; oy < out_h; ++oy)
                {
                    // Map output Y back to input Y
                    // Using "floor" behavior (standard Asymmetric mode)
                    int64_t iy = static_cast<int64_t>(oy * inv_scale_h);
                    
                    // Clamp to handle floating point errors or edge cases
                    iy = std::min(iy, H - 1); 

                    // Optimization: Calculate row pointers once per row
                    int64_t in_row_offset = n * (C * H * W) + c * (H * W) + iy * W;
                    int64_t out_row_offset = n * (C * out_h * out_w) + c * (out_h * out_w) + oy * out_w;

                    for (int ox = 0; ox < out_w; ++ox)
                    {
                        int64_t ix = static_cast<int64_t>(ox * inv_scale_w);
                        ix = std::min(ix, W - 1);

                        y_data[out_row_offset + ox] = x_data[in_row_offset + ix];
                    }
                }
            }
        }
    }
};
