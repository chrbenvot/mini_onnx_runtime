#pragma once
#include "../operator.h"
#include <vector>
#include <cmath>     // For std::ceil
#include <algorithm> // For std::max, std::min

class AvgPoolOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node,std::vector<float>& workspace) override
    {
        const Tensor *X = inputs[0];
        Tensor *Y = outputs[0];

        // 1. Parse Attributes
        auto kernel_shape = get_int_list_attribute(node, "kernel_shape");
        int64_t kern_h = kernel_shape[0];
        int64_t kern_w = kernel_shape[1];

        auto strides = get_int_list_attribute(node, "strides");
        int64_t stride_h = (strides.empty()) ? 1 : strides[0];
        int64_t stride_w = (strides.empty()) ? 1 : strides[1];

        // Parse 4 pad values [y_begin, x_begin, y_end, x_end]
        auto pads = get_int_list_attribute(node, "pads");
        int64_t pad_h_begin = (pads.size() >= 4) ? pads[0] : 0;
        int64_t pad_w_begin = (pads.size() >= 4) ? pads[1] : 0;
        int64_t pad_h_end = (pads.size() >= 4) ? pads[2] : 0;
        int64_t pad_w_end = (pads.size() >= 4) ? pads[3] : 0;

        // Parse logic flags
        int64_t count_include_pad = get_int_attribute(node, "count_include_pad", 0);
        int64_t ceil_mode = get_int_attribute(node, "ceil_mode", 0);

        const auto &in_shape = X->shape();
        int64_t N = in_shape[0];
        int64_t C = in_shape[1];
        int64_t H = in_shape[2];
        int64_t W = in_shape[3];

        // Calculate Output Dims with Ceil support and Asymmetric Pads
        int64_t out_h, out_w;
        if (ceil_mode != 0)
        {
            out_h = std::ceil((float)(H + pad_h_begin + pad_h_end - kern_h) / stride_h) + 1;
            out_w = std::ceil((float)(W + pad_w_begin + pad_w_end - kern_w) / stride_w) + 1;
        }
        else
        {
            out_h = (H + pad_h_begin + pad_h_end - kern_h) / stride_h + 1;
            out_w = (W + pad_w_begin + pad_w_end - kern_w) / stride_w + 1;
        }

        Y->reshape({N, C, out_h, out_w});

        const float *X_data = X->data<float>();
        float *y_data = Y->data<float>();

        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int oy = 0; oy < out_h; ++oy)
                {
                    for (int ox = 0; ox < out_w; ++ox)
                    {

                        float sum = 0.0f;

                        // Coordinates of the kernel window top-left
                        int64_t h_start = oy * stride_h - pad_h_begin;
                        int64_t w_start = ox * stride_w - pad_w_begin;
                        int64_t h_end = h_start + kern_h;
                        int64_t w_end = w_start + kern_w;

                        // Clamp to actual image boundaries for the sum loop
                        int64_t h_start_valid = std::max(h_start, (int64_t)0);
                        int64_t w_start_valid = std::max(w_start, (int64_t)0);
                        int64_t h_end_valid = std::min(h_end, H);
                        int64_t w_end_valid = std::min(w_end, W);

                        for (int y = h_start_valid; y < h_end_valid; ++y)
                        {
                            for (int x = w_start_valid; x < w_end_valid; ++x)
                            {
                                int64_t idx = n * (C * H * W) + c * (H * W) + y * W + x;
                                sum += X_data[idx];
                            }
                        }

                        // Divisor Logic
                        float divisor;
                        if (count_include_pad)
                        {
                            // Divide by fixed kernel area (counting zeros in padding)
                            divisor = static_cast<float>(kern_h * kern_w);
                        }
                        else
                        {
                            // Divide only by valid pixels (intersection of kernel and image)
                            // This prevents edges from becoming artificially dark
                            int64_t valid_h = h_end_valid - h_start_valid;
                            int64_t valid_w = w_end_valid - w_start_valid;
                            // Avoid division by zero (though output size calc should prevent this)
                            divisor = static_cast<float>(std::max(valid_h * valid_w, (int64_t)1));
                        }

                        int64_t out_idx = n * (C * out_h * out_w) + c * (out_h * out_w) + oy * out_w + ox;
                        y_data[out_idx] = sum / divisor;
                    }
                }
            }
        }
    }
};
