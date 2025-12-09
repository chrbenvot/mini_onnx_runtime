#pragma once
#include "../operator.h"
#include <vector> 

class AvgPoolOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node) override
    {
        const Tensor *X = inputs[0];
        Tensor *Y = outputs[0];

        // 1. Parse attributes
        // Kernel shape
        auto kernel_shape = get_int_list_attribute(node, "kernel_shape");
        int64_t kern_h = kernel_shape[0];
        int64_t kern_w = kernel_shape[1];

        // strides
        auto strides = get_int_list_attribute(node, "strides");
        int64_t stride_h = strides[0];
        int64_t stride_w = strides[1];

        // pads with 0 as default
        auto pads = get_int_list_attribute(node, "pads");
        int64_t pad_h = (pads.empty()) ? 0 : pads[0];
        int64_t pad_w = (pads.empty()) ? 0 : pads[1];

        // 2. Calculate Output dims

        const auto &in_shape = X->shape(); // [N,C,H,W]
        int64_t N = in_shape[0];
        int64_t C = in_shape[1];
        int64_t H = in_shape[2];
        int64_t W = in_shape[3];

        int64_t out_h = (H + 2 * pad_h - kern_h) / stride_h + 1;
        int64_t out_w = (W + 2 * pad_w - kern_w) / stride_w + 1;
        Y->reshape({N, C, out_h, out_w});

        // 3. Now we loop
        const float *X_data = X->data<float>();
        float *y_data = Y->data<float>();
        float pool_area = static_cast<float>(kern_h*kern_w);
        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int oy = 0; oy < out_h; ++oy)
                {
                    for (int ox = 0; ox < out_w; ++ox)
                    {
                        float sum=0.0f;
                        for (int ky = 0; ky < kern_h; ++ky)
                        {
                            for (int kx = 0; kx < kern_w; ++kx)
                            {
                                int64_t iy = oy * stride_h - pad_h + ky;
                                int64_t ix = ox * stride_w - pad_w + kx;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                {
                                    int64_t idx = n * (C * H * W) + c * (H * W) + iy * W + ix;
                                    sum+=X_data[idx];
                                }
                            }
                        }
                        int64_t out_idx = n * (C * out_h * out_w) + c * (out_h * out_w) + oy * out_w + ox;
                        y_data[out_idx] = sum / pool_area;
                    }
                }
            }
        }
    }
};
