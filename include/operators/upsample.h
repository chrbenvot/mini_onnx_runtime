#pragma once
#include "../operator.h"
#include <cmath>

class UpsampleOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node) override
    {
        const Tensor *X = inputs[0];
        // TODO: it's just simple x2 upscaling for now,according to doc scales is stored in inputs[2] for future ref
        float scale_h = 2.0f;
        float scale_w = 2.0f;

        // Calculate output
        const auto &in_shape = X->shape();
        int64_t N = in_shape[0];
        int64_t C = in_shape[1];
        int64_t H = in_shape[2];
        int64_t W = in_shape[3];
        int64_t out_h = static_cast<int64_t>(H * scale_h);
        int64_t out_w = static_cast<int64_t>(W * scale_w);

        Tensor *Y = outputs[0];
        Y->reshape({N, C, out_h, out_w});
        const float *x_data = X->data<float>();
        float *y_data = Y->data<float>();
        // Loop,we use nearest neighbor for upsampling
        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int oy = 0; oy < out_h; ++oy)
                {
                    for (int ox = 0; ox < out_w; ++ox)
                    {
                        // Which input pixel covers this output pixel? nearest means rounding down through integer casting of a float
                        int64_t iy = static_cast<int64_t>(oy / scale_h);
                        int64_t ix = static_cast<int64_t>(ox / scale_w);
                        // To be safe ( we dont float point rounding to cause iy=H and thus an out of bound access)
                        if (iy >= H)
                            iy = H - 1;
                        if (ix >= W)
                            ix = W - 1;
                        int64_t in_idx = n * (C * H * W) + c * (H * W) + iy * W + ix;
                        int64_t out_idx = n * (C * out_h * out_w) + c * (out_h * out_w) + oy * out_w + ox;
                        y_data[out_idx] = x_data[in_idx];
                    }
                }
            }
        }
    }
};
