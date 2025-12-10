#pragma once
#include "../operator.h"

class GlobalAvgPoolOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node,std::vector<float>& workspace) override
    {
        const Tensor *X = inputs[0];
        Tensor *Y = outputs[0];

        const auto &dims = X->shape();
        int64_t N = dims[0];
        int64_t C = dims[1];
        int64_t H = dims[2];
        int64_t W = dims[3];
        int64_t image_size = H * W;
        Y->reshape({N, C, 1, 1}); // global average pool averages each feature map into one pixel

        const float *x_data = X->data<float>();
        float *y_data = Y->data<float>();
        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                float sum = 0.0f;
                int64_t offset = n * (C * image_size) + c * image_size;
                for (int i = 0; i < image_size; ++i)
                {
                    sum += x_data[offset + i];
                }
                y_data[n * C + c] = sum / image_size;
            }
        }
    }
};
