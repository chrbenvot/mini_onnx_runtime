#pragma once
#include "../operator.h"
#include <cmath>

class BatchNorm : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node,std::vector<float>& workspace) override
    {
        Tensor *X = inputs[0];
        Tensor *scale = inputs[1]; // Gamma
        Tensor *B = inputs[2];     // Beta
        Tensor *mean = inputs[3];
        Tensor *var = inputs[4];

        Tensor *Y = outputs[0];
        Y->reshape(X->shape());

        // Get the epsilon attribute, used to avoid division by zero if the variance is zero
        // default is 1e-5
        float epsilon = get_float_attribute(node, "epsilon", 1e-5f);

        const float *x_data = X->data<float>();
        const float *s_data = scale->data<float>();
        const float *b_data = B->data<float>();
        const float *m_data = mean->data<float>();
        const float *v_data = var->data<float>();
        float *y_data = Y->data<float>();

        int64_t N = X->shape()[0];
        int64_t C = X->shape()[1];
        int64_t H = X->shape()[2];
        int64_t W = X->shape()[3];
        int64_t image_size = H * W;

        // OPTIMIZE: pre-calculate scale/shift for each channel?

        std::vector<float> new_scale(C);
        std::vector<float> new_bias(C);
        for (int c = 0; c < C; ++c)
        {
            float inv_std = 1.0f / std::sqrt(v_data[c] + epsilon);
            new_scale[c] = s_data[c] * inv_std;
            new_bias[c] = b_data[c] - (m_data[c] * new_scale[c]);
        }
        // Apply to each pixel
        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                float ns = new_scale[c];
                float
                    nb = new_bias[c];
                int64_t channel_offset = n * (C * image_size) + c * image_size;
                for (int i = 0; i < image_size; ++i)
                {
                    y_data[channel_offset + i] = x_data[channel_offset + i] * ns + nb;
                }
            }
        }
    }
};
