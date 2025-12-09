#pragma once
#include "../operator.h"
#include <cmath>
#include <algorithm>
#include <cfloat>

class SoftmaxOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node) override
    {

        Tensor *input = inputs[0];
        Tensor *output = outputs[0];

        output->reshape(input->shape());

        // 1. Get Axis (Optional, usually -1 / last dimension)
        int64_t axis = get_int_attribute(node, "axis", -1);

        // Handle negative axis logic (e.g., -1 means last dim)
        if (axis < 0)
            axis += input->shape().size();

        // 2. Dimensions Logic
        // Softmax treats the tensor as a set of 1D vectors along 'axis'
        // We can flatten it into Outer * AxisDim * Inner loops
        int64_t N = 1;
        for (int i = 0; i < axis; ++i)
            N *= input->shape()[i];

        int64_t D = input->shape()[axis]; // The dimension we sum over (e.g., 10 classes)

        int64_t inner = 1;
        for (size_t i = axis + 1; i < input->shape().size(); ++i)
            inner *= input->shape()[i];

        const float *in_ptr = input->data<float>();
        float *out_ptr = output->data<float>();

        // 3. The Softmax Loop
        for (int i = 0; i < N; ++i)
        {
            for (int k = 0; k < inner; ++k)
            {

                int64_t offset = i * (D * inner) + k;

                // Find Max
                float max_val = -FLT_MAX;
                for (int j = 0; j < D; ++j)
                {

                    float val = in_ptr[offset + j * inner];
                    if (val > max_val)
                        max_val = val;
                }

                // Calculate Exponentials and Sum
                float sum = 0.0f;
                for (int j = 0; j < D; ++j)
                {
                    float val = in_ptr[offset + j * inner];
                    // Subtract max to prevent overflow
                    float exp_val = std::exp(val - max_val);

                    // Store temporarily in output buffer
                    out_ptr[offset + j * inner] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                for (int j = 0; j < D; ++j)
                {
                    out_ptr[offset + j * inner] /= sum;
                }
            }
        }
    }
};
