#pragma once
#include "../operator.h"
#include <algorithm>

class LeakyReluOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node) override
    {
        Tensor *input = inputs[0];
        Tensor *output = outputs[0];
        output->reshape(input->shape());
        // Get alpha ( the slope for negative values,0.01 by default)
        float alpha = 0.01f;
        for (const auto &attr : node.attribute())
        {
            if (attr.name() == "alpha")
                alpha = attr.f();
        }
        const float *in_ptr = input->data<float>();
        float *out_ptr = output->data<float>();
        int64_t size = input->size();

        for (int64_t i = 0; i < size; ++i)
        {
            float val = in_ptr[i];
            out_ptr[i] = (val >= 0) ? val : (val * alpha);
        }
    }
};
