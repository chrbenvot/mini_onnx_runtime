#pragma once
#include "../operator.h"
#include <algorithm>
#include "../simd_utils.h"

class LeakyReluOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node,std::vector<float>& workspace) override
    {
        Tensor *input = inputs[0];
        Tensor *output = outputs[0];
        output->reshape(input->shape());
        // Get alpha ( the slope for negative values,0.01 by default)
        float alpha = get_float_attribute(node,"alpha",0.01f);
        const float *in_ptr = input->data<float>();
        float *out_ptr = output->data<float>();
        int64_t size = input->size();

        leaky_relu_avx(in_ptr,out_ptr,size,alpha);
    }
};
