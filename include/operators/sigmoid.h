#pragma once
#include "../operator.h"
#include <cmath>

class SigmoidOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node,std::vector<float>& workspace) override
    {
        Tensor *input = inputs[0];
        Tensor *output = outputs[0];
        output->reshape(input->shape());

        const float *in_ptr = input->data<float>();
        float *out_ptr = output->data<float>();
        int64_t size = input->size();

        #pragma omp parallel for if(size > 4096)
        for (int i = 0; i < size; ++i)
        {
            float val = in_ptr[i];
            // OPT:clip for very small values to avoid underflow/NaN(exp of very small numbers can be too small for the precision of a float)
            out_ptr[i] = 1.0f / (1.0f + std::exp(-val));
        }
    }
};
