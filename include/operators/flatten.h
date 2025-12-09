#pragma once
#include "../operator.h"

class FlattenOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node) override
    {
        Tensor* input = inputs[0];
        Tensor* output = outputs[0];

        // Flatten keeps the batch dimension!
        const auto& in_shape = input->shape();
        int64_t batch =in_shape[0];

        int64_t remaining =1;
        for (size_t i=1;i<in_shape.size();++i){
            remaining *=in_shape[i];
        }
        // Copy data 
        // OPTIONAL: share the pointer?
        *output=*input;
        output->reshape({batch,remaining});
    }
};
