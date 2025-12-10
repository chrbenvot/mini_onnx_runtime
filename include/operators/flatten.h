#pragma once
#include "../operator.h"

class FlattenOp : public Operator
{
public:
std::string get_op_type() const { return "Flatten"; }
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node,std::vector<float>& workspace) override
    {
        Tensor* input = inputs[0];
        Tensor* output = outputs[0];

        const auto& in_shape = input->shape();
        int64_t rank = in_shape.size();

        // Get Axis (Default 1)
        int64_t axis = get_int_attribute(node, "axis", 1);
        
        // Handle negative axis (e.g., -1 means last dim)
        if (axis < 0) axis += rank;
        
        // Safety clamp
        if (axis < 0) axis = 0;
        if (axis > rank) axis = rank;

        // Calculate the two new dimensions
        // dim_0: Product of dimensions [0, axis)
        // dim_1: Product of dimensions [axis, rank)
        int64_t dim_0 = 1;
        for (int i = 0; i < axis; ++i) {
            dim_0 *= in_shape[i];
        }

        int64_t dim_1 = 1;
        for (int i = axis; i < rank; ++i) {
            dim_1 *= in_shape[i];
        }

        // Copy Data & Reshape
        // Since we don't have "View" semantics (shared memory) yet,
        // we must copy the data to the output tensor.
        // OPTIONAL: share the pointer? View semantics
        *output = *input; 
        
        // Apply the new 2D shape
        output->reshape({dim_0, dim_1});
    }
};
