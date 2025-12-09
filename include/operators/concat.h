#pragma once
#include "../operator.h"

class ConcatOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node) override
    {
        // On which axis are we trying to concat?
        int64_t axis = get_int_attribute(node, "axis", 1); // we'll concat on batch by default
        // Handle negative axis(-1 means last dimension for instance)
        if (axis < 0)
            axis += inputs[0]->shape().size();

        // Calculate Output Shape (all dimensions of input tensors must be the same except for the concat axis)
        std::vector<int64_t> out_shape = inputs[0]->shape();
        int64_t axis_sum = 0;
        for (const Tensor *t : inputs)
        {
            axis_sum += t->shape()[axis];
        }
        out_shape[axis] = axis_sum; // eg: concating Tensors of shape [N,C,H,W] and [M,C,H,W] should output a tensor of Shape [N+M,C,H,W]
        Tensor *Y = outputs[0];
        Y->reshape(out_shape);
        // Now for the actual copying
        // The tensor can be decomposed into
        // [outer_block] x [axis_dim] x [inner_block]
        int64_t outer_size = 1;
        for (int i = 0; i < axis; ++i)
            outer_size *= out_shape[i];
        int64_t inner_size = 1;
        for (int i = axis + 1; i < out_shape.size(); ++i)
            inner_size *= out_shape[i];

        float *y_ptr = Y->data<float>();
        int64_t output_offset = 0;
        // Iterate through every outer block
        for (int i = 0; i < outer_size; ++i)
        {
            // For this outer block,we copy data from each tensor sequentially
            for (const Tensor *input : inputs)
            {
                const float *x_ptr = input->data<float>();
                int64_t axis_dim = input->shape()[axis];  // How big is this tensor along the concat axis?
                int64_t copy_size = axis_dim * inner_size; // How much data we'll be copying

                // Calculate offset from the source
                int64_t input_offset = i * copy_size;
                std::memcpy(y_ptr + output_offset, x_ptr + input_offset, copy_size * sizeof(float));
                // Advance the output pointer
                output_offset += copy_size;
            }
        }
    }
};
