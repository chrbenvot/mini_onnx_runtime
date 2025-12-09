#pragma once
#include "../operator.h"

class AddOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node) override
    {
        Tensor *A = inputs[0];
        Tensor *B = inputs[1];
        Tensor *Y = outputs[0];
        Y->reshape(A->shape());
        const float *a_ptr = A->data<float>();
        const float *b_ptr = B->data<float>();
        float *y_ptr = Y->data<float>();

        int64_t size = A->size();
        // B can be broadcasted
        if (B->size() == 1)
        {
            float b_val = b_ptr[0];
            for (int64_t i = 0; i < size; ++i)
            {
                y_ptr[i] = a_ptr[i] + b_val;
            }
        }
        else if (A->shape() == B->shape())
        {
            for (int64_t i = 0; i < size; ++i)
            {
                y_ptr[i] = a_ptr[i] + b_ptr[i];
            }
        }
        else
        {
            std::cerr<< "Warning: Complex broadcasting not implemented in AddOp." <<std::endl;
        }
    }
};
