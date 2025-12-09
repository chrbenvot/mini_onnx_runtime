#pragma once
#include "../operator.h"

class GemmOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node) override
    {
        // Inputs: A (Input), B (Weight) C(Bias)
        Tensor *A = inputs[0]; // Shape : [batch,N]
        Tensor *B = inputs[1]; // Shape : [M,N]
        Tensor *C = inputs[2]; // Shape : [M]
        Tensor *Y = outputs[0];

        // Parse attributtes (alpha,beta,transB)
        int64_t transB = get_int_attribute(node, "transB", 0);
        float alpha = 1.0f;
        float beta = 1.0f; // simplified

        int64_t batch = A->shape()[0];
        int64_t input_dim = A->shape()[1];
        int64_t output_dim = (transB) ? B->shape()[0] : B->shape()[1];
        // Sanity Check to prevent Segfaults
        if (transB)
        {
            if (B->shape()[1] != input_dim)
            {
                std::cerr << "Gemm Error: Dimension mismatch. A=" << input_dim << ", B'=" << B->shape()[1] << std::endl;
                return;
            }
        }
        else
        {
            if (B->shape()[0] != input_dim)
            {
                std::cerr << "Gemm Error: Dimension mismatch. A=" << input_dim << ", B=" << B->shape()[0] << std::endl;
                return;
            }
        }

        // Resize output
        Y->reshape({batch, output_dim});
        const float *a_ptr = A->data<float>(); // TODO: again,add int support
        const float *b_ptr = B->data<float>();
        const float *c_ptr = C->data<float>();
        float *y_ptr = Y->data<float>();

        // Matrix Multiplication Loop
        // Y= A* B + C
        for (int b = 0; b < batch; ++b)
        {
            for (int m = 0; m < output_dim; ++m)
            {
                float sum = 0.0f;
                // Dot product
                for (int n = 0; n < input_dim; ++n)
                {
                    float a_val = a_ptr[b * input_dim + n];
                    // Handle transpose logic
                    float b_val;
                    if (transB)
                    {
                        // B is [Out, In] -> we want row m, col n
                        b_val = b_ptr[m * input_dim + n];
                    }
                    else
                    {
                        // B is [In, Out] -> we want row n, col m
                        b_val = b_ptr[n * output_dim + m];
                    }
                    sum += a_val * b_val;
                }
                // Add bias (broadcasted across batches)
                sum += c_ptr[m];
                y_ptr[b * output_dim + m] = sum;
            }
        }
    }
};
