#pragma once
#include "../operator.h"
#include <algorithm> // For std::max
#include "../simd_utils.h"
#include "relu.cuh"
#include <cublas_v2.h>

class ReluOp : public Operator
{
public:
    std::string get_op_type() const { return "Relu"; }
    void forward_cpu(const std::vector<Tensor *> &inputs,
                     std::vector<Tensor *> &outputs,
                     const onnx::NodeProto &node,
                     std::vector<float> &workspace) override
    {

        Tensor *input = inputs[0];
        Tensor *output = outputs[0];
        output->reshape(input->shape());

        // Call SIMD Helper
        relu_avx(input->data<float>(), output->data<float>(), input->size());
    }
    void forward_gpu(const std::vector<Tensor *> &inputs,
                     std::vector<Tensor *> &outputs,
                     const onnx::NodeProto &node, cublasHandle_t &handle) override;
};
