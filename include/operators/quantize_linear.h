#pragma once
#include "../operator.h"
#include <vector>
#include <cmath>
#include <algorithm>

// Again this turned out to be useless
class QuantizeLinearOp : public Operator
{
public:
    std::string get_op_type() const { return "QuantizeLinear"; }
    void forward_gpu(const std::vector<Tensor *> &inputs,
                     std::vector<Tensor *> &outputs,
                     const onnx::NodeProto &node,
                     cublasHandle_t &handle) override;
    void forward_cpu(const std::vector<Tensor *> &inputs,
                     std::vector<Tensor *> &outputs,
                     const onnx::NodeProto &node,
                     std::vector<float> &workspace) override
    {

        const Tensor *X = inputs[0]; // FLOAT
        const Tensor *scale_tensor = inputs[1];
        const Tensor *zp_tensor = (inputs.size() > 2) ? inputs[2] : nullptr;
        Tensor *Y = outputs[0]; // Output is usually INT8 or UINT8

        Y->reshape(X->shape());

        float scale = scale_tensor->data<float>()[0];

        int32_t zero_point = 0;
        DataType out_type = DataType::UINT8; // Default if ZP is missing

        if (zp_tensor)
        {
            out_type = zp_tensor->dtype();
            if (out_type == DataType::INT8)
                zero_point = static_cast<int32_t>(zp_tensor->data<int8_t>()[0]);
            else
                zero_point = static_cast<int32_t>(zp_tensor->data<uint8_t>()[0]);
        }

        const float *x_data = X->data<float>();
        int64_t size = X->size();

        // processing
        // Formula: y = clamp(round(x / scale) + zp)

        if (out_type == DataType::INT8)
        {
            // Range [-128, 127]
            int8_t *y_data = Y->data<int8_t>();

#pragma omp parallel for if (size > 4096)
            for (int64_t i = 0; i < size; ++i)
            {
                float val = std::round(x_data[i] / scale) + zero_point;
                // Clamp and cast
                val = std::max(-128.0f, std::min(127.0f, val));
                y_data[i] = static_cast<int8_t>(val);
            }
        }
        else
        {
            // Range [0, 255]
            uint8_t *y_data = Y->data<uint8_t>();

#pragma omp parallel for if (size > 4096)
            for (int64_t i = 0; i < size; ++i)
            {
                float val = std::round(x_data[i] / scale) + zero_point;
                // Clamp and cast
                val = std::max(0.0f, std::min(255.0f, val));
                y_data[i] = static_cast<uint8_t>(val);
            }
        }
    }
};
