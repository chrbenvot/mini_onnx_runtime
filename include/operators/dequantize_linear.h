#pragma once
#include "../operator.h"
#include <vector>
#include <cmath>

class DequantizeLinearOp : public Operator {
public:
    void forward(const std::vector<Tensor*>& inputs, 
                 std::vector<Tensor*>& outputs, 
                 const onnx::NodeProto& node,
                 std::vector<float>& workspace) override {
        
        // Input 0: X (INT8 or UINT8)
        // Input 1: Scale (FLOAT)
        // Input 2: Zero Point (INT8 or UINT8) - Optional (default 0)
        
        const Tensor* X = inputs[0];
        const Tensor* scale_tensor = inputs[1];
        const Tensor* zp_tensor = (inputs.size() > 2) ? inputs[2] : nullptr;
        Tensor* Y = outputs[0];

        Y->reshape(X->shape());

        // Get Scale (Usually a scalar, but can be 1D per-channel)
        float scale = scale_tensor->data<float>()[0];
        
        // Get Zero Point (Default 0)
        int32_t zero_point = 0;
        if (zp_tensor) {
            // Check type: ONNX supports both INT8 and UINT8 quantization
            if (zp_tensor->dtype() == DataType::INT8) {
                zero_point = static_cast<int32_t>(zp_tensor->data<int8_t>()[0]);
            } else if (zp_tensor->dtype() == DataType::UINT8) {
                zero_point = static_cast<int32_t>(zp_tensor->data<uint8_t>()[0]);
            }
        }

        float* y_data = Y->data<float>();
        int64_t size = X->size();

        // --- PROCESSING ---
        if (X->dtype() == DataType::INT8) {
            const int8_t* x_data = X->data<int8_t>();
            
            #pragma omp parallel for if(size > 4096)
            for (int64_t i = 0; i < size; ++i) {
                // Formula: y = (x - zp) * scale
                y_data[i] = static_cast<float>(x_data[i] - zero_point) * scale;
            }
        } 
        else if (X->dtype() == DataType::UINT8) {
            const uint8_t* x_data = X->data<uint8_t>();
            
            #pragma omp parallel for if(size > 4096)
            for (int64_t i = 0; i < size; ++i) {
                y_data[i] = static_cast<float>(static_cast<int32_t>(x_data[i]) - zero_point) * scale;
            }
        } 
        else {
            std::cerr << "Error: DequantizeLinear expects INT8/UINT8 input." << std::endl;
        }
    }
};
