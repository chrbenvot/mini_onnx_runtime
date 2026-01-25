#include "operators/quantize_linear.h"
#include "operators/dequantize_linear.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

// In retrospect this is uselss but we kept it anyway
// Kernels

// Kernel: Float -> Int8/Uint8
// T represents output type (int8_t or uint8_t)
template <typename T>
__global__ void quantize_kernel(const float* input, T* output, int64_t n, 
                                float scale, int32_t zero_point, 
                                float min_val, float max_val) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        // Formula: y = clamp(round(x / scale) + zp)
        float val = roundf(input[index] / scale) + zero_point;
        
        // Clamp
        if (val < min_val) val = min_val;
        if (val > max_val) val = max_val;
        
        output[index] = static_cast<T>(val);
    }
}

// Kernel: Int8/Uint8 -> Float
// T represents input type (int8_t or uint8_t)
template <typename T>
__global__ void dequantize_kernel(const T* input, float* output, int64_t n, 
                                  float scale, int32_t zero_point) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        // Formula: y = (x - zp) * scale
        float val = static_cast<float>(input[index]);
        output[index] = (val - static_cast<float>(zero_point)) * scale;
    }
}

// Quantize Implementation

void QuantizeLinearOp::forward_gpu(const std::vector<Tensor*>& inputs, 
                                   std::vector<Tensor*>& outputs, 
                                   const onnx::NodeProto& node, 
                                   cublasHandle_t& handle) {
    const Tensor* X = inputs[0];
    const Tensor* scale_tensor = inputs[1];
    const Tensor* zp_tensor = (inputs.size() > 2) ? inputs[2] : nullptr;
    Tensor* Y = outputs[0];

    Y->reshape(X->shape());
    
    // Copy Scale/ZP from GPU to CPU for Kernel Args
    // (Optimization: Since these are usually single scalars, we copy them back 
    // to pass them as kernel arguments by value, which is faster than pointer dereferencing on GPU)
    float scale = 0.0f;
    cudaMemcpy(&scale, scale_tensor->device_data(), sizeof(float), cudaMemcpyDeviceToHost);

    int32_t zero_point = 0;
    DataType out_type = DataType::UINT8;

    if (zp_tensor) {
        out_type = zp_tensor->dtype();
        if (out_type == DataType::INT8) {
            int8_t zp_val;
            cudaMemcpy(&zp_val, zp_tensor->device_data(), sizeof(int8_t), cudaMemcpyDeviceToHost);
            zero_point = static_cast<int32_t>(zp_val);
        } else {
            uint8_t zp_val;
            cudaMemcpy(&zp_val, zp_tensor->device_data(), sizeof(uint8_t), cudaMemcpyDeviceToHost);
            zero_point = static_cast<int32_t>(zp_val);
        }
    }

    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();

    int threads = 256;
    int blocks = (X->size() + threads - 1) / threads;

    if (out_type == DataType::INT8) {
        quantize_kernel<int8_t><<<blocks, threads>>>(
            (const float*)X->device_data(), 
            (int8_t*)Y->device_data(), 
            X->size(), scale, zero_point, -128.0f, 127.0f
        );
    } else {
        quantize_kernel<uint8_t><<<blocks, threads>>>(
            (const float*)X->device_data(), 
            (uint8_t*)Y->device_data(), 
            X->size(), scale, zero_point, 0.0f, 255.0f
        );
    }
}

// Dequantize implementation

void DequantizeLinearOp::forward_gpu(const std::vector<Tensor*>& inputs, 
                                     std::vector<Tensor*>& outputs, 
                                     const onnx::NodeProto& node, 
                                     cublasHandle_t& handle) {
    const Tensor* X = inputs[0];
    const Tensor* scale_tensor = inputs[1];
    const Tensor* zp_tensor = (inputs.size() > 2) ? inputs[2] : nullptr;
    Tensor* Y = outputs[0];

    Y->reshape(X->shape());

    float scale = 0.0f;
    cudaMemcpy(&scale, scale_tensor->device_data(), sizeof(float), cudaMemcpyDeviceToHost);

    int32_t zero_point = 0;
    if (zp_tensor) {
        if (zp_tensor->dtype() == DataType::INT8) {
            int8_t zp_val;
            cudaMemcpy(&zp_val, zp_tensor->device_data(), sizeof(int8_t), cudaMemcpyDeviceToHost);
            zero_point = static_cast<int32_t>(zp_val);
        } else {
            uint8_t zp_val;
            cudaMemcpy(&zp_val, zp_tensor->device_data(), sizeof(uint8_t), cudaMemcpyDeviceToHost);
            zero_point = static_cast<int32_t>(zp_val);
        }
    }

    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();

    int threads = 256;
    int blocks = (X->size() + threads - 1) / threads;

    if (X->dtype() == DataType::INT8) {
        dequantize_kernel<int8_t><<<blocks, threads>>>(
            (const int8_t*)X->device_data(), 
            (float*)Y->device_data(), 
            X->size(), scale, zero_point
        );
    } else {
        dequantize_kernel<uint8_t><<<blocks, threads>>>(
            (const uint8_t*)X->device_data(), 
            (float*)Y->device_data(), 
            X->size(), scale, zero_point
        );
    }
}
