#include "operators/relu.h"
#include <cuda_runtime.h>

// Kernel
__global__ void relu_kernel(const float* in, float* out, int64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = in[i];
        out[i] = (val > 0.0f) ? val : 0.0f;
    }
}

// Implementation
void ReluOp::forward_gpu(const std::vector<Tensor*>& inputs, 
                         std::vector<Tensor*>& outputs, 
                         const onnx::NodeProto& node, 
                         cublasHandle_t& handle) {
    
    const Tensor* input = inputs[0];
    Tensor* output = outputs[0];
    
    // 1. Reshape Output (Syncs CPU shape)
    output->reshape(input->shape());

    // 2. ALLOCATE GPU MEMORY 
    if (output->is_on_device()) output->free_device_memory();
    output->allocate_device_memory();

    // 3. Launch Kernel
    int64_t size = input->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(
        (const float*)input->device_data(), 
        (float*)output->device_data(), 
        size
    );
}
