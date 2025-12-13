#include "operators/sigmoid.h"
#include <cuda_runtime.h>
#include <math.h> // for expf

// Kernel
// Each thread handles one element.
// Formula: y = 1.0 / (1.0 + exp(-x))
__global__ void sigmoid_kernel(const float* input, float* output, int64_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        float val = input[index];
        // Use expf() for float precision on CUDA
        output[index] = 1.0f / (1.0f + expf(-val));
    }
}

// Implementation
void SigmoidOp::forward_gpu(const std::vector<Tensor *> &inputs,
                            std::vector<Tensor *> &outputs,
                            const onnx::NodeProto &node,
                            cublasHandle_t &handle) {
    
    const Tensor *input = inputs[0];
    Tensor *output = outputs[0];
    
    // 1. Reshape Output
    output->reshape(input->shape());

    // 2. Allocate Output on GPU
    if (output->is_on_device()) output->free_device_memory();
    output->allocate_device_memory();

    // 3. Launch Kernel
    int64_t size = input->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    sigmoid_kernel<<<blocks, threads>>>(
        (const float*)input->device_data(),
        (float*)output->device_data(),
        size
    );
}
