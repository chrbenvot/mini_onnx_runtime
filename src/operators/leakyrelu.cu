#include "operators/leakyrelu.h"
#include <cuda_runtime.h>
#include "debug_utils.h" // for debugging

__global__ void leaky_relu_kernel(const float *in, float *out, int64_t n, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float val = in[i];
        out[i] = (val > 0.0f) ? val : (val * alpha);
    }
}

void LeakyReluOp::forward_gpu(const std::vector<Tensor *> &inputs,
                              std::vector<Tensor *> &outputs,
                              const onnx::NodeProto &node,
                              cublasHandle_t &handle)
{

    const Tensor *input = inputs[0];
    Tensor *output = outputs[0];
    output->reshape(input->shape());

    float alpha = get_float_attribute(node, "alpha", 0.01f); // Default for YOLO is often 0.01

    // Allocation
    if (output->is_on_device())
        output->free_device_memory();
    output->allocate_device_memory();

    int64_t size = input->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    leaky_relu_kernel<<<blocks, threads>>>(
        (const float *)input->device_data(),
        (float *)output->device_data(),
        size,
        alpha);
}
