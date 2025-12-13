#include "operators/concat.h"
#include <cuda_runtime.h>
#include <iostream>

// KERNEL 
// Each thread copies one element from Input -> Output
// It calculates the destination address based on strides.
__global__ void concat_kernel(const float* input, float* output, 
                              int64_t total_elements,
                              int64_t inner_size, 
                              int64_t input_axis_dim, 
                              int64_t output_axis_dim,
                              int64_t axis_offset) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // 1. Map Linear Input Index to (Outer, Axis, Inner)
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t axis_idx = temp % input_axis_dim;
        int64_t outer_idx = temp / input_axis_dim;

        // 2. Map to Linear Output Index
        // The Output has a larger 'Axis' dimension (output_axis_dim)
        // We shift the axis index by 'axis_offset'
        int64_t output_idx = outer_idx * (output_axis_dim * inner_size) + 
                             (axis_idx + axis_offset) * inner_size + 
                             inner_idx;
        
        output[output_idx] = input[idx];
    }
}

// IMPLEMENTATION 
void ConcatOp::forward_gpu(const std::vector<Tensor *> &inputs,
                           std::vector<Tensor *> &outputs,
                           const onnx::NodeProto &node,
                           cublasHandle_t &handle) {
    
    if (inputs.empty()) return;
    Tensor *Y = outputs[0];

    // 1. Attributes & Shape Calculation
    int64_t axis = get_int_attribute(node, "axis", 1);
    if (axis < 0) axis += inputs[0]->shape().size();

    std::vector<int64_t> out_shape = inputs[0]->shape();
    int64_t output_axis_sum = 0;
    
    for (const Tensor *t : inputs) {
        output_axis_sum += t->shape()[axis];
    }
    out_shape[axis] = output_axis_sum;
    
    Y->reshape(out_shape);

    // 2. Allocate Output
    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();
    float* d_Y = (float*)Y->device_data();

    // 3. Calculate Block Sizes
    // [Outer] x [Axis] x [Inner]
    int64_t inner_size = 1;
    for (int i = axis + 1; i < out_shape.size(); ++i) inner_size *= out_shape[i];

    // 4. Launch Kernel Loop
    // We launch one kernel per input tensor.
    // This is much faster than looping 'outer_size' times.
    
    int64_t current_axis_offset = 0;

    for (const Tensor *input : inputs) {
        int64_t input_elements = input->size();
        int64_t input_axis_dim = input->shape()[axis];
        
        int threads = 256;
        int blocks = (input_elements + threads - 1) / threads;

        concat_kernel<<<blocks, threads>>>(
            (const float*)input->device_data(), 
            d_Y,
            input_elements,
            inner_size,
            input_axis_dim,
            output_axis_sum,    // The full size of the output's axis dim
            current_axis_offset // Where to start writing for this tensor
        );

        current_axis_offset += input_axis_dim;
    }
}
