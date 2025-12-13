#include "operators/flatten.h"
#include <cuda_runtime.h>
#include <iostream>

void FlattenOp::forward_gpu(const std::vector<Tensor *> &inputs,
                            std::vector<Tensor *> &outputs,
                            const onnx::NodeProto &node,
                            cublasHandle_t &handle) {
    
    const Tensor* X = inputs[0];
    Tensor* Y = outputs[0];
    const auto& in_shape = X->shape();
    int64_t rank = in_shape.size();

    // 1. Calculate New Shape (Identical logic to CPU)
    int64_t axis = get_int_attribute(node, "axis", 1);
    if (axis < 0) axis += rank;
    if (axis < 0) axis = 0;
    if (axis > rank) axis = rank;

    int64_t dim_0 = 1;
    for (int i = 0; i < axis; ++i) {
        dim_0 *= in_shape[i];
    }

    int64_t dim_1 = 1;
    for (int i = axis; i < rank; ++i) {
        dim_1 *= in_shape[i];
    }

    // 2. Reshape Output
    Y->reshape({dim_0, dim_1});

    // 3. Allocate Output on GPU
    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();

    // 4. Perform Device-to-Device Copy
    // Since Flatten does not change the order of data in memory, 
    // we just clone the raw bytes from X to Y.
    
    cudaError_t err = cudaMemcpy(Y->device_data(), 
                                 X->device_data(), 
                                 X->size() * sizeof(float), 
                                 cudaMemcpyDeviceToDevice); 

    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in FlattenOp: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}
