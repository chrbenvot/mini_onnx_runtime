#include "operators/global_avgpool.h"
#include <cuda_runtime.h>

// Kernel
// Each thread handles ONE channel (C) for ONE batch item (N).
// It sums up all H*W pixels and divides by H*W.
__global__ void global_avgpool_kernel(const float* X, float* Y, 
                                      int N, int C, int SpatialSize) {
    
    // Global thread index corresponds to the flat index of the output (n, c)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = N * C; 

    if (idx < total_channels) {
        // 1. Identify Batch (n) and Channel (c)
        int c = idx % C;
        int n = idx / C;

        // 2. Locate Input Data Start
        // Input layout: [N, C, H, W]
        // Flat offset: n * (C * Spatial) + c * Spatial
        const float* x_ptr = X + (n * C * SpatialSize) + (c * SpatialSize);

        // 3. Reduction Loop
        float sum = 0.0f;
        for (int i = 0; i < SpatialSize; ++i) {
            sum += x_ptr[i];
        }

        // 4. Write Output
        // Output layout: [N, C, 1, 1] -> Flat index is just 'idx'
        Y[idx] = sum / (float)SpatialSize;
    }
}

// Implementation

void GlobalAvgPoolOp::forward_gpu(const std::vector<Tensor *> &inputs,
                                  std::vector<Tensor *> &outputs,
                                  const onnx::NodeProto &node,
                                  cublasHandle_t &handle) {
    
    const Tensor *X = inputs[0];
    Tensor *Y = outputs[0];

    const auto &dims = X->shape();
    int64_t N = dims[0];
    int64_t C = dims[1];
    int64_t H = dims[2];
    int64_t W = dims[3];
    int64_t spatial_size = H * W;

    // 1. Reshape Output [N, C, 1, 1]
    Y->reshape({N, C, 1, 1});

    // 2. Allocate Output on GPU
    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();

    // 3. Launch Kernel
    // We launch one thread for every output pixel (N * C).
    int total_threads = N * C;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    global_avgpool_kernel<<<blocks, threads_per_block>>>(
        (const float*)X->device_data(),
        (float*)Y->device_data(),
        (int)N, (int)C, (int)spatial_size
    );
}
