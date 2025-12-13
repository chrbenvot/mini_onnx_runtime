#include "operators/add.h"
#include "operators/mul.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

// Max dimensions supported for broadcasting (YOLO is usually 4, 8 is safe)
#define MAX_DIMS 8

struct BroadcastStrides {
    int strides[MAX_DIMS];
};

// Kernels

// Fast Path: Identical Shapes
__global__ void add_simple_kernel(const float* A, const float* B, float* Y, int64_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) Y[index] = A[index] + B[index];
}

__global__ void mul_simple_kernel(const float* A, const float* B, float* Y, int64_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) Y[index] = A[index] * B[index];
}

// Slow Path: General Broadcasting
// Each thread handles one output element, but has to calculate input indices
__global__ void broadcast_kernel(const float* A, const float* B, float* Y, 
                                 int64_t total_elements, int dims,
                                 BroadcastStrides s_out, 
                                 BroadcastStrides s_a, 
                                 BroadcastStrides s_b,
                                 bool is_add) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int64_t temp_idx = idx;
        int64_t idx_a = 0;
        int64_t idx_b = 0;

        // Map flat output index to input indices
        // We unroll the loop manually or let compiler handle it
        for (int i = 0; i < dims; ++i) {
            int64_t coord = temp_idx / s_out.strides[i];
            temp_idx %= s_out.strides[i];
            
            idx_a += coord * s_a.strides[i];
            idx_b += coord * s_b.strides[i];
        }

        if (is_add) {
            Y[idx] = A[idx_a] + B[idx_b];
        } else {
            Y[idx] = A[idx_a] * B[idx_b];
        }
    }
}

// HELPER to Calculate Strides (Copied from CPU Logic) 
void prepare_broadcast(const Tensor* A, const Tensor* B, Tensor* Y,
                       BroadcastStrides& s_out, BroadcastStrides& s_a, BroadcastStrides& s_b,
                       int& max_rank) {
    
    int rank_a = A->shape().size();
    int rank_b = B->shape().size();
    max_rank = std::max(rank_a, rank_b);

    if (max_rank > MAX_DIMS) {
        std::cerr << "Error: Tensor rank too high for GPU broadcast kernel (" << max_rank << " > " << MAX_DIMS << ")" << std::endl;
        exit(1);
    }

    std::vector<int64_t> out_shape(max_rank);
    std::vector<int64_t> pad_shape_a(max_rank);
    std::vector<int64_t> pad_shape_b(max_rank);

    // 1. Align Dimensions
    for (int i = 0; i < max_rank; ++i) {
        int idx_out = max_rank - 1 - i;
        int idx_a = rank_a - 1 - i;
        int idx_b = rank_b - 1 - i;

        int64_t dim_a = (idx_a >= 0) ? A->shape()[idx_a] : 1;
        int64_t dim_b = (idx_b >= 0) ? B->shape()[idx_b] : 1;

        out_shape[idx_out] = std::max(dim_a, dim_b);
        pad_shape_a[idx_out] = dim_a;
        pad_shape_b[idx_out] = dim_b;
    }
    
    Y->reshape(out_shape);

    // 2. Calculate Strides
    int64_t current_s_out = 1;
    int64_t current_s_a = 1;
    int64_t current_s_b = 1;

    for (int i = max_rank - 1; i >= 0; --i) {
        s_out.strides[i] = current_s_out;
        
        // For inputs, if dim is 1, stride is 0 (broadcast)
        s_a.strides[i] = (pad_shape_a[i] == 1) ? 0 : current_s_a;
        s_b.strides[i] = (pad_shape_b[i] == 1) ? 0 : current_s_b;

        // Update current strides
        current_s_out *= out_shape[i];
        
        if (i >= (max_rank - rank_a)) 
            current_s_a *= A->shape()[i - (max_rank - rank_a)];
            
        if (i >= (max_rank - rank_b)) 
            current_s_b *= B->shape()[i - (max_rank - rank_b)];
    }
}

// AddOp implementation

void AddOp::forward_gpu(const std::vector<Tensor*>& inputs, 
                        std::vector<Tensor*>& outputs, 
                        const onnx::NodeProto& node, 
                        cublasHandle_t& handle) {
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    Tensor* Y = outputs[0];

    // Path 1: Simple (Identical Shapes)
    if (A->shape() == B->shape()) {
        Y->reshape(A->shape());
        if (Y->is_on_device()) Y->free_device_memory();
        Y->allocate_device_memory();
        
        int threads = 256;
        int blocks = (A->size() + threads - 1) / threads;
        add_simple_kernel<<<blocks, threads>>>((const float*)A->device_data(), (const float*)B->device_data(), (float*)Y->device_data(), A->size());
        return;
    }

    // Path 2: Broadcasting
    BroadcastStrides s_out, s_a, s_b;
    int dims;
    prepare_broadcast(A, B, Y, s_out, s_a, s_b, dims);
    
    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();
    
    int threads = 256;
    int blocks = (Y->size() + threads - 1) / threads;
    
    broadcast_kernel<<<blocks, threads>>>(
        (const float*)A->device_data(), (const float*)B->device_data(), (float*)Y->device_data(),
        Y->size(), dims, s_out, s_a, s_b, true // is_add = true
    );
}

// MulOp implementation

void MulOp::forward_gpu(const std::vector<Tensor*>& inputs, 
                        std::vector<Tensor*>& outputs, 
                        const onnx::NodeProto& node, 
                        cublasHandle_t& handle) {
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    Tensor* Y = outputs[0];

    // Path 1: Simple
    if (A->shape() == B->shape()) {
        Y->reshape(A->shape());
        if (Y->is_on_device()) Y->free_device_memory();
        Y->allocate_device_memory();
        
        int threads = 256;
        int blocks = (A->size() + threads - 1) / threads;
        mul_simple_kernel<<<blocks, threads>>>((const float*)A->device_data(), (const float*)B->device_data(), (float*)Y->device_data(), A->size());
        return;
    }

    // Path 2: Broadcasting
    BroadcastStrides s_out, s_a, s_b;
    int dims;
    prepare_broadcast(A, B, Y, s_out, s_a, s_b, dims);

    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();

    int threads = 256;
    int blocks = (Y->size() + threads - 1) / threads;
    
    broadcast_kernel<<<blocks, threads>>>(
        (const float*)A->device_data(), (const float*)B->device_data(), (float*)Y->device_data(),
        Y->size(), dims, s_out, s_a, s_b, false // is_add = false (Multiply)
    );
}
