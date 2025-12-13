#include "operators/upsample.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// KERNEL
// Nearest Neighbor Interpolation
// Maps Output (n, c, oh, ow) -> Input (n, c, ih, iw)
__global__ void upsample_nearest_kernel(const float* input, float* output,
                                        int N, int C, 
                                        int in_h, int in_w,
                                        int out_h, int out_w,
                                        float inv_scale_h, float inv_scale_w) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * out_h * out_w;

    if (index < total_elements) {
        // 1. Decode Output Index
        int ow = index % out_w;
        int temp = index / out_w;
        int oh = temp % out_h;
        int c  = (temp / out_h) % C;
        int n  = temp / (out_h * C);

        // 2. Map to Input Coordinates (Nearest Neighbor "Floor")
        // Formula: input_coord = floor(output_coord * (1 / scale))
        int ih = (int)(oh * inv_scale_h);
        int iw = (int)(ow * inv_scale_w);

        // 3. Bounds Check (Clamp)
        if (ih >= in_h) ih = in_h - 1;
        if (iw >= in_w) iw = in_w - 1;

        // 4. Read/Write
        int input_idx = n * (C * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw;
        output[index] = input[input_idx];
    }
}

// Implementation

void UpsampleOp::forward_gpu(const std::vector<Tensor *> &inputs,
                             std::vector<Tensor *> &outputs,
                             const onnx::NodeProto &node,
                             cublasHandle_t &handle) {
    
    const Tensor *X = inputs[0];
    Tensor *Y = outputs[0];

    const auto &in_shape = X->shape();
    int64_t N = in_shape[0];
    int64_t C = in_shape[1];
    int64_t H = in_shape[2];
    int64_t W = in_shape[3];

    float scale_h = 1.0f;
    float scale_w = 1.0f;

    // 1. Parse Scales / Sizes
    // Since inputs might be on GPU, we copy the relevant scalars to CPU
    // so we can calculate dimensions and launch params.
    
    if (inputs.size() > 2 && inputs[2]->size() > 0) {
        // Case A: Scales provided (FLOAT)
        const Tensor *scales_tensor = inputs[2];
        
        // We need to read up to 4 floats. 
        std::vector<float> s_data(scales_tensor->size());
        
        if (scales_tensor->is_on_device()) {
            cudaMemcpy(s_data.data(), scales_tensor->device_data(), 
                       scales_tensor->size() * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            // Fallback if somehow it's on CPU (rare in full GPU run)
            const float* ptr = scales_tensor->data<float>();
            s_data.assign(ptr, ptr + scales_tensor->size());
        }

        if (s_data.size() >= 4) {
            scale_h = s_data[2];
            scale_w = s_data[3];
        } else if (s_data.size() == 2) {
            scale_h = s_data[0];
            scale_w = s_data[1];
        }
    } 
    else if (inputs.size() > 3 && inputs[3]->size() > 0) {
        // Case B: Sizes provided (INT64)
        const Tensor *sizes_tensor = inputs[3];
        std::vector<int64_t> size_data(sizes_tensor->size());

        if (sizes_tensor->is_on_device()) {
             cudaMemcpy(size_data.data(), sizes_tensor->device_data(), 
                        sizes_tensor->size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
        } else {
            const int64_t* ptr = sizes_tensor->data<int64_t>();
            size_data.assign(ptr, ptr + sizes_tensor->size());
        }

        int64_t target_h = size_data[2];
        int64_t target_w = size_data[3];
        scale_h = (float)target_h / (float)H;
        scale_w = (float)target_w / (float)W;
    }

    // 2. Calculate Output Dims
    int64_t out_h = static_cast<int64_t>(H * scale_h);
    int64_t out_w = static_cast<int64_t>(W * scale_w);

    Y->reshape({N, C, out_h, out_w});

    // 3. Allocate GPU
    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();

    // 4. Launch
    int total_elements = N * C * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    // Use inverse scale for multiplication in kernel (faster than division)
    float inv_scale_h = 1.0f / scale_h;
    float inv_scale_w = 1.0f / scale_w;

    upsample_nearest_kernel<<<blocks, threads>>>(
        (const float*)X->device_data(),
        (float*)Y->device_data(),
        N, C,
        H, W,
        out_h, out_w,
        inv_scale_h, inv_scale_w
    );
}
