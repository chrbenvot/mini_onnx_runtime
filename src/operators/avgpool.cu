#include "operators/avgpool.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <algorithm>

__global__ void avgpool_kernel(const float* X, float* Y, 
                               int N, int C, int H, int W,
                               int out_h, int out_w,
                               int kern_h, int kern_w,
                               int pad_t, int pad_l,
                               int stride_h, int stride_w,
                               int count_include_pad) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * out_h * out_w;

    if (index < total_elements) {
        // 1. Decode Index
        int pw = index % out_w;
        int ph = (index / out_w) % out_h;
        int c  = (index / (out_w * out_h)) % C;
        int n  = index / (out_w * out_h * C);

        // 2. Calculate Window (Unclamped)
        int h_start = ph * stride_h - pad_t;
        int w_start = pw * stride_w - pad_l;
        int h_end = h_start + kern_h;
        int w_end = w_start + kern_w;

        // 3. Clamp for Summation
        int h_start_valid = max(h_start, 0);
        int w_start_valid = max(w_start, 0);
        int h_end_valid = min(h_end, H);
        int w_end_valid = min(w_end, W);

        float sum = 0.0f;
        
        // Input pointer offset
        const float* x_ptr = X + (n * C * H * W) + (c * H * W);

        for (int h = h_start_valid; h < h_end_valid; ++h) {
            for (int w = w_start_valid; w < w_end_valid; ++w) {
                sum += x_ptr[h * W + w];
            }
        }

        // 4. Divisor Logic
        float divisor;
        if (count_include_pad) {
            // Divide by full kernel area (including padded zeros)
            divisor = (float)(kern_h * kern_w);
        } else {
            // Divide by only valid pixels
            int valid_h = h_end_valid - h_start_valid;
            int valid_w = w_end_valid - w_start_valid;
            divisor = (float)(valid_h * valid_w);
            // Protect against div-by-zero (though layout calc usually prevents this)
            if (divisor < 1.0f) divisor = 1.0f;
        }

        Y[index] = sum / divisor;
    }
}

void AvgPoolOp::forward_gpu(const std::vector<Tensor *> &inputs,
                            std::vector<Tensor *> &outputs,
                            const onnx::NodeProto &node,
                            cublasHandle_t &handle) {
    
    const Tensor *X = inputs[0];
    Tensor *Y = outputs[0];
    const auto &in_shape = X->shape();

    // 1. Attributes
    auto kernel_shape = get_int_list_attribute(node, "kernel_shape");
    int64_t kern_h = kernel_shape[0];
    int64_t kern_w = kernel_shape[1];

    auto strides = get_int_list_attribute(node, "strides");
    int64_t stride_h = (strides.empty()) ? 1 : strides[0];
    int64_t stride_w = (strides.empty()) ? 1 : strides[1];

    int64_t count_include_pad = get_int_attribute(node, "count_include_pad", 0);
    int64_t ceil_mode = get_int_attribute(node, "ceil_mode", 0);

    // Padding (Asymmetric support)
    int64_t pad_t = 0, pad_l = 0, pad_b = 0, pad_r = 0;
    auto pads = get_int_list_attribute(node, "pads");
    if (!pads.empty()) {
        if (pads.size() == 4) {
            pad_t = pads[0]; pad_l = pads[1]; pad_b = pads[2]; pad_r = pads[3];
        } else if (pads.size() == 2) {
            pad_t = pads[0]; pad_b = pads[0]; pad_l = pads[1]; pad_r = pads[1];
        }
    }
    // (if buggy,maybe it's autopad,i should probably copy it here from max pool)

    // 2. Dimensions
    int64_t N = in_shape[0];
    int64_t C = in_shape[1];
    int64_t H = in_shape[2];
    int64_t W = in_shape[3];

    int64_t out_h, out_w;
    if (ceil_mode != 0) {
        out_h = std::ceil((float)(H + pad_t + pad_b - kern_h) / stride_h) + 1;
        out_w = std::ceil((float)(W + pad_l + pad_r - kern_w) / stride_w) + 1;
    } else {
        out_h = (H + pad_t + pad_b - kern_h) / stride_h + 1;
        out_w = (W + pad_l + pad_r - kern_w) / stride_w + 1;
    }

    if (out_h < 1) out_h = 1;
    if (out_w < 1) out_w = 1;

    Y->reshape({N, C, out_h, out_w});

    // 3. Allocate GPU Output
    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();

    // 4. Launch
    int total_elements = N * C * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    avgpool_kernel<<<blocks, threads>>>(
        (const float*)X->device_data(), 
        (float*)Y->device_data(),
        N, C, H, W,
        out_h, out_w,
        kern_h, kern_w,
        pad_t, pad_l,
        stride_h, stride_w,
        (int)count_include_pad
    );
}
