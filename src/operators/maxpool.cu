#include "operators/maxpool.h"
#include <cuda_runtime.h>
#include <cfloat> // for FLT_MAX
#include <cmath>  // for ceil
#include "debug_utils.h" // for debugging

// CUDA Kernel
__global__ void maxpool_kernel(const float* X, float* Y, 
                               int N, int C, int H, int W,
                               int out_h, int out_w,
                               int kern_h, int kern_w,
                               int pad_t, int pad_l,
                               int stride_h, int stride_w) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * out_h * out_w;

    if (index < total_elements) {
        // 1. Decode Index to (n, c, oh, ow)
        int pw = index % out_w;
        int ph = (index / out_w) % out_h;
        int c  = (index / (out_w * out_h)) % C;
        int n  = index / (out_w * out_h * C);

        // 2. Calculate Input Window (Apply Top/Left Padding)
        int h_start = ph * stride_h - pad_t;
        int w_start = pw * stride_w - pad_l;
        int h_end = h_start + kern_h;
        int w_end = w_start + kern_w;

        // 3. Find Max
        float max_val = -FLT_MAX;
        
        // Input pointer offset for this batch/channel
        const float* x_ptr = X + (n * C * H * W) + (c * H * W);

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                // Check Bounds (Implicitly handles Bottom/Right padding)
                if (h >= 0 && w >= 0 && h < H && w < W) {
                    float val = x_ptr[h * W + w];
                    if (val > max_val) max_val = val;
                }
            }
        }
        Y[index] = max_val;
    }
}

void MaxPoolOp::forward_gpu(const std::vector<Tensor*>& inputs, 
                            std::vector<Tensor*>& outputs, 
                            const onnx::NodeProto& node, 
                            cublasHandle_t& handle) {

    const Tensor* X = inputs[0];
    Tensor* Y = outputs[0];

    // 1. Parse Attributes (Identical to CPU)
    auto kernel_shape = get_int_list_attribute(node, "kernel_shape");
    int64_t kern_h = kernel_shape[0];
    int64_t kern_w = kernel_shape[1];

    auto strides = get_int_list_attribute(node, "strides");
    int64_t stride_h = (strides.empty()) ? 1 : strides[0];
    int64_t stride_w = (strides.empty()) ? 1 : strides[1];
    
    int64_t ceil_mode = get_int_attribute(node, "ceil_mode", 0);

    // 2. Padding Logic
    int64_t pad_t = 0, pad_l = 0, pad_b = 0, pad_r = 0;
    auto pads = get_int_list_attribute(node, "pads");

    if (!pads.empty()) {
        if (pads.size() == 4) {
            pad_t = pads[0]; pad_l = pads[1]; pad_b = pads[2]; pad_r = pads[3];
        } else if (pads.size() == 2) {
            pad_t = pads[0]; pad_b = pads[0]; pad_l = pads[1]; pad_r = pads[1];
        }
    } else {
        std::string auto_pad = get_string_attribute(node, "auto_pad", "NOTSET");
        if (auto_pad != "NOTSET" && auto_pad != "VALID") {
            int64_t H = X->shape()[2];
            int64_t W = X->shape()[3];
            int64_t out_h_temp = std::ceil((float)H / stride_h);
            int64_t out_w_temp = std::ceil((float)W / stride_w);
            int64_t pad_h_needed = (out_h_temp - 1) * stride_h + kern_h - H;
            int64_t pad_w_needed = (out_w_temp - 1) * stride_w + kern_w - W;
            if (pad_h_needed < 0) pad_h_needed = 0;
            if (pad_w_needed < 0) pad_w_needed = 0;

            if (auto_pad == "SAME_UPPER") {
                pad_t = pad_h_needed / 2; pad_b = pad_h_needed - pad_t;
                pad_l = pad_w_needed / 2; pad_r = pad_w_needed - pad_l;
            } else if (auto_pad == "SAME_LOWER") {
                pad_b = pad_h_needed / 2; pad_t = pad_h_needed - pad_b;
                pad_r = pad_w_needed / 2; pad_l = pad_w_needed - pad_r;
            }
        }
    }

    // 3. Calculate Dimensions
    int64_t N = X->shape()[0];
    int64_t C = X->shape()[1];
    int64_t H = X->shape()[2];
    int64_t W = X->shape()[3];

    int64_t out_h, out_w;

    //  Handle Ceil Mode 
    if (ceil_mode != 0) {
        out_h = std::ceil((float)(H + pad_t + pad_b - kern_h) / stride_h) + 1;
        out_w = std::ceil((float)(W + pad_l + pad_r - kern_w) / stride_w) + 1;
    } else {
        out_h = (H + pad_t + pad_b - kern_h) / stride_h + 1;
        out_w = (W + pad_l + pad_r - kern_w) / stride_w + 1;
    }

    // Safety clamp
    if (out_h < 1) out_h = 1;
    if (out_w < 1) out_w = 1;

    Y->reshape({N, C, out_h, out_w});
    
    // 4. Allocate Memory 
    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();

    // 5. Launch
    int total_outputs = N * C * out_h * out_w;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    debug_gpu_tensor("MaxPool Input X", inputs[0]);
    maxpool_kernel<<<blocks, threads>>>(
        (const float*)X->device_data(), (float*)Y->device_data(),
        N, C, H, W,
        out_h, out_w,
        kern_h, kern_w,
        pad_t, pad_l,     // Pass T/L padding
        stride_h, stride_w
    );
    debug_gpu_tensor("maxpool Output", Y);
}
