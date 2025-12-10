#pragma once
#include "../operator.h"
#include <vector>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>

class MaxPoolOp : public Operator {
public:
    void forward(const std::vector<Tensor*>& inputs, 
                 std::vector<Tensor*>& outputs, 
                 const onnx::NodeProto& node,std::vector<float>& workspace) override {
        
        const Tensor* X = inputs[0];
        Tensor* Y = outputs[0];
        const auto& in_shape = X->shape(); // [N, C, H, W]

        //  Parse Attributes
        auto kernel_shape = get_int_list_attribute(node, "kernel_shape");
        int64_t kern_h = kernel_shape[0];
        int64_t kern_w = kernel_shape[1];

        auto strides = get_int_list_attribute(node, "strides");
        int64_t stride_h = (strides.empty()) ? 1 : strides[0];
        int64_t stride_w = (strides.empty()) ? 1 : strides[1];

        int64_t ceil_mode = get_int_attribute(node, "ceil_mode", 0);
        int64_t storage_order = get_int_attribute(node, "storage_order", 0); // 0 is Row Major (Standard)

        // Padding Logic (Explicit vs Auto)
        int64_t pad_t = 0, pad_l = 0, pad_b = 0, pad_r = 0;
        
        // Check for explicit 'pads' attribute first
        auto pads = get_int_list_attribute(node, "pads");

        if (!pads.empty()) {
            if (pads.size() == 4) {
                // [top, left, bottom, right]
                pad_t = pads[0]; pad_l = pads[1]; 
                pad_b = pads[2]; pad_r = pads[3];
            } else if (pads.size() == 2) {
                // Symmetric [h, w]
                pad_t = pads[0]; pad_b = pads[0];
                pad_l = pads[1]; pad_r = pads[1];
            }
        } else {
            // Check 'auto_pad' if 'pads' is missing
            std::string auto_pad = get_string_attribute(node, "auto_pad", "NOTSET");
            
            if (auto_pad != "NOTSET" && auto_pad != "VALID") {
                // Calculate Output size based on Ceil(Input / Stride)
                int64_t out_h_temp = std::ceil((float)in_shape[2] / stride_h);
                int64_t out_w_temp = std::ceil((float)in_shape[3] / stride_w);
                
                // Calculate total padding needed
                int64_t pad_h_needed = (out_h_temp - 1) * stride_h + kern_h - in_shape[2];
                int64_t pad_w_needed = (out_w_temp - 1) * stride_w + kern_w - in_shape[3];

                // Ensure non-negative
                pad_h_needed = std::max(pad_h_needed, (int64_t)0);
                pad_w_needed = std::max(pad_w_needed, (int64_t)0);

                if (auto_pad == "SAME_UPPER") {
                    pad_t = pad_h_needed / 2;
                    pad_b = pad_h_needed - pad_t; // Remaining goes to bottom
                    pad_l = pad_w_needed / 2;
                    pad_r = pad_w_needed - pad_l; // Remaining goes to right
                } else if (auto_pad == "SAME_LOWER") {
                    pad_b = pad_h_needed / 2;
                    pad_t = pad_h_needed - pad_b; // Remaining goes to top
                    pad_r = pad_w_needed / 2;
                    pad_l = pad_w_needed - pad_r; // Remaining goes to left
                }
            }
        }

        // Calculate Output Dimensions
        
        int64_t N = in_shape[0];
        int64_t C = in_shape[1];
        int64_t H = in_shape[2];
        int64_t W = in_shape[3];

        int64_t out_h, out_w;

        if (ceil_mode != 0) {
            // Use float division + ceil
            out_h = std::ceil((float)(H + pad_t + pad_b - kern_h) / stride_h) + 1;
            out_w = std::ceil((float)(W + pad_l + pad_r - kern_w) / stride_w) + 1;
        } else {
            // Use integer division (floor)
            out_h = (H + pad_t + pad_b - kern_h) / stride_h + 1;
            out_w = (W + pad_l + pad_r - kern_w) / stride_w + 1;
        }

        // Safety clamp (in case padding is huge or calculation goes wrong)
        if (out_h < 1) out_h = 1; 
        if (out_w < 1) out_w = 1;

        Y->reshape({N, C, out_h, out_w});

        // Execution Loop
        
        const float* x_data = X->data<float>();
        float* y_data = Y->data<float>();
        
        // Fill output with lowest possible float to handle "dead" padding zones
        std::fill(y_data, y_data + Y->size(), -FLT_MAX);

        for (int n = 0; n < N; ++n) {
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < C; ++c) {
                for (int oy = 0; oy < out_h; ++oy) {
                    for (int ox = 0; ox < out_w; ++ox) {
                        
                        float max_val = -FLT_MAX;
                        bool found_valid_pixel = false;

                        // Loop over kernel
                        for (int ky = 0; ky < kern_h; ++ky) {
                            for (int kx = 0; kx < kern_w; ++kx) {
                                
                                int64_t iy = oy * stride_h - pad_t + ky;
                                int64_t ix = ox * stride_w - pad_l + kx;

                                // Boundary Check
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                    int64_t idx = n * (C * H * W) + c * (H * W) + iy * W + ix;
                                    float val = x_data[idx];
                                    if (val > max_val) max_val = val;
                                    found_valid_pixel = true;
                                }
                            }
                        }
                        
                        // If the window was completely in padding (unlikely but possible),
                        // max_val stays -FLT_MAX, which is correct for ONNX.
                        int64_t out_idx = n * (C * out_h * out_w) + c * (out_h * out_w) + oy * out_w + ox;
                        y_data[out_idx] = max_val;
                    }
                }
            }
        }
    }
};
