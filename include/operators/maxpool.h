#pragma once
#include "../operator.h"
#include <vector>
#include <algorithm>
#include <cfloat>
#include <cmath>

class MaxPoolOp : public Operator {
public:
    void forward(const std::vector<Tensor*>& inputs, 
                 std::vector<Tensor*>& outputs, 
                 const onnx::NodeProto& node) override {
        
        const Tensor* X = inputs[0];
        Tensor* Y = outputs[0];

        // 1. Get Attributes
        auto kernel_shape = get_int_list_attribute(node, "kernel_shape");
        int64_t kern_h = kernel_shape[0];
        int64_t kern_w = kernel_shape[1];

        auto strides = get_int_list_attribute(node, "strides");
        int64_t stride_h = (strides.empty()) ? 1 : strides[0];
        int64_t stride_w = (strides.empty()) ? 1 : strides[1];

        // 2. Handle Padding (Asymmetric + Auto_Pad)
        auto pads = get_int_list_attribute(node, "pads");
        
        // Defaults
        int64_t pad_top = 0, pad_left = 0, pad_bottom = 0, pad_right = 0;

        if (!pads.empty()) {
            if (pads.size() == 4) {
                // [top, left, bottom, right]
                pad_top = pads[0];
                pad_left = pads[1];
                pad_bottom = pads[2];
                pad_right = pads[3];
            } else if (pads.size() == 2) {
                // Symmetric
                pad_top = pads[0]; pad_bottom = pads[0];
                pad_left = pads[1]; pad_right = pads[1];
            }
        } else {
            // Check for Auto Pad if no explicit pads
            std::string auto_pad = "";
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "auto_pad") {
                    auto_pad = attr.s();
                    break;
                }
            }
            if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
                // Same padding logic: Output = ceil(Input / Stride)
                // Total Pad = (Output - 1) * Stride + Kernel - Input
                // We simplify for the common case where Output == Input (Stride 1)
                if (stride_h == 1) {
                    int64_t total_pad = kern_h - 1;
                    pad_top = total_pad / 2;
                    pad_bottom = total_pad - pad_top;
                }
                if (stride_w == 1) {
                    int64_t total_pad = kern_w - 1;
                    pad_left = total_pad / 2;
                    pad_right = total_pad - pad_left;
                }
            }
        }

        // 3. Output Dimensions
        const auto& in_shape = X->shape();
        int64_t N = in_shape[0];
        int64_t C = in_shape[1];
        int64_t H = in_shape[2];
        int64_t W = in_shape[3];

        // Proper formula using explicit top/bottom pads
        int64_t out_h = (H + pad_top + pad_bottom - kern_h) / stride_h + 1;
        int64_t out_w = (W + pad_left + pad_right - kern_w) / stride_w + 1;

        Y->reshape({N, C, out_h, out_w});

        const float* x_data = X->data<float>();
        float* y_data = Y->data<float>();

        // 4. The Loop
        // Initialize with lowest float
        std::fill(y_data, y_data + Y->size(), -FLT_MAX);

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int oy = 0; oy < out_h; ++oy) {
                    for (int ox = 0; ox < out_w; ++ox) {
                        
                        float max_val = -FLT_MAX;

                        for (int ky = 0; ky < kern_h; ++ky) {
                            for (int kx = 0; kx < kern_w; ++kx) {
                                
                                // Coordinate mapping uses pad_top / pad_left
                                int64_t iy = oy * stride_h - pad_top + ky;
                                int64_t ix = ox * stride_w - pad_left + kx;

                                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                    int64_t idx = n * (C*H*W) + c * (H*W) + iy * W + ix;
                                    float val = x_data[idx];
                                    if (val > max_val) max_val = val;
                                }
                            }
                        }
                        
                        int64_t out_idx = n * (C*out_h*out_w) + c * (out_h*out_w) + oy * out_w + ox;
                        y_data[out_idx] = max_val;
                    }
                }
            }
        }
    }
};
