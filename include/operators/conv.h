#pragma once
#include "../operator.h"
#include <cmath>
#include <string>

class ConvOp : public Operator {
public:
    void forward(const std::vector<Tensor*>& inputs, 
                 std::vector<Tensor*>& outputs, 
                 const onnx::NodeProto& node) override {
        const Tensor* X = inputs[0];
        const Tensor* W = inputs[1];
        
        // 1. Get Attributes
        auto strides = get_int_list_attribute(node, "strides");
        int64_t stride_h = (strides.empty()) ? 1 : strides[0];
        int64_t stride_w = (strides.empty()) ? 1 : strides[1];

        // Read explicit pads
        auto pads = get_int_list_attribute(node, "pads");
        int64_t pad_h = 0, pad_w = 0;

        // Check for Auto Pad
        std::string auto_pad = "";
        for (const auto& attr : node.attribute()) {
            if (attr.name() == "auto_pad") {
                auto_pad = attr.s(); // Get string attribute
                break;
            }
        }

        const auto& x_shape = X->shape();
        const auto& w_shape = W->shape();

        int64_t in_h = x_shape[2];
        int64_t in_w = x_shape[3];
        int64_t kern_h = w_shape[2];
        int64_t kern_w = w_shape[3];

        // 2. Resolve Padding Logic
        if (!pads.empty()) {
            // Explicit pads provided
            if (pads.size() == 2) {
                pad_h = pads[0];
                pad_w = pads[1];
            } else if (pads.size() == 4) {
                // [top, left, bottom, right]
                // We simplify and assume symmetric for this engine, taking the max or average
                pad_h = pads[0]; 
                pad_w = pads[1];
            }
        } 
        else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
            // Calculate "Same" padding
            // Output = Input / Stride (ceil)
            // We need: (Input + 2*P - Kernel) / Stride + 1 == Input / Stride
            // Simplified for Stride 1: 2*P = Kernel - 1
            if (stride_h == 1) pad_h = (kern_h - 1) / 2;
            if (stride_w == 1) pad_w = (kern_w - 1) / 2;
        }

        // 3. Calculate Output Dimensions
        int64_t batch = x_shape[0];
        int64_t out_c = w_shape[0];
        int64_t in_c  = x_shape[1];

        int64_t out_h = (in_h + 2 * pad_h - kern_h) / stride_h + 1;
        int64_t out_w = (in_w + 2 * pad_w - kern_w) / stride_w + 1;

        Tensor* Y = outputs[0];
        Y->reshape({batch, out_c, out_h, out_w});

        const float* x_data = X->data<float>();
        const float* w_data = W->data<float>();
        const float* b_data = (inputs.size() > 2) ? inputs[2]->data<float>() : nullptr;
        float* y_data = Y->data<float>();

        // 4. The Loop
        for (int b = 0; b < batch; ++b) {
            for (int oc = 0; oc < out_c; ++oc) {
                float bias_val = (b_data) ? b_data[oc] : 0.0f;
                for (int oy = 0; oy < out_h; ++oy) {
                    for (int ox = 0; ox < out_w; ++ox) {
                        float sum = 0.0f;
                        for (int ic = 0; ic < in_c; ++ic) {
                            for (int ky = 0; ky < kern_h; ++ky) {
                                for (int kx = 0; kx < kern_w; ++kx) {
                                    int64_t iy = oy * stride_h - pad_h + ky;
                                    int64_t ix = ox * stride_w - pad_w + kx;

                                    if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                                        int64_t x_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + iy * in_w + ix;
                                        int64_t w_idx = oc * (in_c * kern_h * kern_w) + ic * (kern_h * kern_w) + ky * kern_w + kx;
                                        sum += x_data[x_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }
                        int64_t y_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + oy * out_w + ox;
                        y_data[y_idx] = sum + bias_val;
                    }
                }
            }
        }
    }
};
