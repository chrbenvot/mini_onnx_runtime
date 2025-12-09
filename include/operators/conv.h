#pragma once
#include "../operator.h"
#include <cmath>

class ConvOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs, const onnx::NodeProto &node) override
    {
        const Tensor *X = inputs[0];
        const Tensor *W = inputs[1];

        // Parse attributes
        auto pads = get_int_list_attribute(node, "pads");
        int64_t pad_h = (pads.empty()) ? 0 : pads[0];
        int64_t pad_w = (pads.empty()) ? 0 : pads[1];
        auto strides = get_int_list_attribute(node, "strides");
        int64_t stride_h = (strides.empty()) ? 1 : strides[0];
        int64_t stride_w = (strides.empty()) ? 1 : strides[1]; // TODO: add support for more attributes ( stride...)

        // Get dimensions
        const auto &x_shape = X->shape();
        const auto &w_shape = W->shape();
        int64_t batch = x_shape[0];
        int64_t in_c = x_shape[1];
        int64_t in_h = x_shape[2];
        int64_t in_w = x_shape[3];

        int64_t out_c = w_shape[0];
        int64_t kern_h = w_shape[2];
        int64_t kern_w = w_shape[3];

        // Calculate output dimensions
        // The formula is : Output = (Input +2*pad - kernel)/stride +1
        int64_t out_h = (in_h + 2 * pad_h - kern_h) / stride_h + 1;
        int64_t out_w = (in_w + 2 * pad_w - kern_w) / stride_w + 1;

        // Initialize output Tensor
        Tensor *Y = outputs[0];
        Y->reshape({batch, out_c, out_h, out_w});
        // OPTIONAL: zero out output?
        const float *x_data = X->data<float>(); // TODO: fuck it,we'll just work with floats for now and refactor for polymorphism later
        const float *w_data = W->data<float>();
        const float *b_data = (inputs.size() > 2) ? inputs[2]->data<float>() : nullptr; // bias if it exists
        float *y_data = Y->data<float>();

        // Naive Conv loop
        // The order is batch->OutChannel/>outY->outX
        for (int b = 0; b < batch; ++b)
        {
            for (int oc = 0; oc < out_c; ++oc)
            {
                // Get channel bias
                float bias_val = (b_data) ? b_data[oc] : 0.0f;

                for (int oy = 0; oy < out_h; ++oy)
                {
                    for (int ox = 0; ox < out_w; ++ox)
                    {
                        float sum = 0.0f;

                        // Inner loops : (InChannel->Kernel Y->KernelX)
                        for (int ic = 0; ic < in_c; ++ic)
                        {
                            for (int ky = 0; ky < kern_h; ++ky)
                            {
                                for (int kx = 0; kx < kern_w; ++kx)
                                {
                                    // Calculate Input coordinates
                                    int64_t iy = oy * stride_h - pad_h + ky;
                                    int64_t ix = ox * stride_w - pad_w + kx;
                                    // Padding boundary check
                                    if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w)
                                    {
                                        // Calculate the flattened index
                                        // X index: b * (C*H*W) + ic * (H*W) + iy * W + ix
                                        int64_t x_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + iy * (in_w) + ix;

                                        // W index: oc * (C*KH*KW) + ic * (KH*KW) + ky * KW + kx
                                        int64_t w_idx = oc * (in_c * kern_h * kern_w) + ic * (kern_h * kern_w) + ky * (kern_w) + kx;

                                        sum += x_data[x_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }
                        // Add the bias and write to output
                        // Y index: b * (OC*OH*OW) + oc * (OH*OW) + oy * OW + ox
                        int64_t y_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + oy * (out_w) + ox;

                        y_data[y_idx] = sum + bias_val;
                    }
                }
            }
        }
    }
};
