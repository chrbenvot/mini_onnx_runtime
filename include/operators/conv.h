#pragma once
#include "../operator.h"
#include <cmath>
#include <vector>
#include <cstring>

class ConvOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node) override
    {

        const Tensor *X = inputs[0];
        const Tensor *W = inputs[1];
        const Tensor *B = (inputs.size() > 2) ? inputs[2] : nullptr;
        Tensor *Y = outputs[0];

        // 1. Attributes
        auto strides = get_int_list_attribute(node, "strides");
        int64_t stride_h = (strides.empty()) ? 1 : strides[0];
        int64_t stride_w = (strides.empty()) ? 1 : strides[1];

        auto pads = get_int_list_attribute(node, "pads");
        int64_t pad_h = 0, pad_w = 0;

        // Handle Padding Logic
        if (!pads.empty())
        {
            pad_h = pads[0];
            pad_w = pads[1];
        }
        else
        {
            std::string auto_pad = get_string_attribute(node, "auto_pad", "NOTSET");
            if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER")
            {
                pad_h = (W->shape()[2] - 1) / 2;
                pad_w = (W->shape()[3] - 1) / 2;
            }
        }

        // 2. Dimensions
        int64_t N = X->shape()[0];
        int64_t C = X->shape()[1];
        int64_t H = X->shape()[2];
        int64_t width = X->shape()[3]; // 'W' is taken by Weights Tensor name

        int64_t out_c = W->shape()[0];
        int64_t in_c = W->shape()[1]; // Should equal C / groups (assuming group=1 for now)
        int64_t kern_h = W->shape()[2];
        int64_t kern_w = W->shape()[3];

        int64_t out_h = (H + 2 * pad_h - kern_h) / stride_h + 1;
        int64_t out_w = (width + 2 * pad_w - kern_w) / stride_w + 1;

        Y->reshape({N, out_c, out_h, out_w});

        // 3. Allocations
        // im2col Buffer: [in_c * kern_h * kern_w, out_h * out_w]
        // This is potentially large, so we reuse it per batch item.
        int64_t col_buffer_size = (in_c * kern_h * kern_w) * (out_h * out_w);
        std::vector<float> col_buffer(col_buffer_size);

        const float *w_data = W->data<float>();
        const float *b_data = (B) ? B->data<float>() : nullptr;
        float *y_data = Y->data<float>();

        // 4. Processing Loop (Per Image in Batch)
        for (int n = 0; n < N; ++n)
        {
            const float *x_data = X->data<float>() + n * (C * H * width);
            float *y_out_ptr = y_data + n * (out_c * out_h * out_w);

            // A. Perform im2col
            // Turns the convolution input into a matrix multiplication problem
            im2col_cpu(x_data, C, H, width, kern_h, kern_w, pad_h, pad_w, stride_h, stride_w, out_h, out_w, col_buffer.data());

            // B. GEMM (General Matrix Multiply)
            // Y (output) = Weights x ColBuffer
            // Matrix A (Weights):   [out_c, in_c * kh * kw]
            // Matrix B (ColBuffer): [in_c * kh * kw, out_h * out_w]
            // Result C (Y):         [out_c, out_h * out_w]

            int64_t M_gemm = out_c;
            int64_t K_gemm = in_c * kern_h * kern_w;
            int64_t N_gemm = out_h * out_w;

            // Simple GEMM Loop (Can be optimized with SIMD later)
            for (int m = 0; m < M_gemm; ++m)
            {
                for (int i = 0; i < N_gemm; ++i)
                {
                    float sum = 0.0f;
                    // Pre-fetch Bias
                    if (b_data)
                        sum = b_data[m];

                    // Dot Product
                    // Optimizing this inner loop is the key to performance (AVX/SIMD)
                    for (int k = 0; k < K_gemm; ++k)
                    {
                        sum += w_data[m * K_gemm + k] * col_buffer[k * N_gemm + i];
                    }

                    y_out_ptr[m * N_gemm + i] = sum;
                }
            }
        }
    }

private:
    // Helper: Rearrange image data into columns
    void im2col_cpu(const float *data_im,
                    int channels, int height, int width,
                    int kernel_h, int kernel_w,
                    int pad_h, int pad_w,
                    int stride_h, int stride_w,
                    int output_h, int output_w,
                    float *data_col)
    {

        // Loop over every pixel location in the kernel volume (C * K_H * K_W)
        // This determines the "Row" in the column matrix
        int channels_col = channels * kernel_h * kernel_w;

        for (int c = 0; c < channels_col; ++c)
        {
            // Determine which input pixel (c_im, k_h, k_w) contributes to this row
            int w_offset = c % kernel_w;
            int h_offset = (c / kernel_w) % kernel_h;
            int c_im = c / (kernel_w * kernel_h);

            // Loop over every sliding window position (Output H * Output W)
            // This determines the "Column" in the column matrix
            for (int h = 0; h < output_h; ++h)
            {
                for (int w = 0; w < output_w; ++w)
                {

                    // Calculate input coordinates
                    int im_row = h * stride_h - pad_h + h_offset;
                    int im_col = w * stride_w - pad_w + w_offset;

                    // Determine index in output buffer
                    // Row-Major: (c * output_h * output_w) + (h * output_w) + w
                    int col_index = (c * output_h * output_w) + (h * output_w) + w;

                    // Check bounds (Padding)
                    if (im_row > -1 && im_col > -1 && im_row < height && im_col < width)
                    {
                        // Copy valid pixel
                        int im_index = (c_im * height + im_row) * width + im_col;
                        data_col[col_index] = data_im[im_index];
                    }
                    else
                    {
                        // Padding (Zero)
                        data_col[col_index] = 0;
                    }
                }
            }
        }
    }
};
