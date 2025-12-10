#pragma once
#include "../operator.h"
#include "../simd_utils.h" // Must include your AVX helper
#include <cmath>
#include <vector>
#include <string>
#include <cstring>

class ConvOp : public Operator
{
public:
    void forward(const std::vector<Tensor *> &inputs,
                 std::vector<Tensor *> &outputs,
                 const onnx::NodeProto &node, std::vector<float> &workspace) override
    {

        const Tensor *X = inputs[0];
        const Tensor *W = inputs[1];
        const Tensor *B = (inputs.size() > 2) ? inputs[2] : nullptr;
        Tensor *Y = outputs[0];

        // 1. Parse Attributes
        auto strides = get_int_list_attribute(node, "strides");
        int64_t stride_h = 1;
        int64_t stride_w = 1;
        if (strides.size() >= 1)
            stride_h = strides[0];
        if (strides.size() >= 2)
            stride_w = strides[1];

        auto pads = get_int_list_attribute(node, "pads");
        int64_t pad_h = 0, pad_w = 0;

        if (!pads.empty())
        {
            if (pads.size() == 2)
            {
                pad_h = pads[0];
                pad_w = pads[1];
            }
            else if (pads.size() == 4)
            {
                // [top, left, bottom, right]
                pad_h = pads[0];
                pad_w = pads[1];
            }
        }
        else
        {
            std::string auto_pad = get_string_attribute(node, "auto_pad", "NOTSET");
            if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER")
            {
                int64_t kern_h = W->shape()[2];
                int64_t kern_w = W->shape()[3];
                // For stride=1, pad = (k-1)/2
                // For general: Output = ceil(Input/Stride)
                // We simplify for typical YOLO usage (Usually Stride=1 or Symmetric)
                pad_h = (kern_h - 1) / 2;
                pad_w = (kern_w - 1) / 2;
            }
        }

        // 2. Resolve Dimensions
        int64_t N = X->shape()[0];
        int64_t C = X->shape()[1];
        int64_t H = X->shape()[2];
        int64_t width = X->shape()[3];

        int64_t out_c = W->shape()[0];
        int64_t in_c = W->shape()[1];
        int64_t kern_h = W->shape()[2];
        int64_t kern_w = W->shape()[3];

        int64_t out_h = (H + 2 * pad_h - kern_h) / stride_h + 1;
        int64_t out_w = (width + 2 * pad_w - kern_w) / stride_w + 1;

        Y->reshape({N, out_c, out_h, out_w});

        // 3. Prepare Allocations
        // K_gemm = Size of a single 3D kernel patch (e.g. 3*3*3 = 27)
        // N_gemm = Number of sliding windows (e.g. 13*13 = 169)
        int64_t M_gemm = out_c;
        int64_t K_gemm = in_c * kern_h * kern_w;
        int64_t N_gemm = out_h * out_w;

        // Buffer: [N_gemm, K_gemm] (Transposed layout for SIMD)
        int64_t col_buffer_size = K_gemm * N_gemm;
        // Optimization: Keep this vector static/thread-local to avoid allocs if you want,
        // but for now local is safer.
        if (workspace.size() < col_buffer_size)
        {
            workspace.resize(col_buffer_size);
        }
        float* col_buffer_ptr = workspace.data();

        const float *w_data = W->data<float>();
        const float *b_data = (B) ? B->data<float>() : nullptr;
        float *y_data = Y->data<float>();

        // 4. Processing Loop (Per Batch Item)
        for (int n = 0; n < N; ++n)
        {
            const float *x_data = X->data<float>() + n * (C * H * width);
            float *y_out_ptr = y_data + n * (out_c * out_h * out_w);

            // A. Perform im2col (Writing Transposed!)
            im2col_transposed(x_data, C, H, width, kern_h, kern_w, pad_h, pad_w, stride_h, stride_w, out_h, out_w, col_buffer_ptr);

            // B. SIMD GEMM
            // Weights Matrix (Rows) dot Transposed Col Buffer (Rows)
            #pragma omp parallel for    // multithreading
            for (int m = 0; m < M_gemm; ++m)
            {
                const float *w_row = w_data + m * K_gemm; // Contiguous Weights
                float bias = (b_data) ? b_data[m] : 0.0f;

                for (int i = 0; i < N_gemm; ++i)
                {
                    const float *col_row = col_buffer_ptr + i * K_gemm; // Contiguous Patch

                    // --- AVX ACCELERATION ---
                    // Dot product of two contiguous float arrays of length K_gemm
                    float val = dot_product_avx(w_row, col_row, K_gemm);

                    y_out_ptr[m * N_gemm + i] = val + bias;
                }
            }
        }
    }

private:
    // Writes image patches into rows [PixelIndex, KernelIndex]
    // This allows the GEMM inner loop to read contiguous memory.
    void im2col_transposed(const float *data_im,
                           int channels, int height, int width,
                           int kernel_h, int kernel_w,
                           int pad_h, int pad_w,
                           int stride_h, int stride_w,
                           int output_h, int output_w,
                           float *data_col)
    {

        int channels_col = channels * kernel_h * kernel_w;

        for (int c = 0; c < channels_col; ++c)
        {
            int w_offset = c % kernel_w;
            int h_offset = (c / kernel_w) % kernel_h;
            int c_im = c / (kernel_w * kernel_h);

            for (int h = 0; h < output_h; ++h)
            {
                for (int w = 0; w < output_w; ++w)
                {
                    int im_row = h * stride_h - pad_h + h_offset;
                    int im_col = w * stride_w - pad_w + w_offset;

                    // Transposed Index Mapping:
                    // Row (Outer) = h * width_out + w  (The specific window index)
                    // Col (Inner) = c                  (The specific kernel weight index)
                    int col_index = (h * output_w + w) * channels_col + c;

                    if (im_row > -1 && im_col > -1 && im_row < height && im_col < width)
                    {
                        data_col[col_index] = data_im[(c_im * height + im_row) * width + im_col];
                    }
                    else
                    {
                        data_col[col_index] = 0;
                    }
                }
            }
        }
    }
};
