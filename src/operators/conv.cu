#include "operators/conv.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "debug_utils.h" // for debugging

// Forward declare the helper from im2col.cu
void im2col_gpu(const float* data_im, int channels, int height, int width,
                int ksize_h, int ksize_w, int pad_h, int pad_w,
                int stride_h, int stride_w, int height_col, int width_col,
                float* data_col);

void ConvOp::forward_gpu(const std::vector<Tensor*>& inputs, 
                         std::vector<Tensor*>& outputs, 
                         const onnx::NodeProto& node, 
                         cublasHandle_t& handle) {
    
    const Tensor* X = inputs[0];
    const Tensor* W = inputs[1];
    // Bias is technically inputs[2] if present, usually fused or handled separately
    Tensor* Y = outputs[0];

    // 1. Attributes & Dimensions
    auto strides = get_int_list_attribute(node, "strides");
    int64_t stride_h = (strides.empty()) ? 1 : strides[0];
    int64_t stride_w = (strides.empty()) ? 1 : strides[1];
    
    auto pads = get_int_list_attribute(node, "pads");
    int64_t pad_h = 0;
    int64_t pad_w = 0;

    if (!pads.empty()) {
        pad_h = pads[0];
        pad_w = pads[1];
    } else {
        std::string auto_pad = get_string_attribute(node, "auto_pad", "NOTSET");
        if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
            int64_t kern_h = W->shape()[2];
            int64_t kern_w = W->shape()[3];
            pad_h = (kern_h - 1) / 2;
            pad_w = (kern_w - 1) / 2;
        }
    }

    int64_t N = X->shape()[0];
    int64_t C = X->shape()[1];
    int64_t H = X->shape()[2];
    int64_t width = X->shape()[3]; 

    int64_t out_c = W->shape()[0];
    int64_t in_c  = W->shape()[1];
    int64_t kern_h = W->shape()[2];
    int64_t kern_w = W->shape()[3];

    int64_t out_h = (H + 2 * pad_h - kern_h) / stride_h + 1;
    int64_t out_w = (width + 2 * pad_w - kern_w) / stride_w + 1;

    // 2. Reshape and Allocate Output
    Y->reshape({N, out_c, out_h, out_w});
    
    // Ensure GPU memory is allocated for the new shape
    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();

    // 3. Get Device Pointers
    const float* d_X = (const float*)X->device_data();
    const float* d_W = (const float*)W->device_data();
    float* d_Y = (float*)Y->device_data();

    // 4. Allocate Workspace (Column Buffer)
    // Size: [in_c * kern_h * kern_w] x [out_h * out_w]
    int64_t col_buffer_size = (in_c * kern_h * kern_w) * (out_h * out_w);
    float* d_col_buffer;
    cudaMalloc((void**)&d_col_buffer, col_buffer_size * sizeof(float));

    // 5. Processing Loop (Per Batch)
    for (int n = 0; n < N; ++n) {
        
        // Offset input pointer to current image
        const float* d_X_batch = d_X + n * (C * H * width);
        
        // A. Expand Image to Columns (im2col)
        im2col_gpu(d_X_batch, C, H, width, kern_h, kern_w, 
                   pad_h, pad_w, stride_h, stride_w, out_h, out_w, 
                   d_col_buffer);
        
        // B. Matrix Multiplication (GEMM via cuBLAS)
        // Y = Weights * ColBuffer
        // cuBLAS is Column-Major, so we calculate C^T = B^T * A^T
        // Effectively passing B (ColBuffer) first, then A (Weights)
        
        int m_gemm = out_c;
        int k_gemm = in_c * kern_h * kern_w;
        int n_gemm = out_h * out_w;
        
        float alpha = 1.0f;
        float beta = 0.0f; 
        
        float* d_Y_batch = d_Y + n * (out_c * out_h * out_w);

        cublasSgemm(handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    n_gemm, m_gemm, k_gemm, 
                    &alpha, 
                    d_col_buffer, n_gemm,   // "B"
                    d_W, k_gemm,            // "A"
                    &beta, 
                    d_Y_batch, n_gemm);     // "C"
    }

    // 6. Cleanup Workspace
    cudaFree(d_col_buffer);
    debug_gpu_tensor("Conv Output", Y);
}
