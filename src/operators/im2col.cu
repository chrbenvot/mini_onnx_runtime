#include <cuda_runtime.h>
#include <cstdint>

// Standard CUDA Kernel for im2col
// Each thread copies ONE pixel from the input image to the column buffer
__global__ void im2col_kernel(const int n, const float* data_im,
                              const int height, const int width,
                              const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int height_col, const int width_col,
                              float* data_col) {
    
    // Grid-Stride Loop
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < n; index += blockDim.x * gridDim.x) {
        
        // Calculate (c, h, w) in the *output column matrix*
        // The index maps to: (c * kH * kW) * (outH * outW) + ...
        
        // 1. Find which part of the kernel we are inside
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        
        int channel_out = channel_in * kernel_h * kernel_w;
        int h_in = h_out * stride_h - pad_h;
        int w_in = w_out * stride_w - pad_w;
        
        // 2. Pointer to the output column buffer
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        
        // 3. Pointer to input image
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        
        // 4. Unroll the kernel window for this specific location
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                
                // Copy pixel if inside bounds, else 0 (padding)
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? 
                                data_im_ptr[i * width + j] : 0;
                
                // Move output pointer to next "channel block"
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

// Wrapper to launch the kernel
void im2col_gpu(const float* data_im, int channels, int height, int width,
                int ksize_h, int ksize_w, int pad_h, int pad_w,
                int stride_h, int stride_w, int height_col, int width_col,
                float* data_col) {
    
    // Total number of elements in the OUTPUT matrix (roughly)
    int num_kernels = channels * height_col * width_col;
    
    int threads = 512;
    int blocks = (num_kernels + threads - 1) / threads;
    
    im2col_kernel<<<blocks, threads>>>(
        num_kernels, data_im, height, width, ksize_h, ksize_w, 
        pad_h, pad_w, stride_h, stride_w, height_col, width_col, data_col
    );
}
