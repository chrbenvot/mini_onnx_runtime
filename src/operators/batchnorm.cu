#include "operators/batchnorm.h"
#include <cuda_runtime.h>
#include "debug_utils.h"

// CUDA Kernel
// Every thread handles one pixel (n,c,h,w)
__global__ void batchnorm_kernel(const float *X,
                                 float *Y,
                                 const float *scale,
                                 const float *B,
                                 const float *mean,
                                 const float *var,
                                 float epsilon,
                                 int N,
                                 int C,
                                 int Spatial) // spatial = H *W
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * Spatial;
    if (index < total_elements)
    {
        // Figure out the channel this index belongs to
        int c = (index / Spatial) % C;
        float val = X[index];
        float mu = mean[c];
        float sigma_sq = var[c];
        float gamma = scale[c];
        float beta = B[c];
        // Opt: pre compute scale_factor = gamma/sqrt(...) in a separate kernel
        float inv_std = rsqrtf(sigma_sq + epsilon); // CUDA fast inverse square
        float norm = (val - mu) * inv_std;
        Y[index] = norm * gamma + beta;
    }
}

void BatchNorm::forward_gpu(const std::vector<Tensor*>& inputs,
std::vector<Tensor*>& outputs,
const onnx::NodeProto& node,
cublasHandle_t& handle){
    const Tensor* X = inputs[0];
    const Tensor* scale = inputs[1];
    const Tensor* B = inputs[2];
    const Tensor* mean = inputs[3];
    const Tensor* var = inputs[4];
    Tensor* Y = outputs[0];
    float epsilon =get_float_attribute(node,"epsilon",1e-5f);

    const auto& shape = X->shape();
    Y->reshape(shape);

    if (Y->is_on_device()) Y->free_device_memory();
    Y->allocate_device_memory();

    int N = shape[0];
    int C = shape[1];
    int H = shape[2];
    int W = shape[3];
    int Spatial = H * W;
    int total_elements = X->size();

    //  Launch Kernel
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads; 

    const float* d_X = (const float*)X->device_data();
    const float* d_scale = (const float*)scale->device_data();
    const float* d_B = (const float*)B->device_data();
    const float* d_mean = (const float*)mean->device_data();
    const float* d_var = (const float*)var->device_data();
    float* d_Y = (float*)Y->device_data();

    batchnorm_kernel<<<blocks,threads>>>(d_X,d_Y,d_scale,d_B,d_mean,d_var,epsilon,N,C,Spatial);
    debug_gpu_tensor("BN Output", Y);
}
