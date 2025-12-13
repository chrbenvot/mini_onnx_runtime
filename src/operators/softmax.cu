#include "operators/softmax.h"
#include <cuda_runtime.h>
#include <cfloat> // for -FLT_MAX
#include <math.h> // for expf

// Kernel
// We treat the tensor as a collection of 1D vectors of length D.
// The tensor shape is effectively: [N, D, Inner]
// N = Product of dimensions before axis
// D = Dimension at axis
// Inner = Product of dimensions after axis
// Total independent vectors = N * Inner

__global__ void softmax_kernel(const float* input, float* output, 
                               int64_t count, // Total threads (N * Inner)
                               int64_t D, 
                               int64_t inner) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        // 1. Map linear thread index to (n, k) coordinates
        // idx = n * inner + k
        int64_t n = idx / inner;
        int64_t k = idx % inner;

        // Base offset for this specific vector
        // The vector elements are at: base + 0*inner, base + 1*inner, ...
        int64_t offset = n * (D * inner) + k;

        // 2. Find Max (for numerical stability)
        float max_val = -FLT_MAX;
        for (int j = 0; j < D; ++j) {
            float val = input[offset + j * inner];
            if (val > max_val) max_val = val;
        }

        // 3. Calculate Exponentials and Sum
        float sum = 0.0f;
        for (int j = 0; j < D; ++j) {
            float val = input[offset + j * inner];
            float exp_val = expf(val - max_val); // GPU fast exp
            
            // Store intermediate result in output to avoid re-calculating exp later
            output[offset + j * inner] = exp_val;
            sum += exp_val;
        }

        // 4. Normalize
        for (int j = 0; j < D; ++j) {
            output[offset + j * inner] /= sum;
        }
    }
}

// Implementation

void SoftmaxOp::forward_gpu(const std::vector<Tensor *> &inputs,
                            std::vector<Tensor *> &outputs,
                            const onnx::NodeProto &node,
                            cublasHandle_t &handle) {
    
    const Tensor *input = inputs[0];
    Tensor *output = outputs[0];
    output->reshape(input->shape());

    // 1. Parse Dimensions (Same logic as CPU)
    int64_t axis = get_int_attribute(node, "axis", -1);
    if (axis < 0) axis += input->shape().size();

    int64_t N = 1;
    for (int i = 0; i < axis; ++i) N *= input->shape()[i];

    int64_t D = input->shape()[axis];

    int64_t inner = 1;
    for (size_t i = axis + 1; i < input->shape().size(); ++i) inner *= input->shape()[i];

    // 2. Allocate Output
    if (output->is_on_device()) output->free_device_memory();
    output->allocate_device_memory();

    // 3. Launch Kernel
    // We launch one thread for every independent vector (N * Inner)
    int64_t total_vectors = N * inner;
    int threads = 256;
    int blocks = (total_vectors + threads - 1) / threads;

    softmax_kernel<<<blocks, threads>>>(
        (const float*)input->device_data(),
        (float*)output->device_data(),
        total_vectors,
        D,
        inner
    );
}
