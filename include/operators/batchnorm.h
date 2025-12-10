#pragma once
#include "../operator.h"
#include <cmath>
#include <vector>
#include "../simd_utils.h" // Include SIMD helper

class BatchNorm : public Operator {
public:
std::string get_op_type() const { return "BatchNormalization"; }
    void forward(const std::vector<Tensor*>& inputs, 
                 std::vector<Tensor*>& outputs, 
                 const onnx::NodeProto& node,
                 std::vector<float>& workspace) override {
        
        // 1. Inputs
        // X: [N, C, H, W]
        // Scale (gamma): [C]
        // B (beta): [C]
        // Mean: [C]
        // Var: [C]
        const Tensor* X = inputs[0];
        const Tensor* scale = inputs[1];
        const Tensor* B = inputs[2];
        const Tensor* mean = inputs[3];
        const Tensor* var = inputs[4];
        
        Tensor* Y = outputs[0];
        Y->reshape(X->shape());

        // 2. Attributes
        float epsilon = get_float_attribute(node, "epsilon", 1e-5f);

        // 3. Dimensions
        const auto& shape = X->shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        // Calculate spatial size (H * W)
        int64_t spatial = 1;
        for (size_t i = 2; i < shape.size(); ++i) spatial *= shape[i];

        // 4. Pre-compute Effective Scale & Bias per Channel
        // This avoids doing sqrt/division inside the hot loop
        // We can reuse the workspace if we want, or just a small local vector (C is usually small, e.g. 1024)
        std::vector<float> eff_scale(C);
        std::vector<float> eff_bias(C);

        const float* s_data = scale->data<float>();
        const float* b_data = B->data<float>();
        const float* m_data = mean->data<float>();
        const float* v_data = var->data<float>();

        for (int c = 0; c < C; ++c) {
            // Factor = gamma / sqrt(var + epsilon)
            float factor = s_data[c] / std::sqrt(v_data[c] + epsilon);
            
            eff_scale[c] = factor;
            eff_bias[c]  = b_data[c] - (m_data[c] * factor);
        }

        // 5. Apply to Pixels (SIMD Optimized)
        const float* x_ptr = X->data<float>();
        float* y_ptr = Y->data<float>();

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                // Calculate pointer to the start of this channel's spatial data
                // Offset = (n * C + c) * spatial
                int64_t offset = (n * C + c) * spatial;
                
                const float* in_plane = x_ptr + offset;
                float* out_plane = y_ptr + offset;
                
                // Use the SIMD helper: out = in * scale + bias
                scale_shift_avx(in_plane, out_plane, spatial, eff_scale[c], eff_bias[c]);
            }
        }
    }
};
