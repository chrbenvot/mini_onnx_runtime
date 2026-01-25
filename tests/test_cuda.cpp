#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "tensor.h"
#include "operators/relu.h"
#include "operators/leakyrelu.h"
#include "operators/sigmoid.h"
#include "operators/add.h"
#include "operators/mul.h"
#include "operators/batchnorm.h"
#include "operators/maxpool.h"
#include "operators/avgpool.h"
#include "operators/global_avgpool.h"
#include "operators/concat.h"
#include "operators/upsample.h"
#include "operators/softmax.h"
#include "operators/flatten.h"
#include "operators/conv.h" 

// --- Helper: RAII Handle for cuBLAS ---
struct TestHandle {
    cublasHandle_t h;
    TestHandle() { cublasCreate(&h); }
    ~TestHandle() { cublasDestroy(h); }
    operator cublasHandle_t&() { return h; }
};

// --- Helper: Attribute Builder ---
void add_int_attr(onnx::NodeProto& node, const std::string& name, int64_t val) {
    auto* attr = node.add_attribute();
    attr->set_name(name);
    attr->set_i(val);
}

void add_float_attr(onnx::NodeProto& node, const std::string& name, float val) {
    auto* attr = node.add_attribute();
    attr->set_name(name);
    attr->set_f(val);
}

void add_ints_attr(onnx::NodeProto& node, const std::string& name, const std::vector<int64_t>& vals) {
    auto* attr = node.add_attribute();
    attr->set_name(name);
    for (auto v : vals) attr->add_ints(v);
}

//  Helper: Compare Floats 
bool check_close(float a, float b, float tol = 1e-3f) {
    return std::abs(a - b) < tol;
}

//
// TEST IMPLEMENTATIONS
// 

void test_elementwise_activations() {
    std::cout << "[Test] Element-wise Activations (Relu, Leaky, Sigmoid)..." << std::endl;
    TestHandle handle;
    std::vector<float> ws;
    
    // Data: [-1.0, 0.0, 1.0, 2.0]
    Tensor input(DataType::FLOAT32, {1, 4});
    float* d = input.data<float>();
    d[0] = -1.0f; d[1] = 0.0f; d[2] = 1.0f; d[3] = 2.0f;
    
    input.allocate_device_memory();
    input.copy_to_device();

    // 1. RELU
    {
        Tensor out(DataType::FLOAT32, {1, 4});
        ReluOp op;
        std::vector<Tensor*> in = {&input}; std::vector<Tensor*> res = {&out};
        onnx::NodeProto n;
        
        op.forward(CUDA, in, res, n, ws, handle);
        out.copy_to_host();
        
        float* o = out.data<float>();
        assert(o[0] == 0.0f && o[1] == 0.0f && o[2] == 1.0f && o[3] == 2.0f);
        out.free_device_memory();
    }

    // 2. LEAKY RELU (alpha = 0.1)
    {
        Tensor out(DataType::FLOAT32, {1, 4});
        LeakyReluOp op;
        std::vector<Tensor*> in = {&input}; std::vector<Tensor*> res = {&out};
        onnx::NodeProto n;
        add_float_attr(n, "alpha", 0.1f);
        
        op.forward(CUDA, in, res, n, ws, handle);
        out.copy_to_host();
        
        float* o = out.data<float>();
        assert(check_close(o[0], -0.1f)); // -1 * 0.1
        assert(o[1] == 0.0f);
        assert(o[2] == 1.0f);
        out.free_device_memory();
    }

    // 3. SIGMOID
    {
        Tensor out(DataType::FLOAT32, {1, 4});
        SigmoidOp op;
        std::vector<Tensor*> in = {&input}; std::vector<Tensor*> res = {&out};
        onnx::NodeProto n;
        
        op.forward(CUDA, in, res, n, ws, handle);
        out.copy_to_host();
        
        float* o = out.data<float>();
        // sig(-1) ~= 0.2689, sig(0)=0.5, sig(1)~=0.731
        assert(check_close(o[0], 0.2689f));
        assert(check_close(o[1], 0.5f));
        assert(check_close(o[2], 0.7310f));
        out.free_device_memory();
    }
    
    input.free_device_memory();
    std::cout << "  -> Passed" << std::endl;
}

void test_add_mul() {
    std::cout << "[Test] Add & Mul (Broadcasting)..." << std::endl;
    TestHandle handle;
    std::vector<float> ws;

    // A: [2, 2] -> [[1, 2], [3, 4]]
    Tensor A(DataType::FLOAT32, {2, 2});
    float* a_ptr = A.data<float>();
    a_ptr[0] = 1; a_ptr[1] = 2; a_ptr[2] = 3; a_ptr[3] = 4;

    // B: [1, 2] -> [[10, 20]] (Broadcasts to both rows of A)
    Tensor B(DataType::FLOAT32, {1, 2});
    float* b_ptr = B.data<float>();
    b_ptr[0] = 10; b_ptr[1] = 20;

    A.allocate_device_memory(); A.copy_to_device();
    B.allocate_device_memory(); B.copy_to_device();
    
    // 1. ADD
    {
        Tensor Y(DataType::FLOAT32, {2, 2});
        AddOp op;
        std::vector<Tensor*> in = {&A, &B}; std::vector<Tensor*> out = {&Y};
        onnx::NodeProto n;

        op.forward(CUDA, in, out, n, ws, handle);
        Y.copy_to_host();
        
        float* y = Y.data<float>();
        // Row 1: 1+10, 2+20 -> 11, 22
        // Row 2: 3+10, 4+20 -> 13, 24
        assert(y[0] == 11 && y[1] == 22);
        assert(y[2] == 13 && y[3] == 24);
        Y.free_device_memory();
    }

    // 2. MUL
    {
        Tensor Y(DataType::FLOAT32, {2, 2});
        MulOp op;
        std::vector<Tensor*> in = {&A, &B}; std::vector<Tensor*> out = {&Y};
        onnx::NodeProto n;

        op.forward(CUDA, in, out, n, ws, handle);
        Y.copy_to_host();
        
        float* y = Y.data<float>();
        // Row 1: 1*10, 2*20 -> 10, 40
        // Row 2: 3*10, 4*20 -> 30, 80
        assert(y[0] == 10 && y[1] == 40);
        assert(y[2] == 30 && y[3] == 80);
        Y.free_device_memory();
    }

    A.free_device_memory();
    B.free_device_memory();
    std::cout << "  -> Passed" << std::endl;
}

void test_batchnorm() {
    std::cout << "[Test] BatchNormalization..." << std::endl;
    TestHandle handle;
    std::vector<float> ws;

    // Input: [1, 1, 2, 2] -> [[1, 2], [3, 4]]
    Tensor X(DataType::FLOAT32, {1, 1, 2, 2});
    float* x = X.data<float>();
    x[0]=1; x[1]=2; x[2]=3; x[3]=4;

    // BN Params for 1 channel
    Tensor scale(DataType::FLOAT32, {1}); scale.data<float>()[0] = 2.0f;
    Tensor B(DataType::FLOAT32, {1});     B.data<float>()[0] = 0.5f;
    Tensor mean(DataType::FLOAT32, {1});  mean.data<float>()[0] = 2.5f; // avg of 1,2,3,4
    Tensor var(DataType::FLOAT32, {1});   var.data<float>()[0] = 1.25f; // approx var
    
    // Move all to GPU
    X.allocate_device_memory(); X.copy_to_device();
    scale.allocate_device_memory(); scale.copy_to_device();
    B.allocate_device_memory(); B.copy_to_device();
    mean.allocate_device_memory(); mean.copy_to_device();
    var.allocate_device_memory(); var.copy_to_device();

    Tensor Y(DataType::FLOAT32, {1, 1, 2, 2});
    BatchNorm op;
    std::vector<Tensor*> in = {&X, &scale, &B, &mean, &var};
    std::vector<Tensor*> out = {&Y};
    onnx::NodeProto n;
    add_float_attr(n, "epsilon", 0.0f); // Simplify math

    op.forward(CUDA, in, out, n, ws, handle);
    Y.copy_to_host();

    float* y = Y.data<float>();
    // std = sqrt(1.25) ~= 1.118
    // scale_factor = 2.0 / 1.118 ~= 1.7888
    // bias_factor = 0.5 - (2.5 * 1.7888) = 0.5 - 4.472 = -3.972
    // Expected y[0] (input 1): 1 * 1.788 - 3.972 = -2.18
    
    // Manual check logic: (x - mean)/sqrt(var) * scale + B
    // (1 - 2.5)/1.118 * 2 + 0.5 = -1.5/1.118*2 + 0.5 = -1.341*2 + 0.5 = -2.68 + 0.5 = -2.18
    assert(check_close(y[0], -2.183f, 0.01f));
    assert(check_close(y[3], 3.183f, 0.01f)); // (4-2.5)/1.118*2 + 0.5 = 1.5/1.118*2 + 0.5 = 3.18

    X.free_device_memory(); scale.free_device_memory();
    B.free_device_memory(); mean.free_device_memory(); var.free_device_memory();
    Y.free_device_memory();
    
    std::cout << "  -> Passed" << std::endl;
}

void test_pooling() {
    std::cout << "[Test] MaxPool & AvgPool..." << std::endl;
    TestHandle handle;
    std::vector<float> ws;

    // 4x4 Input
    // 1  2  3  4
    // 5  6  7  8
    // 9  10 11 12
    // 13 14 15 16
    Tensor X(DataType::FLOAT32, {1, 1, 4, 4});
    float* x = X.data<float>();
    for(int i=0; i<16; ++i) x[i] = (float)(i+1);

    X.allocate_device_memory(); X.copy_to_device();

    // 1. MAX POOL (2x2 kernel, stride 2)
    {
        Tensor Y(DataType::FLOAT32, {});
        MaxPoolOp op;
        std::vector<Tensor*> in = {&X}; std::vector<Tensor*> out = {&Y};
        onnx::NodeProto n;
        add_ints_attr(n, "kernel_shape", {2, 2});
        add_ints_attr(n, "strides", {2, 2});
        // Pads = 0

        op.forward(CUDA, in, out, n, ws, handle);
        Y.copy_to_host();

        // Expected Output (2x2):
        // max(1,2,5,6) = 6     max(3,4,7,8) = 8
        // max(9,10,13,14)=14   max(11,12,15,16)=16
        float* y = Y.data<float>();
        assert(y[0] == 6 && y[1] == 8);
        assert(y[2] == 14 && y[3] == 16);
        Y.free_device_memory();
    }

    // 2. AVG POOL (2x2 kernel, stride 2)
    {
        Tensor Y(DataType::FLOAT32, {});
        AvgPoolOp op;
        std::vector<Tensor*> in = {&X}; std::vector<Tensor*> out = {&Y};
        onnx::NodeProto n;
        add_ints_attr(n, "kernel_shape", {2, 2});
        add_ints_attr(n, "strides", {2, 2});

        op.forward(CUDA, in, out, n, ws, handle);
        Y.copy_to_host();

        // Expected Output:
        // avg(1,2,5,6) = 14/4 = 3.5
        float* y = Y.data<float>();
        assert(y[0] == 3.5f);
        Y.free_device_memory();
    }
    
    // 3. GLOBAL AVG POOL
    {
        Tensor Y(DataType::FLOAT32, {});
        GlobalAvgPoolOp op;
        std::vector<Tensor*> in = {&X}; std::vector<Tensor*> out = {&Y};
        onnx::NodeProto n;

        op.forward(CUDA, in, out, n, ws, handle);
        Y.copy_to_host();

        // Avg of 1..16 = (1+16)*16/2 / 16 = 8.5
        assert(Y.shape()[2] == 1 && Y.shape()[3] == 1); // 1x1 output
        assert(Y.data<float>()[0] == 8.5f);
        Y.free_device_memory();
    }

    X.free_device_memory();
    std::cout << "  -> Passed" << std::endl;
}

void test_concat() {
    std::cout << "[Test] Concat..." << std::endl;
    TestHandle handle;
    std::vector<float> ws;

    // T1: 2x2 [[1, 2], [3, 4]]
    Tensor T1(DataType::FLOAT32, {1, 1, 2, 2});
    float* t1 = T1.data<float>();
    t1[0]=1; t1[1]=2; t1[2]=3; t1[3]=4;

    // T2: 2x2 [[5, 6], [7, 8]]
    Tensor T2(DataType::FLOAT32, {1, 1, 2, 2});
    float* t2 = T2.data<float>();
    t2[0]=5; t2[1]=6; t2[2]=7; t2[3]=8;

    T1.allocate_device_memory(); T1.copy_to_device();
    T2.allocate_device_memory(); T2.copy_to_device();

    Tensor Y(DataType::FLOAT32, {});
    ConcatOp op;
    std::vector<Tensor*> in = {&T1, &T2}; std::vector<Tensor*> out = {&Y};
    onnx::NodeProto n;
    add_int_attr(n, "axis", 1); // Concat on channels

    op.forward(CUDA, in, out, n, ws, handle);
    Y.copy_to_host();

    // Output should be [1, 2, 2, 2]
    // Channel 0: [[1,2],[3,4]]
    // Channel 1: [[5,6],[7,8]]
    float* y = Y.data<float>();
    assert(y[0] == 1 && y[3] == 4); // Chan 0
    assert(y[4] == 5 && y[7] == 8); // Chan 1
    
    T1.free_device_memory();
    T2.free_device_memory();
    Y.free_device_memory();
    std::cout << "  -> Passed" << std::endl;
}

void test_upsample() {
    std::cout << "[Test] Upsample (Resample)..." << std::endl;
    TestHandle handle;
    std::vector<float> ws;

    // 2x2 Input: [[1, 2], [3, 4]]
    Tensor X(DataType::FLOAT32, {1, 1, 2, 2});
    float* x = X.data<float>();
    x[0]=1; x[1]=2; x[2]=3; x[3]=4;
    
    // Scales: 2.0 (Nearest Neighbor)
    Tensor S(DataType::FLOAT32, {4});
    float* s = S.data<float>();
    s[0]=1; s[1]=1; s[2]=2.0f; s[3]=2.0f; 

    X.allocate_device_memory(); X.copy_to_device();
    // Note: Scales can be on CPU or GPU depending on implementation, 
    // but our forward_gpu handles CPU-side read or GPU-side copy. 
    // we'll keep it on CPU for simplicity as our forward_gpu handles it.

    Tensor Y(DataType::FLOAT32, {});
    UpsampleOp op;
    std::vector<Tensor*> in = {&X, nullptr, &S}; // scales are idx 2
    std::vector<Tensor*> out = {&Y};
    onnx::NodeProto n;

    op.forward(CUDA, in, out, n, ws, handle);
    Y.copy_to_host();

    // Output 4x4. Top Left 2x2 should all be 1.
    // 1 1 2 2
    // 1 1 2 2
    // 3 3 4 4 
    // 3 3 4 4
    float* y = Y.data<float>();
    assert(y[0] == 1 && y[1] == 1); // (0,0) -> 1, (0,1) -> 1
    assert(y[2] == 2);              // (0,2) -> 2
    assert(y[8] == 3);              // (2,0) -> 3
    
    X.free_device_memory();
    Y.free_device_memory();
    std::cout << "  -> Passed" << std::endl;
}

void test_softmax() {
    std::cout << "[Test] Softmax..." << std::endl;
    TestHandle handle;
    std::vector<float> ws;

    // 1x4: [0, 1, 2, 3]
    Tensor X(DataType::FLOAT32, {1, 4});
    float* x = X.data<float>();
    x[0]=0; x[1]=1; x[2]=2; x[3]=3;

    X.allocate_device_memory(); X.copy_to_device();
    
    Tensor Y(DataType::FLOAT32, {});
    SoftmaxOp op;
    std::vector<Tensor*> in = {&X}; std::vector<Tensor*> out = {&Y};
    onnx::NodeProto n;
    add_int_attr(n, "axis", 1);

    op.forward(CUDA, in, out, n, ws, handle);
    Y.copy_to_host();

    float* y = Y.data<float>();
    // exp(0)+exp(1)+exp(2)+exp(3) ~= 1 + 2.71 + 7.38 + 20.08 = 31.17
    // y[3] = 20.08 / 31.17 ~= 0.64
    // y[0] = 1 / 31.17 ~= 0.032
    assert(check_close(y[0], 0.032f, 0.01f));
    assert(check_close(y[3], 0.644f, 0.01f));

    X.free_device_memory();
    Y.free_device_memory();
    std::cout << "  -> Passed" << std::endl;
}

void test_conv_integration() {
    std::cout << "[Test] Convolution (Integration Check)..." << std::endl;
    TestHandle handle;
    std::vector<float> ws;

    // 1x1x3x3 Input: 1..9
    Tensor X(DataType::FLOAT32, {1, 1, 3, 3});
    float* d = X.data<float>();
    for(int i=0; i<9; ++i) d[i] = (float)(i+1);

    // 1x1x2x2 Identity Kernel
    Tensor W(DataType::FLOAT32, {1, 1, 2, 2});
    float* w = W.data<float>();
    w[0]=1; w[1]=0; w[2]=0; w[3]=1;

    X.allocate_device_memory(); X.copy_to_device();
    W.allocate_device_memory(); W.copy_to_device();

    Tensor Y(DataType::FLOAT32, {});
    ConvOp op;
    std::vector<Tensor*> in = {&X, &W}; std::vector<Tensor*> out = {&Y};
    onnx::NodeProto n;
    // Defaults: stride=1, pad=0

    op.forward(CUDA, in, out, n, ws, handle);
    Y.copy_to_host();

    // Output 2x2. 
    // (0,0) = 1*1 + 5*1 = 6
    // (0,1) = 2*1 + 6*1 = 8
    // (1,0) = 4*1 + 8*1 = 12
    // (1,1) = 5*1 + 9*1 = 14
    float* y = Y.data<float>();
    assert(y[0] == 6.0f);
    assert(y[3] == 14.0f);

    X.free_device_memory(); W.free_device_memory(); Y.free_device_memory();
    std::cout << "  -> Passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " RUNNING CUDA OPERATOR TEST SUITE" << std::endl;
    std::cout << "========================================" << std::endl;

    test_elementwise_activations();
    test_add_mul();
    test_batchnorm();
    test_pooling();
    test_concat();
    test_upsample();
    test_softmax();
    test_conv_integration();

    std::cout << "========================================" << std::endl;
    std::cout << " ALL CUDA TESTS PASSED" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
