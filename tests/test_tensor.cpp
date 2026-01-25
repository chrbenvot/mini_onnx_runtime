#include <iostream>
#include <vector>
#include <cassert>
#include <cmath> // for std::abs
#include "tensor.h"
#include "operators/relu.h"
#include "operators/conv.h"
#include "operators/concat.h"
#include "operators/upsample.h"

struct TestHandle
{
    cublasHandle_t h;
    TestHandle() { cublasCreate(&h); }
    ~TestHandle() { cublasDestroy(h); }
    operator cublasHandle_t &() { return h; }
}; // struct is needed for RAII ...

void test_initialization()
{
    std::cout << "Testing Initialization..." << std::endl;
    std::vector<int64_t> shape = {1, 3, 224, 224};
    Tensor t(DataType::FLOAT32, shape, "InputImage");

    assert(t.shape() == shape);
    assert(t.size() == 1 * 3 * 224 * 224);
    assert(t.element_size() == 4); // Float32 = 4 bytes
    assert(t.name() == "InputImage");
    std::cout << "  -> Passed" << std::endl;
}

void test_strides()
{
    std::cout << "Testing Strides (Row-Major Logic)..." << std::endl;
    Tensor t(DataType::FLOAT32, {2, 3, 4});

    const auto &strides = t.strides();
    assert(strides[0] == 12);
    assert(strides[1] == 4);
    assert(strides[2] == 1);
    std::cout << "  -> Passed" << std::endl;
}

void test_data_access()
{
    std::cout << "Testing Data Write/Read..." << std::endl;
    Tensor t(DataType::FLOAT32, {2, 2});

    t.at<float>({0, 0}) = 10.0f;
    t.at<float>({0, 1}) = 20.0f;
    t.at<float>({1, 0}) = 30.0f;
    t.at<float>({1, 1}) = 40.0f;

    assert(std::abs(t.at<float>({0, 0}) - 10.0f) < 0.001);
    assert(std::abs(t.at<float>({1, 1}) - 40.0f) < 0.001);

    float *raw = t.data<float>();
    assert(raw[0] == 10.0f);
    assert(raw[3] == 40.0f);
    std::cout << "  -> Passed" << std::endl;
}

void test_reshape()
{
    std::cout << "Testing Reshape..." << std::endl;
    Tensor t(DataType::FLOAT32, {10});
    assert(t.strides()[0] == 1);

    t.reshape({2, 5});
    assert(t.shape()[0] == 2 && t.shape()[1] == 5);
    assert(t.strides()[0] == 5 && t.strides()[1] == 1);
    std::cout << "  -> Passed" << std::endl;
}

void test_relu()
{
    std::cout << "Testing ReLU Operator..." << std::endl;

    Tensor input(DataType::FLOAT32, {4});
    float *in_data = input.data<float>();
    in_data[0] = -10.0f;
    in_data[1] = 5.0f;
    in_data[2] = -2.0f;
    in_data[3] = 0.0f;

    Tensor output(DataType::FLOAT32, {4});

    std::vector<float> workspace;
    TestHandle handle;
    input.allocate_device_memory();
    input.copy_to_device(); // Host -> Device

    // Prepare output on GPU
    output.allocate_device_memory();
    ReluOp op;
    std::vector<Tensor *> inputs = {&input};
    std::vector<Tensor *> outputs = {&output};
    onnx::NodeProto node;

    op.forward(CUDA, inputs, outputs, node, workspace, handle); // Added workspace
    output.copy_to_host();

    float *out_data = output.data<float>();
    assert(out_data[0] == 0.0f);
    assert(out_data[1] == 5.0f);
    assert(out_data[2] == 0.0f);
    assert(out_data[3] == 0.0f);
    input.free_device_memory();
    output.free_device_memory();

    std::cout << "  -> Passed" << std::endl;
}

/*void test_relu_int8()
{
    std::cout << "Testing ReLU (INT8 Quantized)..." << std::endl;

    Tensor input(DataType::INT8, {4});
    int8_t *in_data = input.data<int8_t>();
    in_data[0] = -10; in_data[1] = 5; in_data[2] = -100; in_data[3] = 20;

    Tensor output(DataType::INT8, {4});
    std::vector<float> workspace;

    ReluOp op;
    std::vector<Tensor *> inputs = {&input};
    std::vector<Tensor *> outputs = {&output};
    onnx::NodeProto node;

    op.forward(CPU,inputs, outputs, node, workspace); // Added workspace

    int8_t *out_data = output.data<int8_t>();
    assert(out_data[0] == 0);
    assert(out_data[1] == 5);
    assert(out_data[2] == 0);
    assert(out_data[3] == 20);

    std::cout << "  -> Passed" << std::endl;
}
*/
void test_conv_simple()
{
    std::cout << "Testing Convolution (Manual Math Check)..." << std::endl;

    // 1  2  3
    // 4  5  6
    // 7  8  9
    Tensor input(DataType::FLOAT32, {1, 1, 3, 3});
    float *in_ptr = input.data<float>();
    for (int i = 0; i < 9; ++i)
        in_ptr[i] = (float)(i + 1);

    // Identity Diagonal Kernel
    Tensor weight(DataType::FLOAT32, {1, 1, 2, 2});
    float *w_ptr = weight.data<float>();
    w_ptr[0] = 1.0f;
    w_ptr[1] = 0.0f;
    w_ptr[2] = 0.0f;
    w_ptr[3] = 1.0f;

    onnx::NodeProto node;
    // pads = [0,0,0,0]
    auto *pads = node.add_attribute();
    pads->set_name("pads");
    pads->add_ints(0);
    pads->add_ints(0);
    pads->add_ints(0);
    pads->add_ints(0);

    // strides = [1,1]
    auto *strides = node.add_attribute();
    strides->set_name("strides");
    strides->add_ints(1);
    strides->add_ints(1);

    input.allocate_device_memory();
    input.copy_to_device(); // Host -> Device

    // Prepare Output on GPU
    weight.allocate_device_memory();
    weight.copy_to_device();
    ConvOp op;
    Tensor output(DataType::FLOAT32, {1, 1, 2, 2});
    std::vector<float> workspace; // NEW
    TestHandle handle;
    output.allocate_device_memory();

    std::vector<Tensor *> inputs = {&input, &weight};
    std::vector<Tensor *> outputs = {&output};

    op.forward(CUDA, inputs, outputs, node, workspace, handle); // Added workspace
    output.copy_to_host();

    float *out_ptr = output.data<float>();
    assert(output.shape()[2] == 2);
    assert(output.shape()[3] == 2);
    assert(out_ptr[0] == 6.0f);
    assert(out_ptr[1] == 8.0f);
    assert(out_ptr[2] == 12.0f);
    assert(out_ptr[3] == 14.0f);
    input.free_device_memory();
    output.free_device_memory();
    weight.free_device_memory();

    std::cout << "  -> Passed" << std::endl;
}

void test_concat_upsample()
{
    std::cout << "Testing Upsample & Concat (Integration)..." << std::endl;

    Tensor input(DataType::FLOAT32, {1, 1, 2, 2});
    float *in_ptr = input.data<float>();
    in_ptr[0] = 1.0f;
    in_ptr[1] = 2.0f;
    in_ptr[2] = 3.0f;
    in_ptr[3] = 4.0f;
    Tensor roi(DataType::FLOAT32, {0}); // Empty ROI

    // Scales: [1, 1, 2, 2] -> Scale H and W by 2.0x
    Tensor scales(DataType::FLOAT32, {4});
    float *s_data = scales.data<float>();
    s_data[0] = 1.0f;
    s_data[1] = 1.0f;
    s_data[2] = 2.0f;
    s_data[3] = 2.0f;

    UpsampleOp resize_op;
    Tensor upsampled_out(DataType::FLOAT32, {});
    std::vector<float> workspace; 
    TestHandle handle;

    std::vector<Tensor *> up_inputs = {&input, &roi, &scales};
    std::vector<Tensor *> up_outputs = {&upsampled_out};
    onnx::NodeProto resize_node;

    resize_op.forward(CPU, up_inputs, up_outputs, resize_node, workspace, handle); 

    assert(upsampled_out.shape()[2] == 4);
    assert(upsampled_out.shape()[3] == 4);

    ConcatOp concat_op;
    Tensor concat_out(DataType::FLOAT32, {});

    std::vector<Tensor *> cat_inputs = {&upsampled_out, &upsampled_out};
    std::vector<Tensor *> cat_outputs = {&concat_out};

    onnx::NodeProto concat_node;
    auto *axis_attr = concat_node.add_attribute();
    axis_attr->set_name("axis");
    axis_attr->set_i(1);

    concat_op.forward(CPU, cat_inputs, cat_outputs, concat_node, workspace, handle);
    assert(concat_out.shape()[0] == 1);
    assert(concat_out.shape()[1] == 2);
    assert(concat_out.shape()[2] == 4);
    assert(concat_out.shape()[3] == 4);

    std::cout << "  -> Passed" << std::endl;
}

int main()
{
    std::cout << "Running Tensor Unit Tests..." << std::endl;
    test_initialization();
    test_strides();
    test_data_access();
    test_reshape();
    test_relu();
    // test_relu_int8();
    test_conv_simple();
    test_concat_upsample();
    std::cout << "------------------" << std::endl;
    std::cout << "ALL TESTS PASSED!" << std::endl;
    return 0;
}
