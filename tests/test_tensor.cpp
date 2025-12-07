#include <iostream>
#include <vector>
#include <cassert>
#include <cmath> // for std::abs
#include "tensor.h" 
#include "operators/relu.h"

void test_initialization() {
    std::cout << "Testing Initialization..." << std::endl;
    std::vector<int64_t> shape = {1, 3, 224, 224};
    Tensor t(DataType::FLOAT32, shape, "InputImage");
    
    assert(t.shape() == shape);
    assert(t.size() == 1 * 3 * 224 * 224);
    assert(t.element_size() == 4); // Float32 = 4 bytes
    assert(t.name() == "InputImage");
    std::cout << "  -> Passed" << std::endl;
}

void test_strides() {
    std::cout << "Testing Strides (Row-Major Logic)..." << std::endl;
    // Shape [2, 3, 4] 
    // Stride[2] should be 1
    // Stride[1] should be 4 (1 * 4)
    // Stride[0] should be 12 (4 * 3)
    Tensor t(DataType::FLOAT32, {2, 3, 4});
    
    const auto& strides = t.strides();
    assert(strides[0] == 12);
    assert(strides[1] == 4);
    assert(strides[2] == 1);
    std::cout << "  -> Passed" << std::endl;
}

void test_data_access() {
    std::cout << "Testing Data Write/Read..." << std::endl;
    Tensor t(DataType::FLOAT32, {2, 2}); // 2x2 matrix
    
    // Write using .at<float>()
    // Index mapping: 
    // (0,0) -> 0
    // (0,1) -> 1
    // (1,0) -> 2
    // (1,1) -> 3
    t.at<float>({0, 0}) = 10.0f;
    t.at<float>({0, 1}) = 20.0f;
    t.at<float>({1, 0}) = 30.0f;
    t.at<float>({1, 1}) = 40.0f;

    // Verify values exist
    assert(std::abs(t.at<float>({0, 0}) - 10.0f) < 0.001);
    assert(std::abs(t.at<float>({1, 1}) - 40.0f) < 0.001);
    
    // Verify Raw Pointer Access matches logical access
    float* raw = t.data<float>();
    assert(raw[0] == 10.0f);
    assert(raw[3] == 40.0f); 
    std::cout << "  -> Passed" << std::endl;
}

void test_reshape() {
    std::cout << "Testing Reshape..." << std::endl;
    Tensor t(DataType::FLOAT32, {10}); // 1D vector
    assert(t.strides()[0] == 1);
    
    t.reshape({2, 5}); // Turn into 2x5 matrix
    assert(t.shape()[0] == 2 && t.shape()[1] == 5);
    assert(t.strides()[0] == 5 && t.strides()[1] == 1);
    std::cout << "  -> Passed" << std::endl;
}

void test_relu() {
    std::cout << "Testing ReLU Operator..." << std::endl;
    
    // 1. Create Input: [-10, 5, -2, 0]
    Tensor input(DataType::FLOAT32, {4});
    float* in_data = input.data<float>();
    in_data[0] = -10.0f;
    in_data[1] = 5.0f;
    in_data[2] = -2.0f;
    in_data[3] = 0.0f;

    // 2. Create Output Tensor (Empty for now)
    Tensor output(DataType::FLOAT32, {4});

    // 3. Run Operator
    ReluOp op;
    std::vector<Tensor*> inputs = {&input};
    std::vector<Tensor*> outputs = {&output};
    op.forward(inputs, outputs);

    // 4. Verify Results: [0, 5, 0, 0]
    float* out_data = output.data<float>();
    assert(out_data[0] == 0.0f);
    assert(out_data[1] == 5.0f);
    assert(out_data[2] == 0.0f);
    assert(out_data[3] == 0.0f);

    std::cout << "  -> Passed" << std::endl;
}
void test_relu_int8() {
    std::cout << "Testing ReLU (INT8 Quantized)..." << std::endl;

    // 1. Create INT8 Input: [-10, 5, -100, 20]
    // Note: int8_t range is -128 to 127
    Tensor input(DataType::INT8, {4});
    int8_t* in_data = input.data<int8_t>();
    in_data[0] = -10;
    in_data[1] = 5;
    in_data[2] = -100;
    in_data[3] = 20;

    Tensor output(DataType::INT8, {4});
    
    // 2. Run Operator
    ReluOp op;
    std::vector<Tensor*> inputs = {&input};
    std::vector<Tensor*> outputs = {&output};
    op.forward(inputs, outputs);

    // 3. Verify
    int8_t* out_data = output.data<int8_t>();
    assert(out_data[0] == 0);   // -10 -> 0
    assert(out_data[1] == 5);   // 5 -> 5
    assert(out_data[2] == 0);   // -100 -> 0
    assert(out_data[3] == 20);  // 20 -> 20

    std::cout << "  -> Passed" << std::endl;
}



int main() {
    std::cout << "Running Tensor Unit Tests..." << std::endl;
    test_initialization();
    test_strides();
    test_data_access();
    test_reshape();
    test_relu();
    test_relu_int8();
    std::cout << "------------------" << std::endl;
    std::cout << "ALL TESTS PASSED!" << std::endl;
    return 0;
}
