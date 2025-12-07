#include <iostream>
#include <vector>
#include <cassert>
#include <cmath> // for std::abs
#include "tensor.h" // Assumes tensor.h is in your include/ folder

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

int main() {
    std::cout << "Running Tensor Unit Tests..." << std::endl;
    test_initialization();
    test_strides();
    test_data_access();
    test_reshape();
    
    std::cout << "------------------" << std::endl;
    std::cout << "ALL TESTS PASSED!" << std::endl;
    return 0;
}
