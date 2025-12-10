#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h> // Essential CUDA headers
#include "../include/tensor.h"


void test_cuda_memory() {
    std::cout << "--- Starting CUDA Memory Tests ---" << std::endl;

    // 1. Setup Host Tensor
    std::vector<int64_t> shape = {2, 3}; // 6 elements total
    Tensor t(DataType::FLOAT32, shape, "TestTensor");

    // Initialize data on the Host (CPU)
    float* host_data = t.data<float>();
    for (int i = 0; i < 6; ++i) {
        host_data[i] = (float)(i * 10.0 + 1.5); // Unique data: 1.5, 11.5, 21.5, ...
    }
    std::cout << "  Host data initialized." << std::endl;

    // 2. Allocate Device Memory (Calls cudaMalloc inside Tensor)
    std::cout << "  Attempting device allocation..." << std::endl;
    t.allocate_device_memory();
    assert(t.device_data() != nullptr);
    assert(t.is_on_device() == true);
    std::cout << "  Device memory allocated successfully." << std::endl;

    // 3. Copy Host to Device (Calls cudaMemcpy HtoD)
    std::cout << "  Copying H -> D..." << std::endl;
    t.copy_to_device();
    std::cout << "  Copy complete." << std::endl;
    
    // --- Verification: We must modify the Host data to verify the read-back ---
    for (int i = 0; i < 6; ++i) {
        host_data[i] = 0.0f; // Wipe the CPU buffer
    }
    std::cout << "  Host data wiped to zeros." << std::endl;

    // 4. Copy Device to Host (Calls cudaMemcpy DtoH)
    std::cout << "  Copying D -> H (read-back)..." << std::endl;
    t.copy_to_host();
    std::cout << "  Read-back complete." << std::endl;

    // 5. Verify Data Integrity
    bool integrity_check = true;
    for (int i = 0; i < 6; ++i) {
        float expected = (float)(i * 10.0 + 1.5);
        if (std::abs(host_data[i] - expected) > 0.001f) {
            std::cerr << "  Verification FAILED at index " << i 
                      << ". Expected: " << expected 
                      << ", Got: " << host_data[i] << std::endl;
            integrity_check = false;
        }
    }
    assert(integrity_check);
    
    // 6. Free Device Memory
    std::cout << "  Freeing device memory..." << std::endl;
    t.free_device_memory();
    assert(t.device_data() == nullptr);
    assert(t.is_on_device() == false);

    std::cout << "--- CUDA Memory Tests PASSED ---" << std::endl;
}

int main() {
    test_cuda_memory();
    return 0;
}
