#pragma once
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "tensor.h"

inline void debug_gpu_tensor(const std::string& label, Tensor* t) {
    if (!t->is_on_device()) {
        std::cout << "[DEBUG " << label << "] Not on device!" << std::endl;
        return;
    }
    
    // Copy 10 values to CPU to check
    std::vector<float> dump(10);
    // Safety check size
    size_t count = std::min((size_t)10, (size_t)t->size());
    cudaMemcpy(dump.data(), t->device_data(), count * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0;
    // Also grab full sum for sanity (expensive but useful for debugging)
    // For now, just check head
    std::cout << "[DEBUG " << label << "] Head: ";
    for(size_t i=0; i<count; ++i) {
        std::cout << dump[i] << ", ";
        sum += std::abs(dump[i]);
    }
    std::cout << (sum == 0 ? " (ALL ZEROS?)" : "") << std::endl;
}
