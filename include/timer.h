#pragma once
#include <iostream>
#include <chrono>
#include <string>

class ScopedTimer {
public:
    ScopedTimer(const std::string& name) 
        : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        // Print in milliseconds (ms) for readability
        std::cout << "[TIMER] " << name_ << ": " << (duration / 1000.0f) << " ms" << std::endl;
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
