#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

#include "model_loader.h"
#include "engine.h"
#include "tensor.h"

// Helper to load simple float text file
bool load_input_file(const std::string& filepath, Tensor& input_tensor) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;
    float* data = input_tensor.data<float>();
    float val;
    int idx = 0;
    while (file >> val && idx < input_tensor.size()) {
        data[idx++] = val;
    }
    return true;
}

int main(int argc, char** argv) {
    std::cout << "--- Upsample (Resize) Scales Input Test ---" << std::endl;

    // Default arguments
    std::string model_path = (argc > 1) ? argv[1] : "upsample_scales_test.onnx";
    std::string input_path = (argc > 2) ? argv[2] : "input_upsample.txt";

    // 1. Load Model
    ModelLoader loader;
    if (!loader.load(model_path)) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }

    // 2. Init Engine (With your new OpFactory)
    InferenceEngine engine;
    try {
        engine.load_model(loader);
    } catch (const std::exception& e) {
        std::cerr << "Engine Init Failed: " << e.what() << std::endl;
        return 1;
    }

    // 3. Prepare Input 
    // The Python test script generates a 2x2 input:
    // [[[ 1, 2 ],
    //   [ 3, 4 ]]]
    std::vector<int64_t> input_shape = {1, 1, 2, 2};
    Tensor input(DataType::FLOAT32, input_shape, "X"); 

    if (!load_input_file(input_path, input)) {
        std::cerr << "Error: Could not load input file " << input_path << std::endl;
        // Fallback check
        std::string alt_path = "../" + input_path;
        if (load_input_file(alt_path, input)) {
            std::cout << "Found input at: " << alt_path << std::endl;
        } else {
             return 1;
        }
    }

    // 4. Run Inference
    try {
        engine.run(input);
    } catch (const std::exception& e) {
        std::cerr << "Inference Failed: " << e.what() << std::endl;
        return 1;
    }

    // 5. Verify Output
    Tensor& output = engine.get_output();
    float* data = output.data<float>();
    
    std::cout << "\n=== C++ Output Results ===" << std::endl;
    std::cout << "Output Shape: [";
    for(auto d : output.shape()) std::cout << d << ",";
    std::cout << "]" << std::endl;
    
    // We expect a 4x4 output. Let's print it nicely as a matrix.
    int out_h = output.shape()[2];
    int out_w = output.shape()[3];
    
    std::cout << "Values (Compare with Python):" << std::endl;
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            std::cout << std::fixed << std::setprecision(1) << data[y * out_w + x] << "  ";
        }
        std::cout << std::endl;
    }

    return 0;
}
