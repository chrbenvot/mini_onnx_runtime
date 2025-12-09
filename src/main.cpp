#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "model_loader.h"
#include "engine.h"
#include "tensor.h"
#include "operators/softmax.h" // Ensure you have this!

// Helper to sort indices based on values (for Top-K)
std::vector<int> argsort(const float* data, int size) {
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [data](int i1, int i2) {
        return data[i1] > data[i2]; // Sort Descending (Max first)
    });
    return indices;
}

// File Loader Helper
bool load_input_file(const std::string& filepath, Tensor& input_tensor) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << std::endl;
        return false;
    }
    float* data = input_tensor.data<float>();
    float val;
    int idx = 0;
    while (file >> val && idx < input_tensor.size()) {
        data[idx++] = val;
    }
    return true;
}

int main(int argc, char** argv) {
    std::cout << "--- C++ SqueezeNet Inference ---" << std::endl;

    std::string model_path = (argc > 1) ? argv[1] : "../squeezenet.onnx";
    std::string input_path = (argc > 2) ? argv[2] : "../input_squeezenet.txt";

    // 1. Load Model
    ModelLoader loader;
    if (!loader.load(model_path)) return 1;

    // 2. Init Engine
    InferenceEngine engine;
    try {
        engine.load_model(loader);
    } catch (const std::exception& e) {
        std::cerr << "Engine Init Failed: " << e.what() << std::endl;
        return 1;
    }

    // 3. Prepare Input Tensor (ImageNet Standard: 1x3x224x224)
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    Tensor input(DataType::FLOAT32, input_shape, "data"); // 'data' is usually the input name for SqueezeNet

    if (!load_input_file(input_path, input)) {
        std::cerr << "Failed to load input image data." << std::endl;
        return 1;
    }

    // 4. Run Inference
    try {
        std::cout << "Running SqueezeNet (This might take a second)..." << std::endl;
        engine.run(input);
    } catch (const std::exception& e) {
        std::cerr << "Inference Failed: " << e.what() << std::endl;
        return 1;
    }

    // 5. Process Output
    Tensor& output = engine.get_output();
    
    // Apply Softmax manually (SqueezeNet output is usually logits)
    Tensor probabilities(DataType::FLOAT32, output.shape());
    SoftmaxOp softmax_op;
    std::vector<Tensor*> sm_inputs = {&output};
    std::vector<Tensor*> sm_outputs = {&probabilities};
    onnx::NodeProto dummy_node; 
    
    // Check if output is 4D [1, 1000, 1, 1] or 2D [1, 1000]
    // SqueezeNet often leaves it as [1, 1000, 1, 1] after GlobalPool
    // Softmax handles this fine if we treat it as flat, or provide correct axis.
    softmax_op.forward(sm_inputs, sm_outputs, dummy_node);

    // 6. Print Top 5 Predictions
    float* prob_data = probabilities.data<float>();
    auto sorted_indices = argsort(prob_data, probabilities.size());

    std::cout << "\n=== Top 5 Predictions ===" << std::endl;
    for (int i = 0; i < 5; ++i) {
        int idx = sorted_indices[i];
        float score = prob_data[idx];
        
        // Print Index and Score (You can lookup class names online "ImageNet labels")
        std::cout << "#" << (i+1) << ": Class " << idx 
                  << " | Confidence: " << std::fixed << std::setprecision(2) << (score * 100.0f) << "%" << std::endl;
    }

    return 0;
}
