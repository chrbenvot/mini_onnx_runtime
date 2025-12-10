#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include "model_loader.h"
#include "engine.h"
#include "tensor.h"

// --- YOLO CONFIGURATION ---
const int NUM_ANCHORS = 5;
const int NUM_CLASSES = 20;
const int BLOCK_SIZE = 5 + NUM_CLASSES; // 5 coords + 20 classes = 25 channels per anchor

// Anchors for Tiny YOLO v2
const float ANCHORS[10] = {1.08, 1.19,  3.42, 4.41,  6.63, 11.38,  9.42, 5.11,  16.62, 10.52};

const std::string CLASS_NAMES[] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

// Helper: Sigmoid
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Helper: Softmax
void softmax_array(float* start, int n) {
    float max_val = -1e9;
    for(int i=0; i<n; ++i) if(start[i] > max_val) max_val = start[i];
    float sum = 0.0f;
    for(int i=0; i<n; ++i) {
        start[i] = std::exp(start[i] - max_val);
        sum += start[i];
    }
    for(int i=0; i<n; ++i) start[i] /= sum;
}

// Helper: Load File
bool load_input_file(const std::string& filepath, Tensor& input_tensor) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;
    float* data = input_tensor.data<float>();
    float val;
    int idx = 0;
    while (file >> val && idx < input_tensor.size()) data[idx++] = val;
    return true;
}

int main(int argc, char** argv) {
    std::cout << "--- Tiny YOLO v2 Inference ---" << std::endl;
    
    // Updated defaults to look in current directory, not parent
    std::string model_path = (argc > 1) ? argv[1] : "tiny_yolo.onnx";
    std::string input_path = (argc > 2) ? argv[2] : "input_yolo.txt";

    ModelLoader loader;
    if (!loader.load(model_path)) {
        std::cerr << "Error: Failed to load model: " << model_path << std::endl;
        return 1;
    }

    InferenceEngine engine;
    try { engine.load_model(loader); } 
    catch (const std::exception& e) { return 1; }

    // 1. Prepare Input (416x416)
    std::vector<int64_t> input_shape = {1, 3, 416, 416};
    Tensor input(DataType::FLOAT32, input_shape, "image"); 
    
    if (!load_input_file(input_path, input)) {
        std::cerr << "Error: Could not load input file: " << input_path << std::endl;
        // Fallback: Check parent directory if current fails (Legacy support)
        std::string alt_path = "../" + input_path;
        if (load_input_file(alt_path, input)) {
            std::cout << "Found input in parent directory: " << alt_path << std::endl;
        } else {
             return 1;
        }
    }

    // 2. Run Engine
    try {
        std::cout << "Starting inference..." << std::endl;
        engine.run(input);
    } catch (const std::exception& e) {
        std::cerr << "Inference Failed: " << e.what() << std::endl;
        return 1;
    }

    // 3. Decode Output
    Tensor& output = engine.get_output();
    float* data = output.data<float>();
    
    // DYNAMICALLY Get Grid Size to prevent Segfaults
    int grid_h = output.shape()[2]; // Should be 13
    int grid_w = output.shape()[3]; // Should be 13
    int channels = output.shape()[1]; // Should be 125

    std::cout << "Output Shape: [" << output.shape()[0] << ", " 
              << channels << ", " << grid_h << ", " << grid_w << "]" << std::endl;

    if (channels != 125) {
        std::cerr << "Error: Unexpected output channel count. Expected 125, got " << channels << std::endl;
        return 1;
    }

    std::cout << "\n=== Detections (Threshold > 0.3) ===" << std::endl;
    int detections = 0;

    for (int cy = 0; cy < grid_h; ++cy) {
        for (int cx = 0; cx < grid_w; ++cx) {
            for (int b = 0; b < NUM_ANCHORS; ++b) {
                
                int channel_start = b * BLOCK_SIZE;
                
                // Index Calculation: Channel * (H*W) + y*W + x
                int obj_channel = channel_start + 4; 
                int obj_idx = obj_channel * (grid_h * grid_w) + cy * grid_w + cx;
                
                float confidence = sigmoid(data[obj_idx]);

                if (confidence > 0.3) {
                    // Extract Coordinates
                    int tx_idx = (channel_start + 0) * (grid_h * grid_w) + cy * grid_w + cx;
                    int ty_idx = (channel_start + 1) * (grid_h * grid_w) + cy * grid_w + cx;
                    int tw_idx = (channel_start + 2) * (grid_h * grid_w) + cy * grid_w + cx;
                    int th_idx = (channel_start + 3) * (grid_h * grid_w) + cy * grid_w + cx;

                    float x = (cx + sigmoid(data[tx_idx])) / grid_w;
                    float y = (cy + sigmoid(data[ty_idx])) / grid_h;
                    float w = (std::exp(data[tw_idx]) * ANCHORS[2*b]) / grid_w;
                    float h = (std::exp(data[th_idx]) * ANCHORS[2*b+1]) / grid_h;

                    // Find Best Class
                    float max_class_score = 0.0f;
                    int best_class = -1;
                    
                    std::vector<float> class_probs(NUM_CLASSES);
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        int c_idx = (channel_start + 5 + c) * (grid_h * grid_w) + cy * grid_w + cx;
                        class_probs[c] = data[c_idx];
                    }
                    softmax_array(class_probs.data(), NUM_CLASSES);

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        if (class_probs[c] > max_class_score) {
                            max_class_score = class_probs[c];
                            best_class = c;
                        }
                    }
                    
                    float final_score = confidence * max_class_score;

                    if (final_score > 0.25) {
                        detections++;
                        std::cout << "Detected: " << CLASS_NAMES[best_class] 
                                  << " | Conf: " << (final_score*100) << "%"
                                  << " | Center: [" << x << ", " << y << "]" << std::endl;
                    }
                }
            }
        }
    }
    
    if (detections == 0) std::cout << "No objects detected." << std::endl;

    return 0;
}
