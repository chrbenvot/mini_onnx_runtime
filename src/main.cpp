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

//  YOLO CONFIGURATION 
const int NUM_ANCHORS = 5;
const int NUM_CLASSES = 20;
const int BLOCK_SIZE = 5 + NUM_CLASSES; // 5 coords + 20 classes = 25 channels per anchor

// Anchors for Tiny YOLO v2
const float ANCHORS[10] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};

const std::string CLASS_NAMES[] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

// Helper: Sigmoid
float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

// Helper: Softmax (not used for current main but used in previous ones so may as well keep it)
void softmax_array(float *start, int n)
{
    float max_val = -1e9;
    for (int i = 0; i < n; ++i)
        if (start[i] > max_val)
            max_val = start[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        start[i] = std::exp(start[i] - max_val);
        sum += start[i];
    }
    for (int i = 0; i < n; ++i)
        start[i] /= sum;
}

// Helper: Load File
bool load_input_file(const std::string &filepath, Tensor &input_tensor)
{
    std::ifstream file(filepath);
    if (!file.is_open())
        return false;
    float *data = input_tensor.data<float>();
    float val;
    int idx = 0;
    while (file >> val && idx < input_tensor.size())
        data[idx++] = val;
    return true;
}

int main(int argc, char **argv)
{
    std::cout << "--- Tiny YOLO v2 Inference ---" << std::endl;

    // Updated defaults to look in current directory, not parent
    std::string model_path = (argc > 1) ? argv[1] : "tiny_yolo.onnx";
    std::string input_path = (argc > 2) ? argv[2] : "input_yolo.txt";

    ModelLoader loader;
    if (!loader.load(model_path))
    {
        std::cerr << "Error: Failed to load model: " << model_path << std::endl;
        return 1;
    }

    InferenceEngine engine;
    try
    {
        engine.load_model(loader);
    }
    catch (const std::exception &e)
    {
        return 1;
    }

    // 1. Prepare Input (416x416)
    std::vector<int64_t> input_shape = {1, 3, 416, 416};
    Tensor input(DataType::FLOAT32, input_shape, "image");

    if (!load_input_file(input_path, input))
    {
        std::cerr << "Error: Could not load input file: " << input_path << std::endl;
        // Fallback: Check parent directory if current fails (This is where i'll probably be more often cuz i rm the build folder frequently)
        std::string alt_path = "../" + input_path;
        if (load_input_file(alt_path, input))
        {
            std::cout << "Found input in parent directory: " << alt_path << std::endl;
        }
        else
        {
            return 1;
        }
    }

    // 2. Run Engine
    try
    {
        std::cout << "Warming up..." << std::endl; // This is mostly debug/internal stuff
        for (int i = 0; i < 5; ++i)
            engine.run(input); // Warmup

        std::cout << "Benchmarking..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        int iterations = 50;
        for (int i = 0; i < iterations; ++i)
        {
            engine.run(input);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Average Inference Time: " << duration / iterations << " ms" << std::endl;
        std::cout << "FPS: " << 1000.0 / (duration / iterations) << std::endl;
        /*std::cout << "Starting inference..." << std::endl;
        engine.run(input);*/
    }
    catch (const std::exception &e)
    {
        std::cerr << "Inference Failed: " << e.what() << std::endl;
        return 1;
    }
    engine.dump_graph("model_graph.dot");

    // 3. Decode Output & Debug
    Tensor &output = engine.get_output();
    float *data = output.data<float>();

    int grid_h = output.shape()[2];
    int grid_w = output.shape()[3];
    int channels = output.shape()[1];

    // --- DEBUG 1: Global Checksum ---
    double global_sum = 0.0;
    for (int i = 0; i < output.size(); ++i)
        global_sum += data[i];

    std::cout << "\n[Global Debug]" << std::endl;
    std::cout << "Output Shape: [" << output.shape()[0] << ", "
              << channels << ", " << grid_h << ", " << grid_w << "]" << std::endl;
    std::cout << "Output Sum:   " << global_sum << std::endl;

    // --- DEBUG 2: Find Max Objectness (Raw Math Check) ---
    float max_conf = -1.0f;
    int best_idx_base = -1;
    int best_grid_x = -1, best_grid_y = -1, best_anchor = -1;

    // Standard Detection Loop
    std::cout << "\n=== Detections (Threshold > 0.3) ===" << std::endl;
    int detections = 0;

    for (int cy = 0; cy < grid_h; ++cy)
    {
        for (int cx = 0; cx < grid_w; ++cx)
        {
            for (int b = 0; b < NUM_ANCHORS; ++b)
            {

                int channel_start = b * BLOCK_SIZE;
                int obj_channel = channel_start + 4;
                int obj_idx = obj_channel * (grid_h * grid_w) + cy * grid_w + cx;

                float raw_obj = data[obj_idx];
                float confidence = sigmoid(raw_obj);

                // Track Best Candidate for Debugging
                if (confidence > max_conf)
                {
                    max_conf = confidence;
                    best_grid_x = cx;
                    best_grid_y = cy;
                    best_anchor = b;
                    // Save the base index for this block (Tx) to print raw values later
                    best_idx_base = channel_start * (grid_h * grid_w) + cy * grid_w + cx;
                }

                // Standard Detection Threshold Logic
                if (confidence > 0.3)
                {
                    // Extract coords...
                    int tx_idx = (channel_start + 0) * (grid_h * grid_w) + cy * grid_w + cx;
                    int ty_idx = (channel_start + 1) * (grid_h * grid_w) + cy * grid_w + cx;
                    // TODO: add rest of detection logic

                    // Simple print for now to ensure loop is working
                    detections++;
                    std::cout << "!! FOUND OBJECT !! Grid(" << cx << "," << cy << ") Conf: " << confidence << std::endl;
                }
            }
        }
    }

    if (detections == 0)
        std::cout << "No objects detected above threshold." << std::endl;

    // --- DEBUG 3: Print Best Candidate Details ---
    std::cout << "\n[Max Objectness Search]" << std::endl;
    std::cout << "Highest Confidence Found: " << max_conf << std::endl;
    std::cout << "Location: Grid(" << best_grid_x << "," << best_grid_y << ") Anchor " << best_anchor << std::endl;

    if (best_idx_base != -1)
    {
        // Re-calculate indices for the best block to show raw logits

        int offset = (grid_h * grid_w);
        int base_channel = best_anchor * BLOCK_SIZE;

        // Helper lambda to get value at specific channel offset
        auto get_val = [&](int ch_offset)
        {
            return data[(base_channel + ch_offset) * offset + best_grid_y * grid_w + best_grid_x];
        };

        std::cout << "\n[Raw Logits at Best Location]" << std::endl;
        std::cout << "  Tx (Box X): " << get_val(0) << std::endl;
        std::cout << "  Ty (Box Y): " << get_val(1) << std::endl;
        std::cout << "  Tw (Width): " << get_val(2) << std::endl;
        std::cout << "  Th (Height): " << get_val(3) << std::endl;
        std::cout << "  To (Objectness): " << get_val(4) << " -> Sigmoid -> " << max_conf << std::endl;
        std::cout << "  Class Scores (First 5): ";
        for (int i = 0; i < 5; ++i)
            std::cout << get_val(5 + i) << " ";
        std::cout << std::endl;
    }

    return 0;
}
