#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "engine.h"
#include "model_loader.h"

// --- PREPROCESSING ---
struct LetterboxInfo {
    float scale;
    int x_offset;
    int y_offset;
    int original_w;
    int original_h;
};

std::vector<float> preprocess_letterbox(const cv::Mat& src) {
    int target_w = 416;
    int target_h = 416;

    // Calculate Scale
    float scale = std::min((float)target_w / src.cols, (float)target_h / src.rows);
    int new_w = (int)(src.cols * scale);
    int new_h = (int)(src.rows * scale);

    //  Resize
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    // Gray Canvas
    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(128, 128, 128));

    // Center Paste
    int x_offset = (target_w - new_w) / 2;
    int y_offset = (target_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(x_offset, y_offset, new_w, new_h)));

    // Convert
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    std::vector<float> output;
    output.reserve(3 * 416 * 416);
    std::vector<cv::Mat> channels(3);
    cv::split(canvas, channels);

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 416; ++y) {
            for (int x = 0; x < 416; ++x) {
                // Keep 0-255 range to match our current setup
                output.push_back(static_cast<float>(channels[c].at<uint8_t>(y, x))); 
            }
        }
    }
    return output;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./debug_single <model.onnx> <image.jpg>" << std::endl;
        return 1;
    }

    //  Load Engine
    ModelLoader loader;
    if (!loader.load(argv[1])) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }
    InferenceEngine engine;
    engine.load_model(loader);

    //  Load Image
    cv::Mat img = cv::imread(argv[2]);
    if (img.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }

    //  Run Inference
    std::vector<float> input_data = preprocess_letterbox(img);
    std::vector<int64_t> input_shape = {1, 3, 416, 416};
    Tensor input_tensor(DataType::FLOAT32, input_shape, "input");
    std::memcpy(input_tensor.data<float>(), input_data.data(), input_data.size() * sizeof(float));

    std::cout << "Running Inference..." << std::endl;
    engine.run(input_tensor);

    // Print Debug Stats (To see if they're Matching Python)
    Tensor& output = engine.get_output();
    const float* out_data = output.data<float>();
    size_t count = output.size();

    float min_val = 1e9, max_val = -1e9;
    double sum = 0.0;

    for (size_t i = 0; i < count; ++i) {
        float val = out_data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "--- C++ Output Stats ---" << std::endl;
    std::cout << "Output Count: " << count << " floats" << std::endl;
    std::cout << "Min Value:    " << min_val << std::endl;
    std::cout << "Max Value:    " << max_val << std::endl;
    std::cout << "Mean Value:   " << (sum / count) << std::endl;
    std::cout << "------------------------" << std::endl;
    
    std::cout << "First 5 values: ";
    for(int i=0; i<5; i++) std::cout << out_data[i] << " ";
    std::cout << std::endl;

    std::cout << "Center 5 values: ";
    size_t center = count / 2;
    for(int i=0; i<5; i++) std::cout << out_data[center + i] << " ";
    std::cout << std::endl;

    return 0;
}
