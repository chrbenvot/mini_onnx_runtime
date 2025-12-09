#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#include <fstream>

#include "model_loader.h"
#include "engine.h"
#include "tensor.h"

// helper to load raw floats from a text file
bool load_input_file(const std::string &filepath, Tensor &input_tensor)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open input file " << filepath << std::endl;
        return false;
    }
    float *data = input_tensor.data<float>();
    float val;
    int idx = 0;
    while (file >> val)
    {
        if (idx >= input_tensor.size())
        {
            std::cerr << "Warning: Input file has more data than tensor size." << std::endl;
            break;
        }
        data[idx++] = val;
    }
    if (idx != input_tensor.size())
    {
        std::cerr << "Warning: Input file size mismatch. Expected " << input_tensor.size()
                  << " got " << idx << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    std::cout << "--- ONNX Inference Engine Demo ---" << std::endl;

    // Default paths
    std::string model_path = (argc > 1) ? argv[1] : "../mnist.onnx";
    std::string input_path = (argc > 2) ? argv[2] : "../input_7.txt";

    // 1. Load Model
    ModelLoader loader;
    if (!loader.load(model_path))
        return 1;

    // 2. Init Engine
    InferenceEngine engine;
    try
    {
        engine.load_model(loader);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Engine Init Failed: " << e.what() << std::endl;
        return 1;
    }

    // 3. Prepare Input Tensor
    std::vector<int64_t> input_shape = {1, 1, 28, 28};
    Tensor input(DataType::FLOAT32, input_shape, "Input3");

    // 4. Load Real Data from Text File
    std::cout << "Loading input image: " << input_path << "..." << std::endl;
    if (!load_input_file(input_path, input))
    {
        // Fallback to dummy data if file missing
        std::cerr << "Falling back to dummy data." << std::endl;
        float *ptr = input.data<float>();
        for (int i = 0; i < input.size(); ++i)
            ptr[i] = 0.0f;
    }

    // 5. Run Inference
    try
    {
        engine.run(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Inference Failed: " << e.what() << std::endl;
        return 1;
    }

    // 6. Print Results (temporary debug to check softmax cuz imported model doesnt have it)
    try
    {
        Tensor &output = engine.get_output();

        // --- MANUAL SOFTMAX STEP ---
        // Create a tensor to hold the probabilities
        Tensor probabilities(DataType::FLOAT32, output.shape());

        // Create and run the Softmax Operator manually
        SoftmaxOp softmax_op;
        std::vector<Tensor *> sm_inputs = {&output};
        std::vector<Tensor *> sm_outputs = {&probabilities};
        onnx::NodeProto dummy_node; // Default attributes (axis=-1) work fine here

        softmax_op.forward(sm_inputs, sm_outputs, dummy_node);

        // --- PRINTING ---
        float *prob_data = probabilities.data<float>();

        int predicted_digit = 0;
        float max_conf = 0.0f; // Probabilities are 0.0 to 1.0

        std::cout << "\n--- Prediction Results ---" << std::endl;
        std::cout << "Class Probabilities: [ ";

        // Iterate through the 10 classes (digits 0-9)
        for (int i = 0; i < 10; ++i)
        {
            // Print neatly with 4 decimal places
            std::cout << std::fixed << std::setprecision(4) << prob_data[i] << " ";

            // Find ArgMax
            if (prob_data[i] > max_conf)
            {
                max_conf = prob_data[i];
                predicted_digit = i;
            }
        }
        std::cout << "]" << std::endl;

        std::cout << "\n========================================" << std::endl;
        std::cout << "   PREDICTED DIGIT: " << predicted_digit << std::endl;
        std::cout << "   Confidence: " << (max_conf * 100.0f) << "%" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error retrieving output: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
