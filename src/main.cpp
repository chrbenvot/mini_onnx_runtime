#include <iostream>
#include "model_loader.h"
#include "engine.h"

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../model.onnx";
    
    // 1. Load Model
    ModelLoader loader;
    if (!loader.load(path)) return 1;

    // 2. Initialize Engine
    InferenceEngine engine;
    engine.load_model(loader);

    // 3. Create Dummy Input (1x1x28x28 Image)
    // We fill it with 1.0f just for testing
    std::vector<int64_t> input_shape = {1, 1, 28, 28};
    Tensor input(DataType::FLOAT32, input_shape, "Input3");
    float* data = input.data<float>();
    for (int i = 0; i < input.size(); ++i) data[i] = 1.0f;

    // 4. Run
    try {
        engine.run(input);
    } catch (const std::exception& e) {
        std::cerr << "Execution Failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
