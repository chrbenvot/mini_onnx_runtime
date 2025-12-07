#include <iostream>
#include "model_loader.h"

int main(int argc, char** argv) {
    std::string model_path = "model.onnx"; 
    
    // Allow passing path as argument: ./engine ../data/my_model.onnx
    if (argc > 1) {
        model_path = argv[1];
    }

    ModelLoader loader;
    if (loader.load(model_path)) {
        std::cout << "[Success] Model Loaded!" << std::endl;
        loader.print_graph_info();
    } else {
        std::cerr << "[Error] Failed to load model." << std::endl;
        return 1;
    }

    return 0;
}
