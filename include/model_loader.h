#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "onnx.pb.h" 

class ModelLoader {
public:
    // Load the file from disk
    bool load(const std::string& filepath) {
        std::ifstream input(filepath, std::ios::ate | std::ios::binary);
        if (!input.is_open()) {
            std::cerr << "Error: Could not open " << filepath << std::endl;
            return false;
        }

        // Parse file size for debug
        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);
        std::cout << "Loading model (" << size << " bytes)..." << std::endl;

        // The Magic Line: Protobuf parsing
        if (!m_model.ParseFromIstream(&input)) {
            std::cerr << "Error: Failed to parse ONNX content." << std::endl;
            return false;
        }

        m_graph = m_model.graph();
        return true;
    }

    // Print a summary of the network structure
    void print_graph_info() {
        std::cout << "--- Graph Summary ---" << std::endl;
        std::cout << "Graph Name: " << m_graph.name() << std::endl;
        std::cout << "Inputs: " << m_graph.input_size() << std::endl;
        std::cout << "Outputs: " << m_graph.output_size() << std::endl;
        std::cout << "Nodes (Layers): " << m_graph.node_size() << std::endl;

        // Iterate through the nodes (The "Recipe")
        for (int i = 0; i < m_graph.node_size(); ++i) {
            const onnx::NodeProto& node = m_graph.node(i);
            std::cout << "  Node " << i << ": " << node.op_type() 
                      << " (" << node.name() << ")" << std::endl;
            
            // Print Inputs for this node
            std::cout << "    In: ";
            for (const auto& input_name : node.input()) {
                std::cout << input_name << " ";
            }
            std::cout << "\n    Out: ";
            for (const auto& output_name : node.output()) {
                std::cout << output_name << " ";
            }
            std::cout << std::endl;
        }
    }

    // We will use this later to get the actual graph object
    const onnx::GraphProto& graph() const { return m_graph; }

private:
    onnx::ModelProto m_model;
    onnx::GraphProto m_graph;
};
