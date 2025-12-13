#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "onnx.pb.h"
#include "tensor.h" //  Now we need to create Tensors

class ModelLoader
{
public:
    // Load the file from disk
    bool load(const std::string &filepath)
    {
        std::ifstream input(filepath, std::ios::ate | std::ios::binary);
        if (!input.is_open())
        {
            std::cerr << "Error: Could not open " << filepath << std::endl;
            return false;
        }

        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);

        if (!m_model.ParseFromIstream(&input))
        {
            return false;
        }

        m_graph = m_model.graph();

        // Immediately parse all weights into our map
        load_initializers();

        return true;
    }

    // Accessor for the weights registry
    const std::map<std::string, Tensor> &weights() const { return m_weights; }
    const onnx::GraphProto &graph() const { return m_graph; }

    // Debug Print
    void print_info()
    {
        std::cout << "--- Model Info ---" << std::endl;
        std::cout << "Nodes: " << m_graph.node_size() << std::endl;
        std::cout << "Weights (Initializers): " << m_weights.size() << std::endl;

        // Print the first few weights found
        int count = 0;
        for (const auto &pair : m_weights)
        {
            if (count++ > 3)
                break; // Don't spam
            const Tensor &t = pair.second;
            std::cout << "  Weight: " << t.name() << " | Shape: [";
            for (auto d : t.shape())
                std::cout << d << ",";
            std::cout << "]" << std::endl;
        }
    }

private:
    onnx::ModelProto m_model;
    onnx::GraphProto m_graph;
    std::map<std::string, Tensor> m_weights; // The Registry

    // The Logic to convert ONNX Proto -> Tensor Class
    void load_initializers()
    {
        for (const auto &initializer : m_graph.initializer())
        {

            // 1. Get Dimensions
            std::vector<int64_t> shape;
            for (auto dim : initializer.dims())
            {
                shape.push_back(dim);
            }

            // 2. Map DataType (Simple version: Support Float only for now)
            // ONNX defines 1 = FLOAT, 7 = INT64
            DataType type = DataType::FLOAT32;
            if (initializer.data_type() == 1)
            {
                type = DataType::FLOAT32;
            }
            else if (initializer.data_type() == 7)
            {
                type = DataType::INT64;
            }
            else
            {
                // Skip unsupported types (like INT32) for now to avoid crashing
                std::cerr << "Warning: Skipping unsupported weight type: " << initializer.name() << std::endl;
                continue;
            }

            // 3. Create the Tensor
            Tensor t(type, shape, initializer.name());

            // 4. Copy Raw Data
            // ONNX stores data in a raw string called 'raw_data'
            if (initializer.has_raw_data())
            {
                const std::string &raw = initializer.raw_data();

                // Safety check
                if (raw.size() != t.size() * t.element_size())
                {
                    std::cerr << "Error: Size mismatch for tensor " << t.name() << std::endl;
                    continue;
                }

                // Direct memory copy (Fast!)
                std::memcpy(t.raw_data(), raw.data(), raw.size()); // raw_data else the assert in tensor.h fails
            }
            // Fallback: Sometimes ONNX stores floats in a float_data list instead of raw_data
            else if (initializer.float_data_size() > 0)
            {
                float *ptr = t.data<float>();
                for (int i = 0; i < initializer.float_data_size(); ++i)
                {
                    ptr[i] = initializer.float_data(i);
                }
            }

            // Store in map
            m_weights[initializer.name()] = t;
        }
    }
};
