#pragma once
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include "tensor.h"
#include "operator.h"
#include "model_loader.h"
#include "timer.h"

#include <operators/relu.h> // May need an umbrella header..
#include <operators/flatten.h>
#include <operators/conv.h>
#include <operators/gemm.h>
#include <operators/maxpool.h>
#include <operators/batchnorm.h>
#include <operators/softmax.h>
#include <operators/avgpool.h>
#include <operators/add.h>
#include <operators/global_avgpool.h>
#include <operators/concat.h>
#include <operators/upsample.h>
#include <operators/leakyrelu.h>
#include <operators/sigmoid.h>
#include <operators/mul.h>

class InferenceEngine
{
public:
    InferenceEngine()
    {
        register_op("Relu", new ReluOp());
        register_op("Conv", new ConvOp());
        register_op("Flatten", new FlattenOp());
        register_op("Gemm", new GemmOp());
        register_op("MaxPool", new MaxPoolOp());
        register_op("BatchNormalization", new BatchNorm());
        register_op("Softmax", new SoftmaxOp());
        register_op("AveragePool", new AvgPoolOp());
        register_op("Add", new AddOp());
        register_op("GlobalAveragePool", new GlobalAvgPoolOp());
        register_op("Concat", new ConcatOp());
        register_op("Resize", new UpsampleOp());
        register_op("LeakyRelu", new LeakyReluOp());
        register_op("Sigmoid", new SigmoidOp());
        register_op("Mul", new MulOp());
        /*
        TODO: add Ops as we go,eg: also probably make a list for this or something...
        register_op("Conv",new ConvOp());
        */
    }
    ~InferenceEngine()
    {
        // Cleanup dynamically allocated Operator Objects
        for (auto &pair : m_op_registry)
        {
            delete pair.second;
        }
    }
    // Load the model
    void load_model(ModelLoader &loader)
    {
        m_graph = loader.graph();
        // Load weights into the Tensor registry
        // Here the engine owns all the tensors
        const auto &weights = loader.weights();
        for (const auto &pair : weights)
        {
            m_tensor_registry[pair.first] = pair.second;
        }
        std::cout << "Engine ready. Tensors in Memory: " << m_tensor_registry.size() << std::endl;
    }
    // The main execution loop
    void run(const Tensor &input_data)
    {

        ScopedTimer total_timer("TOTAL INFERENCE");
        // Inject the input data
        if (m_graph.input_size() > 0)
        {
            std::string input_name = m_graph.input(0).name();
            m_tensor_registry[input_name] = input_data;
        }
        std::cout << "Starting inference..." << std::endl;

        // Iterate through the graph's nodes

        for (int i = 0; i < m_graph.node_size(); ++i)
        {
            const onnx::NodeProto &node = m_graph.node(i);
            // check if this is a Node our engine supports
            if (m_op_registry.find(node.op_type()) == m_op_registry.end())
            { // return iterator is the end so it's not in our registry
                std::cerr << "CRITICAL WARNING: This model contains an Unsupported Operator '" << node.op_type() << "'" << std::endl;
                continue; // This WILL CAUSE CRASHES, but for now we're just debugging
            }
            Operator *op = m_op_registry[node.op_type()];

            // Prepare the inputs and outputs for this operator
            std::vector<Tensor *> op_inputs;
            std::vector<Tensor *> op_outputs;

            // Look up the inputs in the registry
            for (const auto &input_name : node.input())
            {
                if (m_tensor_registry.find(input_name) == m_tensor_registry.end())
                {
                    std::cerr << "  Error: Missing input tensor '" << input_name << "'" << std::endl;
                }
                op_inputs.push_back(&m_tensor_registry[input_name]);
            }

            // Prepare the ouputs (Create empty ones in the registry)
            for (const auto &output_name : node.output())
            {
                // Placeholder Tensor,the Operator is gonna resize it anyway
                m_tensor_registry[output_name] = Tensor(DataType::FLOAT32, {}); // TODO: add other type supports?
                m_tensor_registry[output_name].set_name(output_name);           // do it in constructor?
                op_outputs.push_back(&m_tensor_registry[output_name]);
            }
            // Execute
            std::cout << " Running Op" << node.op_type() << " (" << node.name() << ")" << std::endl;
            {
                std::string label = node.op_type() + " (" + node.name() + ")";
                ScopedTimer layer_timer(label);

                op->forward(op_inputs, op_outputs, node);
            }
        }
        std::cout << "Inference Complete." << std::endl;
    }
    // Helper function to get the final result
    Tensor &get_output()
    {
        std::string output_name = m_graph.output(0).name();
        return m_tensor_registry[output_name];
    }

private:
    onnx::GraphProto m_graph;
    std::map<std::string, Tensor> m_tensor_registry; // For Memory(weights etc...)
    std::map<std::string, Operator *> m_op_registry; // For Logic
    void register_op(std::string name, Operator *op)
    {
        m_op_registry[name] = op;
    }
};
