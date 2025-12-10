#pragma once
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include "tensor.h"
#include "operator.h"
#include "model_loader.h"
#include "timer.h"
#include "op_factory.h" 

// The "Instruction" for a single layer
struct ExecutionStep
{
    Operator *op;
    const onnx::NodeProto *node;
    std::vector<Tensor *> inputs;
    std::vector<Tensor *> outputs;
    std::string debug_name;
};

class InferenceEngine
{
public:
    InferenceEngine() = default;

    ~InferenceEngine()
    {
        // Now we own the operators stored in the plan
        for (auto &step : m_execution_plan)
        {
            delete step.op;
        }
    }

    void load_model(ModelLoader &loader)
    {
        m_graph = loader.graph();

        // 1. Load Weights into Registry
        const auto &weights = loader.weights();
        for (const auto &pair : weights)
        {
            m_tensor_registry[pair.first] = pair.second;
        }

        std::cout << "Building Execution Plan..." << std::endl;

        // 2. Build the Plan (Compile Time)
        for (int i = 0; i < m_graph.node_size(); ++i)
        {
            const onnx::NodeProto &node = m_graph.node(i);

            Operator *op = OpFactory::create(node.op_type());
            if (!op)
            {
                std::cerr << "CRITICAL ERROR: Unsupported Operator '" << node.op_type() << "'" << std::endl;
                continue;
            }

            ExecutionStep step;
            step.op = op;
            step.node = &node;
            step.debug_name = node.op_type() + " (" + node.name() + ")";

            // Pre-resolve Inputs
            for (const auto &input_name : node.input())
            {
                // If input doesn't exist (activations from previous layers), create placeholder
                if (m_tensor_registry.find(input_name) == m_tensor_registry.end())
                {
                    m_tensor_registry[input_name] = Tensor(DataType::FLOAT32, {}, input_name);
                }
                // Store POINTER to the tensor map entry.
                // Map pointers are stable even if we add more keys later.
                step.inputs.push_back(&m_tensor_registry[input_name]);
            }

            // Pre-resolve Outputs
            for (const auto &output_name : node.output())
            {
                m_tensor_registry[output_name] = Tensor(DataType::FLOAT32, {}, output_name);
                step.outputs.push_back(&m_tensor_registry[output_name]);
            }

            m_execution_plan.push_back(std::move(step));
        }

        std::cout << "Plan built. Steps: " << m_execution_plan.size() << std::endl;
    }

    void run(const Tensor &input_data)
    {
        ScopedTimer total_timer("TOTAL INFERENCE");

        // 1. Set Input Data
        // We copy the data into the registry, but the POINTER stored in execution_plan remains valid.
        if (m_graph.input_size() > 0)
        {
            std::string input_name = m_graph.input(0).name();
            m_tensor_registry[input_name] = input_data;
        }

        std::cout << "Executing Plan..." << std::endl;

        // 2. Execute (Fast Loop)
        for (auto &step : m_execution_plan)
        {
            {
                ScopedTimer layer_timer(step.debug_name);
                step.op->forward(step.inputs, step.outputs, *step.node);
            }
        }
        std::cout << "Inference Complete." << std::endl;
    }

    Tensor &get_output()
    {
        std::string output_name = m_graph.output(0).name();
        return m_tensor_registry[output_name];
    }

private:
    onnx::GraphProto m_graph;
    std::map<std::string, Tensor> m_tensor_registry;
    std::vector<ExecutionStep> m_execution_plan; // The list of instructions
};
