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
#include <cublas_v2.h>

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
    InferenceEngine()
    {
        cublasCreate(&m_cublas_handle);
    }

    ~InferenceEngine()
    {
        cublasDestroy(m_cublas_handle);
        // we own the operators stored in the plan so we need to handle clean up
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
            std::string type = step.op->get_op_type();
            if (type == "Conv" || type == "Gemm" || type == "Relu" || type == "BatchNormalization" || type == "LeakyRelu" || type == "MaxPool" || type == "Mul" || type == "Add")
                m_mode = CUDA; // This shouldn't be like this lul but We added them incrementally
            else
                m_mode = CPU;
            if (m_mode == CUDA)
            {
                // We are about to run on GPU. Ensure all inputs are in VRAM.
                for (Tensor *input : step.inputs)
                {
                    if (!input->is_on_device())
                    {
                        input->allocate_device_memory();
                        input->copy_to_device(); // RAM -> VRAM
                    }
                }
            }
            else
            {
                // We are about to run on CPU. Ensure all inputs are in RAM.
                for (Tensor *input : step.inputs)
                {
                    if (input->is_on_device())
                    {
                        input->copy_to_host(); // VRAM -> RAM 
                        // Optional: free device memory to save space,
                        // but keeping it is faster if needed again.
                    }
                }
            }
            {
                ScopedTimer layer_timer(step.debug_name);
                //
                step.op->forward(m_mode, step.inputs, step.outputs, *step.node, m_workspace, m_cublas_handle);
            }
        }
        for (const auto &output_node : m_graph.output())
        {
            std::string name = output_node.name();
            if (m_tensor_registry.count(name))
            {
                Tensor &out = m_tensor_registry[name];
                if (out.is_on_device())
                {
                    out.copy_to_host(); // Force VRAM -> RAM sync
                }
            }
        }
        std::cout << "Inference Complete." << std::endl;
    }

    Tensor &get_output()
    {
        std::string output_name = m_graph.output(0).name();
        return m_tensor_registry[output_name];
    }
    void dump_graph(const std::string &filename) const
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        // DOT language header,this isn't all that useful in restrospect cuz we added parsing in the GUI app directly
        file << "digraph G {" << std::endl;
        file << "  rankdir=LR;" << std::endl; // Left to Right layout
        file << "  node [shape=box, style=\"filled\", fillcolor=\"#E6E6FF\"];" << std::endl;

        // 1. Define all Tensors (Nodes)
        // Set colors based on whether it's a constant/weight or a dynamic activation
        for (const auto &pair : m_tensor_registry)
        {
            bool is_weight = (pair.second.size() > 0 && pair.second.shape().size() > 1 && pair.second.shape()[0] == 1); // Simple heuristic
            std::string color = is_weight ? "gray" : "white";
            std::string label = pair.first + "\\nShape: [";
            for (size_t i = 0; i < pair.second.shape().size(); ++i)
            {
                label += std::to_string(pair.second.shape()[i]);
                if (i < pair.second.shape().size() - 1)
                    label += ",";
            }
            label += "]";

            file << "  \"" << pair.first << "\" [label=\"" << label << "\", fillcolor=\"" << color << "\"];" << std::endl;
        }

        // 2. Define all Operators (Nodes)
        // Operators will be colored light blue
        file << "  node [shape=box, style=\"filled\", fillcolor=\"#B0E0E6\"];" << std::endl;
        for (const auto &step : m_execution_plan)
        {
            file << "  \"" << step.node->name() << "\" [label=\"" << step.op->get_op_type() << "\"];" << std::endl;
        }

        // 3. Define Edges (Connections)
        // Loop through the execution plan again to draw the flow.
        file << "  edge [color=black];" << std::endl;
        for (const auto &step : m_execution_plan)
        {
            const onnx::NodeProto *node = step.node;
            std::string op_name = node->name();

            // Edges from Input Tensors to Operator
            for (const auto &input_name : node->input())
            {
                file << "  \"" << input_name << "\" -> \"" << op_name << "\" [label=\"in\"];" << std::endl;
            }

            // Edges from Operator to Output Tensors
            for (const auto &output_name : node->output())
            {
                file << "  \"" << op_name << "\" -> \"" << output_name << "\" [label=\"out\"];" << std::endl;
            }
        }

        file << "}" << std::endl;
        file.close();
        std::cout << "Graphviz DOT file written to " << filename << std::endl;
    }
    ExecutionMode m_mode = CPU; // TODO: add proper execution switching probably

    // --- Internal architecture MODE HELPERS ---

    // 1. Get a list of all layer names (for the dropdown)
    std::vector<std::string> get_layer_names() const {
        std::vector<std::string> names;
        for (const auto& pair : m_tensor_registry) {
            names.push_back(pair.first);
        }
        return names;
    }

    // 2. Get a pointer to a specific internal tensor
    Tensor* get_internal_tensor(const std::string& name) {
        if (m_tensor_registry.find(name) != m_tensor_registry.end()) {
            return &m_tensor_registry[name];
        }
        return nullptr;
    }
    std::vector<ExecutionStep> get_execution_plan() // This is needed for the GUI so we can show the ORDERED architecture
    {
        return m_execution_plan;
    }
private:
    onnx::GraphProto m_graph;
    std::map<std::string, Tensor> m_tensor_registry;
    std::vector<ExecutionStep> m_execution_plan; // The list of instructions
    std::vector<float> m_workspace;
    cublasHandle_t m_cublas_handle;
};
