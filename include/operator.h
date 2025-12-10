#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "tensor.h"
#include "onnx.pb.h"

// Abstract class for all Layers
class Operator
{
public:
    virtual ~Operator() = default;
    virtual void forward(const std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs, const onnx::NodeProto &node, std::vector<float> &workspace) = 0;
    virtual std::string get_op_type() const = 0;

protected:
    // Helper for extracting attributes (like pad/kernel size from) from an ONNX protoNode
    std::vector<int64_t> get_int_list_attribute(const onnx::NodeProto &node, const std::string &name)
    {
        for (const auto &attr : node.attribute())
        {
            if (attr.name() == name)
            {
                std::vector<int64_t> values;
                for (auto val : attr.ints())
                {
                    values.push_back(val);
                }
                return values;
            }
        }
        return {}; // Empty if nothing found
    }
    // Get Int attribute (eg: axis,group...)
    int64_t get_int_attribute(const onnx::NodeProto &node, const std::string &name, int64_t default_val)
    {
        for (const auto &attr : node.attribute())
        {
            if (attr.name() == name)
            {
                // Protobuf stores single integers in the 'i' field
                return attr.i();
            }
        }
        return default_val;
    }
    // Get float attribute (eg: alpha epsilon )
    float get_float_attribute(const onnx::NodeProto &node, const std::string &name, float default_val)
    {
        for (const auto &attr : node.attribute())
        {
            if (attr.name() == name)
                return attr.f();
        }
        return default_val;
    }
    // Get String attribute(eg:auto_pad)
    std::string get_string_attribute(const onnx::NodeProto &node, const std::string &name, const std::string &default_val)
    {
        for (const auto &attr : node.attribute())
        {
            if (attr.name() == name)
                return attr.s();
        }
        return default_val;
    }
};
