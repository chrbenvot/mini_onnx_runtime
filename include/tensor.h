#pragma once
#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cstdint>
#include <string>
#include <stdexcept>
enum class DataType
{ // So we can store the data as bytes (like ONNX does) and reinterpret according to dtype
    FLOAT32,
    INT64,
    INT32,
    INT8 // If we ever try quantization
};
class Tensor
{
public:
    // Constructors
    Tensor() = default;
    Tensor(DataType dtype, const std::vector<int64_t> &shape, std::string name = "")
        : m_dtype(dtype), m_name(std::move(name)) // Using move semantics
    {
        m_bytes_per_elements = get_dtype_size(dtype);
        reshape(shape);
    }

    // Change dimensions & recalculate strides
    void reshape(const std::vector<int64_t> &shape)
    {
        m_shape = shape;
        update_metadata();
        m_buffer.resize(m_num_elements * m_bytes_per_elements); // eg: For 100 floats we need to allocate 100*4=400 bytes
    }
    // Returns reference to allow for modification (tensor.at(x)=y)
    template <typename T>
    T &at(const std::vector<int64_t> &indices)
    {
        int64_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            offset += indices[i] * m_strides[i];
        }
        return data<T>()[offset];
    }
    // Read only accessor for faster access
    template <typename T>
    const T &at(const std::vector<int64_t> &indices) const
    {
        int64_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            offset += indices[i] * m_strides[i];
        }
        return data<T>()[offset]; // We cast the buffer to a pointer to T and use pointer arithemtic
    }
    // Low-level access (possibly useful for optimization later)
    template <typename T>
    T *data()
    {
        assert(sizeof(T) == m_bytes_per_elements && "Type mismatch in tensor access!");
        return reinterpret_cast<T *>(m_buffer.data());
    }
    template <typename T>
    const T *data() const
    {
        assert(sizeof(T) == m_bytes_per_elements && "Type mismatch in tensor access!");
        return reinterpret_cast<const T *>(m_buffer.data());
    }
    // Getters + Setters
    const std::string &name() const { return m_name; }
    void set_name(const std::string &name) { m_name = name; }
    const std::vector<int64_t> &shape() const { return m_shape; }
    const std::vector<int64_t> &strides() const { return m_strides; }
    DataType dtype() const { return m_dtype; }
    int64_t size() const
    {
        return m_num_elements;
    }
    // Helper for how many bytes per element
    size_t element_size() const
    {
        return m_bytes_per_elements;
    }
    // Debugging
    void print_info() const
    {
        std::cout << "Tensor: '" << m_name << "' | Shape: [";
        for (size_t i = 0; i < m_shape.size(); ++i)
        {
            std::cout << m_shape[i] << (i < m_shape.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }

private:
    std::string m_name; // in ONNX tensors have names (eg: 'Conv1_Weight' )
    DataType m_dtype = DataType::FLOAT32;
    std::vector<uint8_t> m_buffer; // raw memory
    std::vector<int64_t> m_shape;
    std::vector<int64_t> m_strides;
    int64_t m_num_elements = 0;
    size_t m_bytes_per_elements = 4; // default is float

    // Helper function to calculate strides for row-major layout
    // Example: Shape [2, 3, 4 ] should return Strides[12,4,1]
    void update_metadata()
    {
        // Step 1: calculate Strides:
        m_strides.resize(m_shape.size());
        int64_t stride = 1;
        // Iterating backwards( so no size_t because it's unsigned)
        if (m_shape.empty())
            return;
        for (int i = m_shape.size() - 1; i >= 0; --i)
        {
            m_strides[i] = stride;
            stride *= m_shape[i];
        }
        // Step 2: Calculate total elements
        m_num_elements = 1;
        for (auto &dim : m_shape)
            m_num_elements *= dim;
    }
    // static helper (static because invoked insize constructor)
    static size_t get_dtype_size(DataType dt)
    {
        switch (dt)
        {
        case DataType::FLOAT32:
            return 4;
        case DataType::INT64:
            return 8;
        case DataType::INT32:
            return 4;
        case DataType::INT8:
            return 1;
        default:
            return 1;
        }
    }
};
