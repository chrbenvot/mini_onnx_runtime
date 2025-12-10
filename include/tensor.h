#pragma once
#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <cstring> // For memcpy if needed, though vectors handle it

enum class DataType
{ 
    FLOAT32,
    INT64,
    INT32,
    INT8,
    UINT8
};

class Tensor
{
public:
    // --- 1. Default & Standard Constructors ---
    Tensor() = default;

    Tensor(DataType dtype, const std::vector<int64_t> &shape, std::string name = "")
        : m_name(std::move(name)), m_dtype(dtype) 
    {
        m_bytes_per_elements = get_dtype_size(dtype);
        reshape(shape);
    }

    // --- 2. Move Constructor (The Speed Upgrade) ---
    // Called when: Tensor t = std::move(temp_tensor);
    Tensor(Tensor&& other) noexcept 
        : m_name(std::move(other.m_name)),
          m_dtype(other.m_dtype),
          m_buffer(std::move(other.m_buffer)),      // O(1) Pointer Steal
          m_shape(std::move(other.m_shape)),        // O(1)
          m_strides(std::move(other.m_strides)),    // O(1)
          m_num_elements(other.m_num_elements),
          m_bytes_per_elements(other.m_bytes_per_elements)
    {
        // Reset the source to a clean empty state
        other.m_num_elements = 0;
        other.m_shape.clear();
        other.m_strides.clear();
        // other.m_buffer is already empty after std::move
    }

    // --- 3. Move Assignment Operator ---
    // Called when: registry["out"] = Tensor(...);
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            m_name = std::move(other.m_name);
            m_dtype = other.m_dtype;
            m_buffer = std::move(other.m_buffer);   // O(1) Pointer Steal
            m_shape = std::move(other.m_shape);
            m_strides = std::move(other.m_strides);
            m_num_elements = other.m_num_elements;
            m_bytes_per_elements = other.m_bytes_per_elements;

            // Reset source
            other.m_num_elements = 0;
            other.m_shape.clear();
            other.m_strides.clear();
        }
        return *this;
    }

    // --- 4. Copy Constructor (The Fallback) ---
    // Called when: Tensor t = existing_tensor;
    Tensor(const Tensor& other)
        : m_name(other.m_name),
          m_dtype(other.m_dtype),
          m_buffer(other.m_buffer),             // Deep Copy (Slow)
          m_shape(other.m_shape),
          m_strides(other.m_strides),
          m_num_elements(other.m_num_elements),
          m_bytes_per_elements(other.m_bytes_per_elements)
    {
        // std::cout << "[WARN] Deep Copy of Tensor: " << m_name << std::endl;
    }

    // --- 5. Copy Assignment Operator ---
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            // std::cout << "[WARN] Deep Copy Assignment of Tensor: " << other.m_name << std::endl;
            m_name = other.m_name;
            m_dtype = other.m_dtype;
            m_buffer = other.m_buffer;          // Deep Copy (Slow)
            m_shape = other.m_shape;
            m_strides = other.m_strides;
            m_num_elements = other.m_num_elements;
            m_bytes_per_elements = other.m_bytes_per_elements;
        }
        return *this;
    }

    // --- Existing Functionality ---

    void reshape(const std::vector<int64_t> &shape)
    {
        m_shape = shape;
        update_metadata();
        // Only reallocate if size actually grows to avoid thrashing
        size_t required_bytes = m_num_elements * m_bytes_per_elements;
        if (m_buffer.size() < required_bytes) {
            m_buffer.resize(required_bytes);
        }
        // If we shrank, we usually just keep the capacity (optimization)
    }

    template <typename T>
    T &at(const std::vector<int64_t> &indices)
    {
        int64_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * m_strides[i];
        }
        return data<T>()[offset];
    }

    template <typename T>
    const T &at(const std::vector<int64_t> &indices) const
    {
        int64_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * m_strides[i];
        }
        return data<T>()[offset];
    }

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

    void *raw_data() { return m_buffer.data(); }
    const void *raw_data() const { return m_buffer.data(); }

    // Getters + Setters
    const std::string &name() const { return m_name; }
    void set_name(const std::string &name) { m_name = name; }
    const std::vector<int64_t> &shape() const { return m_shape; }
    const std::vector<int64_t> &strides() const { return m_strides; }
    DataType dtype() const { return m_dtype; }
    int64_t size() const { return m_num_elements; }
    size_t element_size() const { return m_bytes_per_elements; }

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
    std::string m_name; 
    DataType m_dtype = DataType::FLOAT32;
    std::vector<uint8_t> m_buffer; 
    std::vector<int64_t> m_shape;
    std::vector<int64_t> m_strides;
    int64_t m_num_elements = 0;
    size_t m_bytes_per_elements = 4;

    void update_metadata()
    {
        m_strides.resize(m_shape.size());
        int64_t stride = 1;
        if (m_shape.empty()) {
            m_num_elements = 0; // Handle scalar/empty edge case
            return;
        }
        for (int i = m_shape.size() - 1; i >= 0; --i)
        {
            m_strides[i] = stride;
            stride *= m_shape[i];
        }
        m_num_elements = 1;
        for (auto &dim : m_shape)
            m_num_elements *= dim;
    }

    static size_t get_dtype_size(DataType dt)
    {
        switch (dt)
        {
        case DataType::FLOAT32: return 4;
        case DataType::INT64:   return 8;
        case DataType::INT32:   return 4;
        case DataType::INT8:    return 1;
        default:                return 1;
        }
    }
};
