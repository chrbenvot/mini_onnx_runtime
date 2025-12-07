#pragma once
#include "../operator.h"
#include <algorithm> // For std::max

class ReluOp : public Operator {
public:
    void forward(const std::vector<Tensor*>& inputs,std::vector<Tensor*>& outputs) override {
        if (inputs.empty()|| outputs.empty()){
            std::cerr<< "Error: ReLU missing input or output tensors." << std::endl;
            return;
        }

        Tensor* input = inputs[0];
        Tensor* output = outputs[0];
        // The shape of the output needs to be the same as that of the input
        if (output->shape() != input->shape() || output->dtype() != input->dtype()) {
            output->reshape(input->shape()); 
            // Note: In a real engine, you'd set output->dtype here too, 
            // but your current Tensor implementation sets dtype in constructor.
        }
        switch (input->dtype()) {
            case DataType::FLOAT32:
                relu_kernel<float>(input, output);
                break;
            case DataType::INT32:
                relu_kernel<int32_t>(input, output);
                break;
            case DataType::INT8:
                relu_kernel<int8_t>(input, output);
                break;
            default:
                std::cerr << "Error: Unsupported DataType for ReLU" << std::endl;
                break;
        }
    }
private:
    template <typename T>
    void relu_kernel(Tensor* input,Tensor* output) {
        T* in_ptr = input->data<T>();
        T* out_ptr = output->data<T>();
        int64_t size = input->size();
        // Potential optimization ?? #pragma omp parallel for
        for (int64_t i =0;i<size;++i){
            out_ptr[i]=std::max(static_cast<T>(0),in_ptr[i]);  // static cast to ensure same type comparison
        }
    }
};
