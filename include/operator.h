#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "tensor.h"

//Abstract class for all Layers
class Operator{
public:
    virtual ~Operator() = default;
    virtual void forward(const std::vector<Tensor*>& inputs,std::vector<Tensor*>& outputs ) = 0;

    std::string name;
};
