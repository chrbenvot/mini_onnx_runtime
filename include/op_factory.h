#pragma once
#include <string>
#include "operator.h"

// Include all ops
#include "operators/relu.h"
#include "operators/conv.h"
#include "operators/flatten.h"
#include "operators/gemm.h"
#include "operators/maxpool.h"
#include "operators/batchnorm.h"
#include "operators/softmax.h"
#include "operators/avgpool.h"
#include "operators/add.h"
#include "operators/global_avgpool.h"
#include "operators/concat.h"
#include "operators/upsample.h"
#include "operators/leakyrelu.h"
#include "operators/sigmoid.h"
#include "operators/mul.h"

class OpFactory {
public:
    static Operator* create(const std::string& type) {
        if (type == "Relu") return new ReluOp();
        if (type == "Conv") return new ConvOp();
        if (type == "Flatten") return new FlattenOp();
        if (type == "Gemm") return new GemmOp();
        if (type == "MaxPool") return new MaxPoolOp();
        if (type == "BatchNormalization") return new BatchNorm();
        if (type == "Softmax") return new SoftmaxOp();
        if (type == "AveragePool") return new AvgPoolOp();
        if (type == "Add") return new AddOp();
        if (type == "GlobalAveragePool") return new GlobalAvgPoolOp();
        if (type == "Concat") return new ConcatOp();
        if (type == "Resize" || type == "Upsample") return new UpsampleOp();
        if (type == "LeakyRelu") return new LeakyReluOp();
        if (type == "Sigmoid") return new SigmoidOp();
        if (type == "Mul") return new MulOp();
        return nullptr;
    }
};
