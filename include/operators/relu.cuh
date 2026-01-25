#pragma once
void relu_cuda_wrapper(const float*in,float* out, int64_t n);

// this was the first GPU coded op,we decided to change how we're doing it
