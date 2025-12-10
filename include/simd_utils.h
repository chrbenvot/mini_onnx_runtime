#pragma once
#include <immintrin.h> // Header for AVX intrinsics
#include <algorithm>   // for std::max

// Computes dot product of two vectors of length N using AVX2
// Note: We DO NOT parallelize this inner loop because it is usually called 
// inside a parallelized outer loop (in Conv/Gemm). Nested parallelism here would hurt performance.
inline float dot_product_avx(const float *a, const float *b, int64_t n)
{
    __m256 sum_vec = _mm256_setzero_ps();
    int64_t i = 0;

    for (; i <= n - 8; i += 8)
    {
        __m256 vec_a = _mm256_loadu_ps(a + i);
        __m256 vec_b = _mm256_loadu_ps(b + i);

#ifdef __FMA__
        sum_vec = _mm256_fmadd_ps(vec_a, vec_b, sum_vec);
#else
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vec_a, vec_b));
#endif
    }

    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);

    float total = 0.0f;
    for (int k = 0; k < 8; ++k)
        total += temp[k];

    for (; i < n; ++i)
    {
        total += a[i] * b[i];
    }

    return total;
}

// --- Parallelized Element-wise Add ---
inline void add_avx(const float *a, const float *b, float *out, int64_t n)
{
    #pragma omp parallel for if (n > 4096)
    for (int64_t i = 0; i <= n - 8; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 res = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, res);
    }
    
    // Remainder loop (Serial)
    int64_t remainder_start = (n / 8) * 8;
    for (int64_t i = remainder_start; i < n; ++i)
        out[i] = a[i] + b[i];
}

// --- Parallelized Element-wise Mul ---
inline void mul_avx(const float *a, const float *b, float *out, int64_t n)
{
    #pragma omp parallel for if (n > 4096)
    for (int64_t i = 0; i <= n - 8; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 res = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, res);
    }

    int64_t remainder_start = (n / 8) * 8;
    for (int64_t i = remainder_start; i < n; ++i)
        out[i] = a[i] * b[i];
}

// --- Parallelized Leaky Relu ---
inline void leaky_relu_avx(const float *in, float *out, int64_t n, float alpha)
{
    __m256 v_alpha = _mm256_set1_ps(alpha);

    #pragma omp parallel for if (n > 4096)
    for (int64_t i = 0; i <= n - 8; i += 8)
    {
        __m256 val = _mm256_loadu_ps(in + i);
        __m256 val_neg = _mm256_mul_ps(val, v_alpha);
        
        // max(val, val * alpha) handles both positive and negative cases correctly for alpha < 1
        __m256 res = _mm256_max_ps(val, val_neg);
        _mm256_storeu_ps(out + i, res);
    }

    int64_t remainder_start = (n / 8) * 8;
    for (int64_t i = remainder_start; i < n; ++i)
    {
        float val = in[i];
        out[i] = (val >= 0) ? val : (val * alpha);
    }
}

// --- Parallelized Standard ReLU ---
inline void relu_avx(const float *in, float *out, int64_t n)
{
    __m256 v_zero = _mm256_setzero_ps(); 

    #pragma omp parallel for if (n > 4096)
    for (int64_t i = 0; i <= n - 8; i += 8)
    {
        __m256 val = _mm256_loadu_ps(in + i);
        __m256 res = _mm256_max_ps(val, v_zero);
        _mm256_storeu_ps(out + i, res);
    }

    int64_t remainder_start = (n / 8) * 8;
    for (int64_t i = remainder_start; i < n; ++i)
    {
        out[i] = std::max(0.0f, in[i]);
    }
}

// --- Parallelized Scale and Shift (BatchNormalization) ---
inline void scale_shift_avx(const float *in, float *out, int64_t n, float scale, float bias)
{
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256 v_bias = _mm256_set1_ps(bias);

    #pragma omp parallel for if (n > 4096)
    for (int64_t i = 0; i <= n - 8; i += 8)
    {
        __m256 val = _mm256_loadu_ps(in + i);
#ifdef __FMA__
        __m256 res = _mm256_fmadd_ps(val, v_scale, v_bias);
#else
        __m256 res = _mm256_add_ps(_mm256_mul_ps(val, v_scale), v_bias);
#endif
        _mm256_storeu_ps(out + i, res);
    }

    int64_t remainder_start = (n / 8) * 8;
    for (int64_t i = remainder_start; i < n; ++i)
    {
        out[i] = in[i] * scale + bias;
    }
}
