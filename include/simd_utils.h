#pragma once
#include <immintrin.h> // Header for AVX intrinsics

// Computes dot product of two vectors of length N using AVX2 (8 floats at a time)
inline float dot_product_avx(const float *a, const float *b, int64_t n)
{
    // 1. Initialize accumulator register to all zeros
    // __m256 is a datatype that holds 8 floats (256 bits)
    __m256 sum_vec = _mm256_setzero_ps();

    int64_t i = 0;

    // 2. Main Loop: Process 8 floats at a time
    // We stop when we have fewer than 8 elements left
    for (; i <= n - 8; i += 8)
    {
        // Load 8 floats from A and B into registers
        // _mm256_loadu_ps handles unaligned memory (safer)
        __m256 vec_a = _mm256_loadu_ps(a + i);
        __m256 vec_b = _mm256_loadu_ps(b + i);

        // Multiply them: vec_a * vec_b
        // Then add result to our running sum vector
        // FMA (Fused Multiply Add) instruction: sum += a * b
        // If FMA is not supported, we use _mm256_add_ps(_mm256_mul_ps(vec_a, vec_b))

#ifdef __FMA__
        sum_vec = _mm256_fmadd_ps(vec_a, vec_b, sum_vec);
#else
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vec_a, vec_b));
#endif
    }

    // 3. Reduction: Sum the 8 values inside sum_vec into a single float
    // This part is a bit tricky, often done by horizontal additions
    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);

    float total = 0.0f;
    for (int k = 0; k < 8; ++k)
        total += temp[k];

    // 4. Handle Remainder (0 to 7 elements left)
    // Standard scalar loop for the leftovers
    for (; i < n; ++i)
    {
        total += a[i] * b[i];
    }

    return total;
}
inline void add_avx(const float *a, const float *b, float *out, int64_t n)
{
    int64_t i = 0;
    for (; i <= n - 8; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 res = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, res);
    }
    // Remainder
    for (; i < n; ++i)
        out[i] = a[i] + b[i];
}

// --- NEW: Element-wise Mul ---
inline void mul_avx(const float *a, const float *b, float *out, int64_t n)
{
    int64_t i = 0;
    for (; i <= n - 8; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 res = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < n; ++i)
        out[i] = a[i] * b[i];
}

// --- NEW: Leaky Relu ---
// Logic: out = (val >= 0) ? val : val * alpha
// AVX Trick: max(val, val*alpha) works if alpha < 1
inline void leaky_relu_avx(const float *in, float *out, int64_t n, float alpha)
{
    __m256 v_alpha = _mm256_set1_ps(alpha); // Broadcast alpha to all 8 slots

    int64_t i = 0;
    for (; i <= n - 8; i += 8)
    {
        __m256 val = _mm256_loadu_ps(in + i);

        // Calculate val * alpha
        __m256 val_neg = _mm256_mul_ps(val, v_alpha);

        // Take the max(val, val*alpha)
        // If val is positive, val > val*0.1, so we pick val.
        // If val is negative, val < val*0.1 (e.g. -10 < -1), so we pick val*alpha.
        // WAIT! Standard max_ps compares values.
        // -10 vs -1. -1 is larger. So max picks correct leaky part.
        // 10 vs 1. 10 is larger. Max picks 10.
        // This logic holds for standard LeakyRelu where 0 < alpha < 1.

        __m256 res = _mm256_max_ps(val, val_neg);

        _mm256_storeu_ps(out + i, res);
    }
    // Remainder
    for (; i < n; ++i)
    {
        float val = in[i];
        out[i] = (val >= 0) ? val : (val * alpha);
    }
}
// --- NEW: Standard ReLU (max(0, x)) ---
inline void relu_avx(const float *in, float *out, int64_t n)
{
    __m256 v_zero = _mm256_setzero_ps(); // Vector of all 0.0f

    int64_t i = 0;
    for (; i <= n - 8; i += 8)
    {
        __m256 val = _mm256_loadu_ps(in + i);
        __m256 res = _mm256_max_ps(val, v_zero); // max(val, 0)
        _mm256_storeu_ps(out + i, res);
    }
    // Remainder
    for (; i < n; ++i)
    {
        out[i] = std::max(0.0f, in[i]);
    }
}

// --- NEW: Scale and Shift (for BatchNormalization) ---
// Formula: out = in * scale + bias
inline void scale_shift_avx(const float *in, float *out, int64_t n, float scale, float bias)
{
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256 v_bias = _mm256_set1_ps(bias);

    int64_t i = 0;
    for (; i <= n - 8; i += 8)
    {
        __m256 val = _mm256_loadu_ps(in + i);
// Fused Multiply Add: (val * scale) + bias
#ifdef __FMA__
        __m256 res = _mm256_fmadd_ps(val, v_scale, v_bias);
#else
        __m256 res = _mm256_add_ps(_mm256_mul_ps(val, v_scale), v_bias);
#endif
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < n; ++i)
    {
        out[i] = in[i] * scale + bias;
    }
}
