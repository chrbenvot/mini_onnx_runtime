#pragma once
#include <immintrin.h> // Header for AVX intrinsics

// Computes dot product of two vectors of length N using AVX2 (8 floats at a time)
inline float dot_product_avx(const float* a, const float* b, int64_t n) {
    // 1. Initialize accumulator register to all zeros
    // __m256 is a datatype that holds 8 floats (256 bits)
    __m256 sum_vec = _mm256_setzero_ps();
    
    int64_t i = 0;
    
    // 2. Main Loop: Process 8 floats at a time
    // We stop when we have fewer than 8 elements left
    for (; i <= n - 8; i += 8) {
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
    for (int k = 0; k < 8; ++k) total += temp[k];

    // 4. Handle Remainder (0 to 7 elements left)
    // Standard scalar loop for the leftovers
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }

    return total;
}
