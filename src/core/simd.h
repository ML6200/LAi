#ifndef LAI_CORE_SIMD_H
#define LAI_CORE_SIMD_H

#include "types.h"

// Platform detection
#if defined(__AVX2__)
    #define LAI_AVX2 1
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define LAI_NEON 1
    #include <arm_neon.h>
#endif

namespace lai {
namespace simd {

// Vector width in floats
#if defined(LAI_AVX2)
    constexpr i32 VECTOR_WIDTH = 8;
#elif defined(LAI_NEON)
    constexpr i32 VECTOR_WIDTH = 4;
#else
    constexpr i32 VECTOR_WIDTH = 1;
#endif

// ============================================================================
// AVX2 Implementation
// ============================================================================
#if defined(LAI_AVX2)

// Dot product of two f32 vectors
inline f32 dot_f32(const f32* a, const f32* b, i64 n) {
    __m256 sum = _mm256_setzero_ps();

    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    f32 result = _mm_cvtss_f32(sum128);

    // Handle remainder
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// Vector-scalar multiply and add: y = a * x + y
inline void fma_f32(f32* y, const f32* x, f32 a, i64 n) {
    __m256 va = _mm256_set1_ps(a);

    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        vy = _mm256_fmadd_ps(va, vx, vy);
        _mm256_storeu_ps(y + i, vy);
    }

    for (; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

// Vector add: y = a + b
inline void add_f32(f32* y, const f32* a, const f32* b, i64 n) {
    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(y + i, _mm256_add_ps(va, vb));
    }

    for (; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
}

// Vector multiply: y = a * b
inline void mul_f32(f32* y, const f32* a, const f32* b, i64 n) {
    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(va, vb));
    }

    for (; i < n; ++i) {
        y[i] = a[i] * b[i];
    }
}

// Scale vector: y = a * scale
inline void scale_f32(f32* y, const f32* a, f32 scale, i64 n) {
    __m256 vs = _mm256_set1_ps(scale);

    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(va, vs));
    }

    for (; i < n; ++i) {
        y[i] = a[i] * scale;
    }
}

// Sum of vector elements
inline f32 sum_f32(const f32* a, i64 n) {
    __m256 sum = _mm256_setzero_ps();

    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(a + i));
    }

    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    f32 result = _mm_cvtss_f32(sum128);

    for (; i < n; ++i) {
        result += a[i];
    }

    return result;
}

// Max of vector elements
inline f32 max_f32(const f32* a, i64 n) {
    if (n == 0) return 0.0f;

    __m256 vmax = _mm256_set1_ps(a[0]);

    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(a + i));
    }

    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 max128 = _mm_max_ps(lo, hi);
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));

    f32 result = _mm_cvtss_f32(max128);

    for (; i < n; ++i) {
        if (a[i] > result) result = a[i];
    }

    return result;
}

// ============================================================================
// NEON Implementation (ARM)
// ============================================================================
#elif defined(LAI_NEON)

inline f32 dot_f32(const f32* a, const f32* b, i64 n) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }

    f32 result = vaddvq_f32(sum);

    for (; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

inline void fma_f32(f32* y, const f32* x, f32 a, i64 n) {
    float32x4_t va = vdupq_n_f32(a);

    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        vy = vfmaq_f32(vy, va, vx);
        vst1q_f32(y + i, vy);
    }

    for (; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

inline void add_f32(f32* y, const f32* a, const f32* b, i64 n) {
    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(y + i, vaddq_f32(va, vb));
    }

    for (; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
}

inline void mul_f32(f32* y, const f32* a, const f32* b, i64 n) {
    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(y + i, vmulq_f32(va, vb));
    }

    for (; i < n; ++i) {
        y[i] = a[i] * b[i];
    }
}

inline void scale_f32(f32* y, const f32* a, f32 scale, i64 n) {
    float32x4_t vs = vdupq_n_f32(scale);

    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(y + i, vmulq_f32(va, vs));
    }

    for (; i < n; ++i) {
        y[i] = a[i] * scale;
    }
}

inline f32 sum_f32(const f32* a, i64 n) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        sum = vaddq_f32(sum, vld1q_f32(a + i));
    }

    f32 result = vaddvq_f32(sum);

    for (; i < n; ++i) {
        result += a[i];
    }

    return result;
}

inline f32 max_f32(const f32* a, i64 n) {
    if (n == 0) return 0.0f;

    float32x4_t vmax = vdupq_n_f32(a[0]);

    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(a + i));
    }

    f32 result = vmaxvq_f32(vmax);

    for (; i < n; ++i) {
        if (a[i] > result) result = a[i];
    }

    return result;
}

// ============================================================================
// Scalar Fallback
// ============================================================================
#else

inline f32 dot_f32(const f32* a, const f32* b, i64 n) {
    f32 sum = 0.0f;
    for (i64 i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline void fma_f32(f32* y, const f32* x, f32 a, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

inline void add_f32(f32* y, const f32* a, const f32* b, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
}

inline void mul_f32(f32* y, const f32* a, const f32* b, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        y[i] = a[i] * b[i];
    }
}

inline void scale_f32(f32* y, const f32* a, f32 scale, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        y[i] = a[i] * scale;
    }
}

inline f32 sum_f32(const f32* a, i64 n) {
    f32 sum = 0.0f;
    for (i64 i = 0; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}

inline f32 max_f32(const f32* a, i64 n) {
    if (n == 0) return 0.0f;
    f32 m = a[0];
    for (i64 i = 1; i < n; ++i) {
        if (a[i] > m) m = a[i];
    }
    return m;
}

#endif

// ============================================================================
// Common operations (built on primitives above)
// ============================================================================

// Copy vector
inline void copy_f32(f32* dst, const f32* src, i64 n) {
    std::memcpy(dst, src, n * sizeof(f32));
}

// Fill vector with constant
inline void fill_f32(f32* a, f32 val, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        a[i] = val;
    }
}

// Compute mean
inline f32 mean_f32(const f32* a, i64 n) {
    return sum_f32(a, n) / static_cast<f32>(n);
}

// Compute variance
inline f32 var_f32(const f32* a, i64 n, f32 mean) {
    f32 sum = 0.0f;
    for (i64 i = 0; i < n; ++i) {
        f32 diff = a[i] - mean;
        sum += diff * diff;
    }
    return sum / static_cast<f32>(n);
}

} // namespace simd
} // namespace lai

#endif // LAI_CORE_SIMD_H
