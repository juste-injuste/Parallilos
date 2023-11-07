// include this file to support signed 32 bit integers
#if not defined(PARALLILOS_HPP)
# error "you must #include Parallilos.hpp before Parallilos_int32.hpp"
#else
#ifndef PARALLILOS_INT32_HPP
#define PARALLILOS_INT32_HPP
#include <cstdint> // for int32_t
namespace Parallilos
{
  inline namespace Frontend
  {
    // define the best SIMD intrinsics to use for signed 32 bit integers
# if defined(PARALLILOS_AVX512F)
#   define PARALLILOS_SET_I32                   "AVX512F"
#   define PARALLILOS_TYPE_I32                  __m512i
#   define PARALLILOS_ALIGNMENT_I32             64
#   define PARALLILOS_LOADU_I32(data)           _mm512_loadu_si512(data)
#   define PARALLILOS_LOADA_I32(data)           _mm512_load_si512(data)
#   define PARALLILOS_STOREU_I32(addr, data)    _mm512_storeu_si512((void*)addr, data)
#   define PARALLILOS_STOREA_I32(addr, data)    _mm512_store_si512((void*)addr, data)
#   define PARALLILOS_SETVAL_I32(value)         _mm512_set1_epi32(value)
#   define PARALLILOS_SETZERO_I32()             _mm512_setzero_epi32()
#   define PARALLILOS_MUL_I32(a, b)             _mm512_mul_epi32(a, b)
#   define PARALLILOS_ADD_I32(a, b)             _mm512_add_epi32(a, b)
#   define PARALLILOS_SUB_I32(a, b)             _mm512_sub_epi32(a, b)
#   define PARALLILOS_DIV_I32(a, b)             _mm512_cvtps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(a), _mm512_cvtepi32_ps(b)))
#   define PARALLILOS_SQRT_I32(a)               _mm512_cvtps_epi32(_mm512_sqrt_ps(_mm512_cvtepi32_ps(a)))
#   define PARALLILOS_ADDMUL_I32(a, b, c)       _mm512_add_epi32(a, _mm512_mul_epi32(b, c))
#   define PARALLILOS_SUBMUL_I32(a, b, c)       _mm512_sub_epi32(a, _mm512_mul_epi32(b, c))
#   if defined(PARALLILOS_SVML)
#     undef  PARALLILOS_DIV_I32
#     define PARALLILOS_DIV_I32(a, b)           _mm512_div_epi32(a, b)
#   endif
# elif defined(PARALLILOS_AVX2)
#   define PARALLILOS_SET_I32                   "AVX2, AVX"
#   define PARALLILOS_TYPE_I32                  __m256i
#   define PARALLILOS_ALIGNMENT_I32             32
#   define PARALLILOS_LOADU_I32(data)           _mm256_loadu_si256((const __m256i*)data)
#   define PARALLILOS_LOADA_I32(data)           _mm256_load_si256((const __m256i*)data)
#   define PARALLILOS_STOREU_I32(addr, data)    _mm256_storeu_si256 ((__m256i*)addr, data)
#   define PARALLILOS_STOREA_I32(addr, data)    _mm256_store_si256((__m256i*)addr, data)
#   define PARALLILOS_SETVAL_I32(value)         _mm256_set1_epi32(value)
#   define PARALLILOS_SETZERO_I32()             _mm256_setzero_si256()
#   define PARALLILOS_MUL_I32(a, b)             _mm256_mul_epi32(a, b)
#   define PARALLILOS_ADD_I32(a, b)             _mm256_add_epi32(a, b)
#   define PARALLILOS_SUB_I32(a, b)             _mm256_sub_epi32(a, b)
#   define PARALLILOS_DIV_I32(a, b)             _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(a), _mm256_cvtepi32_ps(b)))
#   define PARALLILOS_SQRT_I32(a)               _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(a)))
#   define PARALLILOS_ADDMUL_I32(a, b, c)       _mm256_add_epi32(a, _mm256_mul_epi32(b, c))
#   define PARALLILOS_SUBMUL_I32(a, b, c)       _mm256_sub_epi32(a, _mm256_mul_epi32(b, c))
#   if defined(PARALLILOS_SVML)
#     undef  PARALLILOS_DIV_I32
#     define PARALLILOS_DIV_I32(a, b)           _mm256_div_epi32(a, b)
#   endif
# elif defined(PARALLILOS_SSE4_1)
#   define PARALLILOS_SET_I32                   "SSE4.1, SSE2"
#   define PARALLILOS_TYPE_I32                  __m128i
#   define PARALLILOS_ALIGNMENT_I32             16
#   define PARALLILOS_LOADU_I32(data)           _mm_loadu_si128((const __m128i*)data)
#   define PARALLILOS_LOADA_I32(data)           _mm_load_si128((const __m128i*)data)
#   define PARALLILOS_STOREU_I32(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
#   define PARALLILOS_STOREA_I32(addr, data)    _mm_store_si128((__m128i*)addr, data)
#   define PARALLILOS_SETVAL_I32(value)         _mm_set1_epi32(value)
#   define PARALLILOS_SETZERO_I32()             _mm_setzero_si128()
#   define PARALLILOS_MUL_I32(a, b)             _mm_mul_epi32(a, b)
#   define PARALLILOS_ADD_I32(a, b)             _mm_add_epi32(a, b)
#   define PARALLILOS_SUB_I32(a, b)             _mm_sub_epi32(a, b)
#   define PARALLILOS_DIV_I32(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
#   define PARALLILOS_SQRT_I32(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
#   define PARALLILOS_ADDMUL_I32(a, b, c)       _mm_add_epi32(a, _mm_mul_epi32(b, c))
#   define PARALLILOS_SUBMUL_I32(a, b, c)       _mm_sub_epi32(a, _mm_mul_epi32(b, c))
#   if defined(PARALLILOS_SVML)
#     undef  PARALLILOS_DIV_I32
#     define PARALLILOS_DIV_I32(a, b)           _mm_div_epi32(a, b)
#   endif
# elif defined(PARALLILOS_SSE2)
#   define PARALLILOS_SET_I32                   "SSE2"
#   define PARALLILOS_TYPE_I32                  __m128i
#   define PARALLILOS_ALIGNMENT_I32             16
#   define PARALLILOS_LOADU_I32(data)           _mm_loadu_si128((const __m128i*)data)
#   define PARALLILOS_LOADA_I32(data)           _mm_load_si128((const __m128i*)data)
#   define PARALLILOS_STOREU_I32(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
#   define PARALLILOS_STOREA_I32(addr, data)    _mm_store_si128((__m128i*)addr, data)
#   define PARALLILOS_SETVAL_I32(value)         _mm_set1_epi32(value)
#   define PARALLILOS_SETZERO_I32()             _mm_setzero_si128()
#   define PARALLILOS_MUL_I32(a, b)             _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
#   define PARALLILOS_ADD_I32(a, b)             _mm_add_epi32(a, b)
#   define PARALLILOS_SUB_I32(a, b)             _mm_sub_epi32(a, b)
#   define PARALLILOS_DIV_I32(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
#   define PARALLILOS_SQRT_I32(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
#   define PARALLILOS_ADDMUL_I32(a, b, c)       _mm_add_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
#   define PARALLILOS_SUBMUL_I32(a, b, c)       _mm_sub_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
#   if defined(PARALLILOS_SVML)
#     undef  PARALLILOS_DIV_I32
#     define PARALLILOS_DIV_I32(a, b)           _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
#   endif
# endif
  
# ifdef PARALLILOS_TYPE_I32
    template <>
    struct simd<int32_t>
    {
      using vector_type = PARALLILOS_TYPE_I32;
      static constexpr const char* set  = PARALLILOS_SET_I32;
      static constexpr size_t alignment = PARALLILOS_ALIGNMENT_I32;
      static constexpr size_t size      = sizeof(vector_type) / sizeof(int32_t);

      static constexpr size_t inline passes(const size_t n)
      {
        return n / size;
      }

      static constexpr size_t inline sequential(const size_t n)
      {
        return n - passes(n)*size;
      }

      simd(const size_t n_elements) noexcept :
        passes_left(passes(n_elements))
      {}
      inline size_t operator*() noexcept {return current_index;}
      inline void operator++() noexcept {--passes_left, current_index += size;}
      inline bool operator!=(const simd&) noexcept {return passes_left;}
      inline simd& begin() noexcept {return *this;}
      inline simd end() noexcept {return simd{0};}
      private:
        size_t       passes_left;
        size_t       current_index = 0;
    };

    template<>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_setzero<int32_t>(void)
    {
      return PARALLILOS_SETZERO_I32();
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_loadu(const int32_t* data)
    {
      return PARALLILOS_LOADU_I32(data);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_loada(const int32_t* data)
    {
      return PARALLILOS_LOADA_I32(data);
    }

    template <>
    PARALLILOS_INLINE void simd_storeu(int32_t* addr, PARALLILOS_TYPE_I32 data)
    {
      PARALLILOS_STOREU_I32(addr, data);
    }

    template <>
    PARALLILOS_INLINE void simd_storea(int32_t* addr, PARALLILOS_TYPE_I32 data)
    {
      PARALLILOS_STOREA_I32(addr, data);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_setval(const int32_t value)
    {
      return PARALLILOS_SETVAL_I32(value);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_add(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b)
    {
      return PARALLILOS_ADD_I32(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_mul(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b)
    {
      return PARALLILOS_MUL_I32(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_sub(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b)
    {
      return PARALLILOS_SUB_I32(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_div(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b)
    {
      return PARALLILOS_DIV_I32(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_sqrt(PARALLILOS_TYPE_I32 a)
    {
      return PARALLILOS_SQRT_I32(a);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_addmul(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b, PARALLILOS_TYPE_I32 c)
    {
      return PARALLILOS_ADDMUL_I32(a, b, c);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_submul(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b, PARALLILOS_TYPE_I32 c)
    {
      return PARALLILOS_SUBMUL_I32(a, b, c);
    }
# endif

    //
# undef PARALLILOS_TYPE_I32
# undef PARALLILOS_ALIGNMENT_I32
# undef PARALLILOS_LOADU_I32
# undef PARALLILOS_LOADA_I32
# undef PARALLILOS_STOREU_I32
# undef PARALLILOS_STOREA_I32
# undef PARALLILOS_SETVAL_I32
# undef PARALLILOS_SETZERO_I32
# undef PARALLILOS_ADD_I32
# undef PARALLILOS_MUL_I32
# undef PARALLILOS_SUB_I32
# undef PARALLILOS_DIV_I32
# undef PARALLILOS_DIV_I32
# undef PARALLILOS_SQRT_I32
# undef PARALLILOS_ADDMUL_I32
# undef PARALLILOS_SUBMUL_I32
  }
}
#endif
#endif