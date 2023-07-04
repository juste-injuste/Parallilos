/*---author-----------------------------------------------------------------------------------------

Justin Asselin (juste-injuste)
justin.asselin@usherbrooke.ca
https://github.com/juste-injuste/Parallilos

-----liscence---------------------------------------------------------------------------------------
 
MIT License

Copyright (c) 2023 Justin Asselin (juste-injuste)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-----description------------------------------------------------------------------------------------

Parallilos is a simple and lightweight C++11 (and newer) library that abstracts away SIMD usage to
facilitate generic parallelism.

-----inclusion guard------------------------------------------------------------------------------*/
#ifndef PARALLILOS_HPP
#define PARALLILOS_HPP
// --Parallilos library-----------------------------------------------------------------------------
namespace Parallilos
{
// --Parallilos library: backend--------------------------------------------------------------------
  namespace Backend
  {
    // library version
    #define PARALLILOS_VERSION       000001000L
    #define PARALLILOS_VERSION_MAJOR 0
    #define PARALLILOS_VERSION_MINOR 1
    #define PARALLILOS_VERSION_PATCH 0

    // define the best way to inline given a compiler
    #if defined(__GNUC__)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE_AVX
      #define PARALLILOS_COMPILER_SUPPORTS_NEON
      #define PARALLILOS_INLINE __attribute__((always_inline)) inline
    #elif defined(__clang__)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE_AVX
      #define PARALLILOS_COMPILER_SUPPORTS_NEON
      #define PARALLILOS_INLINE __attribute__((always_inline)) inline
    #elif defined(__MINGW32__)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE_AVX
      #define PARALLILOS_INLINE __attribute__((always_inline)) inline
    #elif defined(__MINGW64__)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE_AVX
      #define PARALLILOS_INLINE __attribute__((always_inline)) inline
    #elif defined(__apple_build_version__)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE_AVX
      #define PARALLILOS_INLINE __attribute__((always_inline)) inline
    #elif defined(_MSC_VER)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE_AVX
      #define PARALLILOS_INLINE __forceinline
    #elif defined(__INTEL_COMPILER)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE_AVX
      #define PARALLILOS_INLINE __forceinline
    #elif defined(__ARMCC_VERSION)
      #define PARALLILOS_COMPILER_SUPPORTS_NEON
      #define PARALLILOS_INLINE __forceinline
    #else
      #warning "warning: Parallilos: your compiler is not supported."
    #endif
  }
}
// --Parallilos library: global level---------------------------------------------------------------
  // define parallelism specifiers and include appropriate headers
  #if defined(__AVX512F__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_AVX512F
    #include <immintrin.h>
  #elif defined(__AVX2__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_AVX2
    #include <immintrin.h>
  #elif defined(__AVX__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_AVX
    #include <immintrin.h>
  #elif defined(__SSE__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_SSE
    #include <immintrin.h>
  #elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(PARALLILOS_COMPILER_SUPPORTS_NEON)
    #define PARALLILOS_NO_PARALLELISM
    #warning "warning: Parallilos: ARM Neon has not been implemented yet."
    //#define PARALLILOS_USE_PARALLELISM
    #ifdef __ARM_ARCH_64
      //#define PARALLILOS_USE_NEON64
      //#include <arm64_neon.h
    #else
      //#define PARALLILOS_USE_NEON
      //#include <arm_neon.h>
    #endif
  #else
    #define PARALLILOS_NO_PARALLELISM
  #endif
// --Parallilos library: backend--------------------------------------------------------------------
namespace Parallilos
{
  namespace Backend
  {
    // define the best SIMD intrinsics to use
    #if defined(PARALLILOS_USE_AVX512F)
      #define PARALLILOS_TYPE_PD      __m512d
      #define PARALLILOS_TYPE_PS      __m512
      #define PARALLILOS_LOADU_PD     _mm512_loadu_pd(data)
      #define PARALLILOS_LOADU_PS     _mm512_loadu_ps(data)
      #define PARALLILOS_STOREU_PD    _mm512_storeu_pd((void*)addr, data)
      #define PARALLILOS_STOREU_PS    _mm512_storeu_ps((void*)addr, data)
      #define PARALLILOS_SET1_PD      _mm512_set1_pd(value)
      #define PARALLILOS_SET1_PS      _mm512_set1_ps(value)
      #define PARALLILOS_SETZERO_PD   _mm512_setzero_pd()
      #define PARALLILOS_SETZERO_PS   _mm512_setzero_ps()
      #define PARALLILOS_MUL_PD       _mm512_mul_pd(A, B)
      #define PARALLILOS_MUL_PS       _mm512_mul_ps(A, B)
      #define PARALLILOS_ADD_PD       _mm512_add_pd(A, B)
      #define PARALLILOS_ADD_PS       _mm512_add_ps(A, B)
      #define PARALLILOS_SUB_PD       _mm512_sub_pd(A, B)
      #define PARALLILOS_SUB_PS       _mm512_sub_ps(A, B)
      #define PARALLILOS_DIV_PD       _mm512_div_pd(A, B)
      #define PARALLILOS_DIV_PS       _mm512_div_ps(A, B)
      #define PARALLILOS_SQRT_PD      _mm512_sqrt_pd(A)
      #define PARALLILOS_SQRT_PS      _mm512_sqrt_ps(A)
      #define PARALLILOS_MULADD_PD    _mm512_fmadd_pd(A, B, C)
      #define PARALLILOS_MULADD_PS    _mm512_fmadd_ps(A, B, C)
      #define PARALLILOS_NEGMULADD_PD _mm512_fnmadd_pd(A, B, C)
      #define PARALLILOS_NEGMULADD_PS _mm512_fnmadd_ps(A, B, C)
    #elif defined(PARALLILOS_USE_AVX2) || defined(PARALLILOS_USE_AVX)
      #define PARALLILOS_TYPE_PD      __m256d
      #define PARALLILOS_TYPE_PS      __m256
      #define PARALLILOS_LOADU_PD     _mm256_loadu_pd(data)
      #define PARALLILOS_LOADU_PS     _mm256_loadu_ps(data)
      #define PARALLILOS_STOREU_PD    _mm256_storeu_pd(addr, data)
      #define PARALLILOS_STOREU_PS    _mm256_storeu_ps(addr, data)
      #define PARALLILOS_SET1_PD      _mm256_set1_pd(value)
      #define PARALLILOS_SET1_PS      _mm256_set1_ps(value)
      #define PARALLILOS_SETZERO_PD   _mm256_setzero_pd()
      #define PARALLILOS_SETZERO_PS   _mm256_setzero_ps()
      #define PARALLILOS_MUL_PD       _mm256_mul_pd(A, B)
      #define PARALLILOS_MUL_PS       _mm256_mul_ps(A, B)
      #define PARALLILOS_ADD_PD       _mm256_add_pd(A, B)
      #define PARALLILOS_ADD_PS       _mm256_add_ps(A, B)
      #define PARALLILOS_SUB_PD       _mm256_sub_pd(A, B)
      #define PARALLILOS_SUB_PS       _mm256_sub_ps(A, B)
      #define PARALLILOS_DIV_PD       _mm256_div_pd(A, B)
      #define PARALLILOS_DIV_PS       _mm256_div_ps(A, B)
      #define PARALLILOS_SQRT_PD      _mm256_sqrt_pd(A)
      #define PARALLILOS_SQRT_PS      _mm256_sqrt_ps(A)
      #define PARALLILOS_MULADD_PD    _mm256_fmadd_pd(A, B, C)
      #define PARALLILOS_MULADD_PS    _mm256_fmadd_ps(A, B, C)
      #define PARALLILOS_NEGMULADD_PD _mm256_fnmadd_pd(A, B, C)
      #define PARALLILOS_NEGMULADD_PS _mm256_fnmadd_ps(A, B, C)
    #elif defined(PARALLILOS_USE_SSE)
      #define PARALLILOS_TYPE_PD      __m128d
      #define PARALLILOS_TYPE_PS      __m128
      #define PARALLILOS_LOADU_PD     _mm_loadu_pd(data)
      #define PARALLILOS_LOADU_PS     _mm_loadu_ps(data)
      #define PARALLILOS_STOREU_PD    _mm_storeu_pd(addr, data)
      #define PARALLILOS_STOREU_PS    _mm_storeu_ps(addr, data)
      #define PARALLILOS_SET1_PD      _mm_set1_pd(value)
      #define PARALLILOS_SET1_PS      _mm_set1_ps(value)
      #define PARALLILOS_SETZERO_PD   _mm_setzero_pd()
      #define PARALLILOS_SETZERO_PS   _mm_setzero_ps()
      #define PARALLILOS_MUL_PD       _mm_mul_pd(A, B)
      #define PARALLILOS_MUL_PS       _mm_mul_ps(A, B)
      #define PARALLILOS_ADD_PD       _mm_add_pd(A, B)
      #define PARALLILOS_ADD_PS       _mm_add_ps(A, B)
      #define PARALLILOS_SUB_PD       _mm_sub_pd(A, B)
      #define PARALLILOS_SUB_PS       _mm_sub_ps(A, B)
      #define PARALLILOS_DIV_PD       _mm_div_pd(A, B)
      #define PARALLILOS_DIV_PS       _mm_div_ps(A, B)
      #define PARALLILOS_SQRT_PD      _mm_sqrt_pd(A)
      #define PARALLILOS_SQRT_PS      _mm_sqrt_ps(A)
      #define PARALLILOS_MULADD_PD    _mm_add_pd(_mm_mul_pd(A, B), C)
      #define PARALLILOS_MULADD_PS    _mm_add_ps(_mm_mul_ps(A, B), C)
      #define PARALLILOS_NEGMULADD_PD _mm_sub_pd(C, _mm_mul_pd(A, B))
      #define PARALLILOS_NEGMULADD_PS _mm_sub_ps(C, _mm_mul_ps(A, B))
    #endif
  }
// --Parallilos library: frontend-------------------------------------------------------------------
  inline namespace Frontend
  {
    // define the used SIMD instruction set
    #if defined(PARALLILOS_USE_AVX512F)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "AVX512"
    #elif defined(PARALLILOS_USE_AVX2)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "AVX2"
    #elif defined(PARALLILOS_USE_AVX)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "AVX"
    #elif defined(PARALLILOS_USE_SSE)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "SSE"
    #else
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "no SIMD set available"
    #endif

    // define a standard API to use SIMD intrinsics
    #ifdef PARALLILOS_USE_PARALLELISM
      template<typename T>
      struct simd_get_type;

      template<>
      struct simd_get_type<double> {
        using type = PARALLILOS_TYPE_PD;
      };

      template<>
      struct simd_get_type<float> {
        using type = PARALLILOS_TYPE_PS;
      };

      template<typename T>
      PARALLILOS_INLINE T simd_setzero();

      template<>
      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_setzero()
      {
        return PARALLILOS_SETZERO_PD;
      }

      template<>
      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_setzero()
      {
        return PARALLILOS_SETZERO_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_loadu(const double* data)
      {
        return PARALLILOS_LOADU_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_loadu(const float* data)
      {
        return PARALLILOS_LOADU_PS;
      }
        
      PARALLILOS_INLINE void simd_storeu(double* addr, PARALLILOS_TYPE_PD data)
      {
        PARALLILOS_STOREU_PD;
      }

      PARALLILOS_INLINE void simd_storeu(float* addr, PARALLILOS_TYPE_PS data)
      {
        PARALLILOS_STOREU_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_set1(const double value)
      {
        return PARALLILOS_SET1_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_set1(const float value)
      {
        return PARALLILOS_SET1_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_mul(PARALLILOS_TYPE_PD A, PARALLILOS_TYPE_PD B)
      {
        return PARALLILOS_MUL_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_mul(PARALLILOS_TYPE_PS A, PARALLILOS_TYPE_PS B)
      {
        return PARALLILOS_MUL_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_add(PARALLILOS_TYPE_PD A, PARALLILOS_TYPE_PD B)
      {
        return PARALLILOS_ADD_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_add(PARALLILOS_TYPE_PS A, PARALLILOS_TYPE_PS B)
      {
        return PARALLILOS_ADD_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_sub(PARALLILOS_TYPE_PD A, PARALLILOS_TYPE_PD B)
      {
        return PARALLILOS_SUB_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_sub(PARALLILOS_TYPE_PS A, PARALLILOS_TYPE_PS B)
      {
        return PARALLILOS_SUB_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_div(PARALLILOS_TYPE_PD A, PARALLILOS_TYPE_PD B)
      {
        return PARALLILOS_DIV_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_div(PARALLILOS_TYPE_PS A, PARALLILOS_TYPE_PS B)
      {
        return PARALLILOS_DIV_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_sqrt(PARALLILOS_TYPE_PD A)
      {
        return PARALLILOS_SQRT_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_sqrt(PARALLILOS_TYPE_PS A)
      {
        return PARALLILOS_SQRT_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_muladd(PARALLILOS_TYPE_PD A, PARALLILOS_TYPE_PD B, PARALLILOS_TYPE_PD C)
      {
        return PARALLILOS_MULADD_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_muladd(PARALLILOS_TYPE_PS A, PARALLILOS_TYPE_PS B, PARALLILOS_TYPE_PS C)
      {
        return PARALLILOS_MULADD_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_nmuladd(PARALLILOS_TYPE_PD A, PARALLILOS_TYPE_PD B, PARALLILOS_TYPE_PD C)
      {
        return PARALLILOS_NEGMULADD_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_nmuladd(PARALLILOS_TYPE_PS A, PARALLILOS_TYPE_PS B, PARALLILOS_TYPE_PS C)
      {
        return PARALLILOS_NEGMULADD_PS;
      }
    #endif
  }
// --Parallilos library: backend--------------------------------------------------------------------
  namespace Backend
  {
    // cleanup namespace
    #undef PARALLILOS_INLINE
    #undef PARALLILOS_TYPE_PD
    #undef PARALLILOS_TYPE_PS
    #undef PARALLILOS_LOADU_PD
    #undef PARALLILOS_LOADU_PS
    #undef PARALLILOS_STOREU_PD
    #undef PARALLILOS_STOREU_PS
    #undef PARALLILOS_SET1_PD
    #undef PARALLILOS_SET1_PS
    #undef PARALLILOS_SETZERO_PD
    #undef PARALLILOS_SETZERO_PS
    #undef PARALLILOS_MUL_PD
    #undef PARALLILOS_MUL_PS
    #undef PARALLILOS_ADD_PD
    #undef PARALLILOS_ADD_PS
    #undef PARALLILOS_SUB_PD
    #undef PARALLILOS_SUB_PS
    #undef PARALLILOS_DIV_PD
    #undef PARALLILOS_DIV_PS
    #undef PARALLILOS_SQRT_PD
    #undef PARALLILOS_SQRT_PS
    #undef PARALLILOS_MULADD_PD
    #undef PARALLILOS_MULADD_PS
    #undef PARALLILOS_NEGMULADD_PD
    #undef PARALLILOS_NEGMULADD_PS
    #undef PARALLILOS_COMPILER_SUPPORTS_SSE_AVX
    #undef PARALLILOS_COMPILER_SUPPORTS_NEON
  }
}
#endif