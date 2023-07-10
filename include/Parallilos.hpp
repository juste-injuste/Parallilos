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
// --necessary standard libraries-------------------------------------------------------------------
#include <cstddef> // for size_t
#include <cmath>   // for std::sqrt
#include <cstdlib> // for std::malloc or std::aligned_malloc (if c++17 and newer)
#define __AVX512F__
// --Parallilos library-----------------------------------------------------------------------------
namespace Parallilos
{
// --Parallilos library: frontend forward declarations----------------------------------------------
  inline namespace Frontend
  {
    // library version
    #define PARALLILOS_VERSION       000001000L
    #define PARALLILOS_VERSION_MAJOR 0
    #define PARALLILOS_VERSION_MINOR 1
    #define PARALLILOS_VERSION_PATCH 0

    // [r] = [a] + [b]
    template<typename T>
    inline T* add_arrays(const T* a, const T* b, T* r, const size_t n);

    // [r] = a + [b]
    template<typename T>
    inline T* add_arrays(const T a, const T* b, T* r, const size_t n);

    // [r] = [a] + b
    template<typename T>
    inline T* add_arrays(const T* a, const T b, T* r, const size_t n);

    // [r] = [a] - [b]
    template<typename T>
    inline T* sub_arrays(const T* a, const T* b, T* r, const size_t n);

    // [r] = a - [b]
    template<typename T>
    inline T* sub_arrays(const T a, const T* b, T* r, const size_t n);

    // [r] = [a] - b
    template<typename T>
    inline T* sub_arrays(const T* a, const T b, T* r, const size_t n);

    // [r] = [a] * [b]
    template<typename T>
    inline T* mul_arrays(const T* a, const T* b, T* r, const size_t n);

    // [r] = a * [b]
    template<typename T>
    inline T* mul_arrays(const T a, const T* b, T* r, const size_t n);

    // [r] = [a] * b
    template<typename T>
    inline T* mul_arrays(const T* a, const T b, T* r, const size_t n);

    // [r] = [a] / [b]
    template<typename T>
    inline T* div_arrays(const T* a, const T* b, T* r, const size_t n);

    // [r] = a / [b]
    template<typename T>
    inline T* div_arrays(const T a, const T* b, T* r, const size_t n);

    // [r] = [a] / b
    template<typename T>
    inline T* div_arrays(const T* a, const T b, T* r, const size_t n);

    // [r] = sqrt([a])
    template<typename T>
    inline T* sqrt_array(const T* a, T* r, const size_t n);
  }
// --Parallilos library: backend--------------------------------------------------------------------
  namespace Backend
  {
    // define which instruction set is supported
    // define the best way to inline given the compiler
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
      #define PARALLILOS_INLINE inline
      #if __cplusplus >= 202302L
        #warning "warning: Parallilos: your compiler is not supported."
      #endif
    #endif
  }
}
// --Parallilos library: global level---------------------------------------------------------------
  // define parallelism specifiers and include appropriate headers
  #if defined(__AVX512F__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_AVX512F
    #define PARALLILOS_INTEL
    #include <immintrin.h>
  #elif defined(__AVX2__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_AVX2
    #define PARALLILOS_INTEL
    #include <immintrin.h>
  #elif defined(__AVX__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_AVX
    #define PARALLILOS_INTEL
    #include <immintrin.h>
  #elif defined(__SSE__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_SSE
    #define PARALLILOS_INTEL
    #include <immintrin.h>
  #elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(PARALLILOS_COMPILER_SUPPORTS_NEON)
    #define PARALLILOS_NO_PARALLELISM
    #if __cplusplus >= 202302L
      #warning "warning: Parallilos: ARM Neon has not been implemented yet."
    #endif
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
    #if __cplusplus >= 202302L
      #warning "warning: Parallilos: no SIMD instruction set used, sequential fallback used."
    #endif
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
      #define PARALLILOS_LOAD_PD      _mm512_load_pd(data)
      #define PARALLILOS_LOAD_PS      _mm512_load_ps(data)
      #define PARALLILOS_STOREU_PD    _mm512_storeu_pd((void*)addr, data)
      #define PARALLILOS_STOREU_PS    _mm512_storeu_ps((void*)addr, data)
      #define PARALLILOS_SET1_PD      _mm512_set1_pd(value)
      #define PARALLILOS_SET1_PS      _mm512_set1_ps(value)
      #define PARALLILOS_SETZERO_PD   _mm512_setzero_pd()
      #define PARALLILOS_SETZERO_PS   _mm512_setzero_ps()
      #define PARALLILOS_MUL_PD       _mm512_mul_pd(a, b)
      #define PARALLILOS_MUL_PS       _mm512_mul_ps(a, b)
      #define PARALLILOS_ADD_PD       _mm512_add_pd(a, b)
      #define PARALLILOS_ADD_PS       _mm512_add_ps(a, b)
      #define PARALLILOS_SUB_PD       _mm512_sub_pd(a, b)
      #define PARALLILOS_SUB_PS       _mm512_sub_ps(a, b)
      #define PARALLILOS_DIV_PD       _mm512_div_pd(a, b)
      #define PARALLILOS_DIV_PS       _mm512_div_ps(a, b)
      #define PARALLILOS_SQRT_PD      _mm512_sqrt_pd(a)
      #define PARALLILOS_SQRT_PS      _mm512_sqrt_ps(a)
      #define PARALLILOS_MULADD_PD    _mm512_fmadd_pd(a, b, c)
      #define PARALLILOS_MULADD_PS    _mm512_fmadd_ps(a, b, c)
      #define PARALLILOS_NEGMULADD_PD _mm512_fnmadd_pd(a, b, c)
      #define PARALLILOS_NEGMULADD_PS _mm512_fnmadd_ps(a, b, c)
    #elif defined(PARALLILOS_USE_AVX2) || defined(PARALLILOS_USE_AVX)
      #define PARALLILOS_TYPE_PD      __m256d
      #define PARALLILOS_TYPE_PS      __m256
      #define PARALLILOS_LOADU_PD     _mm256_loadu_pd(data)
      #define PARALLILOS_LOADU_PS     _mm256_loadu_ps(data)
      #define PARALLILOS_LOAD_PD      _mm256_load_pd(data)
      #define PARALLILOS_LOAD_PS      _mm256_load_ps(data)
      #define PARALLILOS_STOREU_PD    _mm256_storeu_pd(addr, data)
      #define PARALLILOS_STOREU_PS    _mm256_storeu_ps(addr, data)
      #define PARALLILOS_SET1_PD      _mm256_set1_pd(value)
      #define PARALLILOS_SET1_PS      _mm256_set1_ps(value)
      #define PARALLILOS_SETZERO_PD   _mm256_setzero_pd()
      #define PARALLILOS_SETZERO_PS   _mm256_setzero_ps()
      #define PARALLILOS_MUL_PD       _mm256_mul_pd(a, b)
      #define PARALLILOS_MUL_PS       _mm256_mul_ps(a, b)
      #define PARALLILOS_ADD_PD       _mm256_add_pd(a, b)
      #define PARALLILOS_ADD_PS       _mm256_add_ps(a, b)
      #define PARALLILOS_SUB_PD       _mm256_sub_pd(a, b)
      #define PARALLILOS_SUB_PS       _mm256_sub_ps(a, b)
      #define PARALLILOS_DIV_PD       _mm256_div_pd(a, b)
      #define PARALLILOS_DIV_PS       _mm256_div_ps(a, b)
      #define PARALLILOS_SQRT_PD      _mm256_sqrt_pd(a)
      #define PARALLILOS_SQRT_PS      _mm256_sqrt_ps(a)
      #define PARALLILOS_MULADD_PD    _mm256_fmadd_pd(a, b, c)
      #define PARALLILOS_MULADD_PS    _mm256_fmadd_ps(a, b, c)
      #define PARALLILOS_NEGMULADD_PD _mm256_fnmadd_pd(a, b, c)
      #define PARALLILOS_NEGMULADD_PS _mm256_fnmadd_ps(a, b, c)
    #elif defined(PARALLILOS_USE_SSE)
      #define PARALLILOS_TYPE_PD      __m128d
      #define PARALLILOS_TYPE_PS      __m128
      #define PARALLILOS_LOADU_PD     _mm_loadu_pd(data)
      #define PARALLILOS_LOADU_PS     _mm_loadu_ps(data)
      #define PARALLILOS_LOAD_PD      _mm_load_pd(data)
      #define PARALLILOS_LOAD_PS      _mm_load_ps(data)
      #define PARALLILOS_STOREU_PD    _mm_storeu_pd(addr, data)
      #define PARALLILOS_STOREU_PS    _mm_storeu_ps(addr, data)
      #define PARALLILOS_SET1_PD      _mm_set1_pd(value)
      #define PARALLILOS_SET1_PS      _mm_set1_ps(value)
      #define PARALLILOS_SETZERO_PD   _mm_setzero_pd()
      #define PARALLILOS_SETZERO_PS   _mm_setzero_ps()
      #define PARALLILOS_MUL_PD       _mm_mul_pd(a, b)
      #define PARALLILOS_MUL_PS       _mm_mul_ps(a, b)
      #define PARALLILOS_ADD_PD       _mm_add_pd(a, b)
      #define PARALLILOS_ADD_PS       _mm_add_ps(a, b)
      #define PARALLILOS_SUB_PD       _mm_sub_pd(a, b)
      #define PARALLILOS_SUB_PS       _mm_sub_ps(a, b)
      #define PARALLILOS_DIV_PD       _mm_div_pd(a, b)
      #define PARALLILOS_DIV_PS       _mm_div_ps(a, b)
      #define PARALLILOS_SQRT_PD      _mm_sqrt_pd(a)
      #define PARALLILOS_SQRT_PS      _mm_sqrt_ps(a)
      #define PARALLILOS_MULADD_PD    _mm_add_pd(_mm_mul_pd(a, b), c)
      #define PARALLILOS_MULADD_PS    _mm_add_ps(_mm_mul_ps(a, b), c)
      #define PARALLILOS_NEGMULADD_PD _mm_sub_pd(c, _mm_mul_pd(a, b))
      #define PARALLILOS_NEGMULADD_PS _mm_sub_ps(c, _mm_mul_ps(a, b))
    #endif
  }
// --Parallilos library: frontend-------------------------------------------------------------------
  inline namespace Frontend
  {
    // define the used SIMD instruction set and the best SIMD alignment
    #if defined(PARALLILOS_USE_AVX512F)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "AVX512"
      #define PARALLILOS_ALIGNMENT    size_t(64)
    #elif defined(PARALLILOS_USE_AVX2)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "AVX2"
      #define PARALLILOS_ALIGNMENT    size_t(32)
    #elif defined(PARALLILOS_USE_AVX)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "AVX"
      #define PARALLILOS_ALIGNMENT    size_t(32)
    #elif defined(PARALLILOS_USE_SSE)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "SSE"
      #define PARALLILOS_ALIGNMENT    size_t(16)
    #else
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "no SIMD set available"
      #define PARALLILOS_ALIGNMENT    size_t(0)
    #endif

    // define a standard API to use SIMD intrinsics
    #ifdef PARALLILOS_USE_PARALLELISM
      template<typename T>
      struct simd_properties;

      template<>
      struct simd_properties<double> {
        using type = PARALLILOS_TYPE_PD;
        static constexpr size_t size = sizeof(type) / sizeof(double);
        static size_t iterations(const size_t n) {
          return n / size;
        }
      };

      template <>
      struct simd_properties<float> {
          using type = PARALLILOS_TYPE_PS;
          static constexpr size_t size = sizeof(type) / sizeof(float);
          static size_t iterations(const size_t n) {
              return n / size;
          }
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

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_load(const double* data)
      {
        return PARALLILOS_LOAD_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_load(const float* data)
      {
        return PARALLILOS_LOAD_PS;
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

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_mul(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b)
      {
        return PARALLILOS_MUL_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_mul(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b)
      {
        return PARALLILOS_MUL_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_add(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b)
      {
        return PARALLILOS_ADD_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_add(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b)
      {
        return PARALLILOS_ADD_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_sub(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b)
      {
        return PARALLILOS_SUB_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_sub(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b)
      {
        return PARALLILOS_SUB_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_div(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b)
      {
        return PARALLILOS_DIV_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_div(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b)
      {
        return PARALLILOS_DIV_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_sqrt(PARALLILOS_TYPE_PD a)
      {
        return PARALLILOS_SQRT_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_sqrt(PARALLILOS_TYPE_PS a)
      {
        return PARALLILOS_SQRT_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_muladd(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b, PARALLILOS_TYPE_PD c)
      {
        return PARALLILOS_MULADD_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_muladd(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b, PARALLILOS_TYPE_PS c)
      {
        return PARALLILOS_MULADD_PS;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_nmuladd(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b, PARALLILOS_TYPE_PD c)
      {
        return PARALLILOS_NEGMULADD_PD;
      }

      PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_nmuladd(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b, PARALLILOS_TYPE_PS c)
      {
        return PARALLILOS_NEGMULADD_PS;
      }
    #endif

    void* allocate(const size_t size_in_bytes)
    {
      const size_t alignment = PARALLILOS_ALIGNMENT;
      // size_in_bytes must be greater than zero
      if (size_in_bytes == 0) {
        return nullptr;
      }

      void* malloc_ptr;
      void* aligned_ptr;
          
      malloc_ptr = std::malloc(size_in_bytes + PARALLILOS_ALIGNMENT);
      if (!malloc_ptr)
        return ((void *) 0);

      // Align  We have at least sizeof(size_t) space below malloc'd ptr.
      aligned_ptr = (void*)(size_t(malloc_ptr + PARALLILOS_ALIGNMENT) & ~(PARALLILOS_ALIGNMENT - 1));

      // Store the original pointer just before p
      ((void**)aligned_ptr)[-1] = malloc_ptr;

      return aligned_ptr;
    }

    void deallocate(void* addr)
    {
      if (addr)
        free(((void**)addr)[-1]);
    }

    template<typename T>
    T* add_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_add(simd_loadu(a+k), simd_loadu(b+k)));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a[k] + b[k];
      }

      return r;
    }

    template<typename T>
    T* add_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const typename simd_properties<T>::type a_vector = simd_set1(a);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_add(a_vector, simd_loadu(b+k)));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a + b[k];
      }

      return r;
    }

    template<typename T>
    T* add_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const typename simd_properties<T>::type b_vector = simd_set1(b);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_add(simd_loadu(a+k), b_vector));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a[k] + b;
      }

      return r;
    }

    template<typename T>
    T* sub_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_sub(simd_loadu(a+k), simd_loadu(b+k)));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a[k] - b[k];
      }

      return r;
    }

    template<typename T>
    T* sub_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const typename simd_properties<T>::type a_vector = simd_set1(a);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_sub(a_vector, simd_loadu(b+k)));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a - b[k];
      }

      return r;
    }

    template<typename T>
    T* sub_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const typename simd_properties<T>::type b_vector = simd_set1(b);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_sub(simd_loadu(a+k), b_vector));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a[k] - b;
      }

      return r;
    }

    template<typename T>
    T* mul_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_mul(simd_loadu(a+k), simd_loadu(b+k)));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a[k] * b[k];
      }

      return r;
    }

    template<typename T>
    T* mul_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const typename simd_properties<T>::type a_vector = simd_set1(a);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_mul(a_vector, simd_loadu(b+k)));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a * b[k];
      }

      return r;
    }

    template<typename T>
    T* mul_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const typename simd_properties<T>::type b_vector = simd_set1(b);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_mul(simd_loadu(a+k), b_vector));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a[k] * b;
      }

      return r;
    }

    template<typename T>
    T* div_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_div(simd_loadu(a+k), simd_loadu(b+k)));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a[k] / b[k];
      }

      return r;
    }

    template<typename T>
    T* div_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const typename simd_properties<T>::type a_vector = simd_set1(a);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_div(a_vector, simd_loadu(b+k)));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a / b[k];
      }

      return r;
    }

    template<typename T>
    T* div_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const typename simd_properties<T>::type b_vector = simd_set1(b);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_div(simd_loadu(a+k), b_vector));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a[k] / b;
      }

      return r;
    }

    template<typename T>
    T* sqrt_array(const T* a, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_sqrt(simd_loadu(a+k)));
        }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = std::sqrt(a[k]);
      }

      return r;
    }
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