/*---author-----------------------------------------------------------------------------------------

Justin Asselin (juste-injuste)
justin.asselin@usherbrooke.ca
https://github.com/juste-injuste/Parallilos

-----licence----------------------------------------------------------------------------------------
 
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
#include <cstdlib> // for std::malloc, std::free
#include <cstdint>
#include <type_traits>
#include <limits>
#include <memory>
/*
#define __AVX512f__
#define __AVX2__
#define __AVX__
#define __SSSE3__
#define __SSE3__
#define __SSE2__
#define __SSE__
//*/
// --Parallilos library-----------------------------------------------------------------------------
namespace Parallilos
{
  namespace Version
  {
    // library version
    constexpr long NUMBER = 000001000;
    constexpr long MAJOR  = 000      ;
    constexpr long MINOR  =    001   ;
    constexpr long PATCH  =       000;
  }
// --Parallilos library: frontend forward declarations----------------------------------------------
  inline namespace Frontend
  {

    // custom deleter which invokes deallocate
    struct deleter;

    // SIMD aligned memory allocation
    template<typename T>
    T* allocate(const size_t number_of_elements);

    // SIMD aligned memory deallocation
    template<typename T>
    void deallocate(T* addr);

    // check if an address is aligned for SIMD
    template<typename T>
    bool is_aligned(const T* addr);

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

    // define which instruction set is supported and the best way to inline given the compiler
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
      #if __cplusplus >= 202302L
        #warning "warning: Parallilos: your compiler is not supported."
      #endif
      #define PARALLILOS_INLINE inline
    #endif

    template<typename T>
    struct simd_properties {
      using type = T;
      static constexpr const char* set = "no SIMD instruction set used";
      static constexpr size_t alignment = 0;
      static constexpr size_t size = 1;
      static constexpr size_t inline iterations(const size_t) { return 0; }
    };

    // load a vector from unaligned data
    template<typename T>
    PARALLILOS_INLINE typename simd_properties<T>::type simd_loadu(const T* data)
    {
      std::cout << "non SIMD loadu" << std::endl; 
      return *data;
    }

    // load a vector from aligned data
    template<typename T>
    PARALLILOS_INLINE typename simd_properties<T>::type simd_loada(const T* data)
    {
      std::cout << "non SIMD loada" << std::endl; 
      return *data;
    }

    // store a vector into unaligned memory
    template<typename T>
    PARALLILOS_INLINE void simd_storeu(T* addr, typename simd_properties<T>::type data)
    {
      std::cout << "non SIMD storeu" << std::endl; 
      *addr = data;
    }

    // store a vector into aligned memory
    template<typename T>
    PARALLILOS_INLINE void simd_storea(T* addr, typename simd_properties<T>::type data)
    {
      std::cout << "non SIMD storea" << std::endl; 
      *addr = data;
    }

    // load a vector with zeros
    template<typename T, typename V>
    PARALLILOS_INLINE V simd_setzero(void)
    {
      std::cout << "non SIMD setzero" << std::endl; 
      return 0;
    }

    // load a vector with a specific value
    template<typename T>
    PARALLILOS_INLINE typename simd_properties<T>::type simd_setval(const T value)
    {
      std::cout << "non SIMD setval" << std::endl; 
      return value;
    }
    
    // [a] + [b]
    template<typename V>
    PARALLILOS_INLINE V simd_add(V a, V b)
    {
      std::cout << "non SIMD add" << std::endl; 
      return a + b;
    }

    // [a] * [b]
    template<typename V>
    PARALLILOS_INLINE V simd_mul(V a, V b)
    {
      std::cout << "non SIMD mul" << std::endl; 
      return a * b;
    }

    // [a] - [b]
    template<typename V>
    PARALLILOS_INLINE V simd_sub(V a, V b)
    {
      std::cout << "non SIMD sub" << std::endl; 
      return a + b;
    }
    
    // [a] / [b]
    template<typename V>
    PARALLILOS_INLINE V simd_div(V a, V b)
    {
      std::cout << "non SIMD div" << std::endl; 
      return a / b;
    }
    
    // sqrt([a])
    template<typename V>
    PARALLILOS_INLINE V simd_sqrt(V a)
    { 
      std::cout << "non SIMD sqrt" << std::endl; 
      return std::sqrt(a);
    }
    // [a] + ([b] * [c]) 
    template<typename V>
    PARALLILOS_INLINE V simd_addmul(V a, V b, V c)
    {
      std::cout << "non SIMD addmul" << std::endl; 
      return a + b * c;
    }

    // [a] - ([b] * [c])
    template<typename V>
    PARALLILOS_INLINE V simd_submul(V a, V b, V c)
    {
      std::cout << "non SIMD submul" << std::endl; 
      return a - b * c;
    }
  }
}
// --Parallilos library: global level---------------------------------------------------------------
  // define parallelism specifiers and include appropriate headers
  #if defined(__AVX512F__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_PARALLELISM
    #define PARALLILOS_AVX512
    #include <immintrin.h>
  #endif
  #if defined(__AVX2__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_PARALLELISM
    #define PARALLILOS_AVX2
    #include <immintrin.h>
  #endif
  #if defined(__AVX__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_PARALLELISM
    #define PARALLILOS_AVX
    #include <immintrin.h>
  #endif
  #if defined(__SSE4_2__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_PARALLELISM
    #define PARALLILOS_SSE4_2
    #include <immintrin.h>
  #endif
  #if defined(__SSE4_1__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_PARALLELISM
    #define PARALLILOS_SSE4_1
    #include <immintrin.h>
  #endif
  #if defined(__SSSE3__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_PARALLELISM
    #define PARALLILOS_SSSE3
    #include <immintrin.h>
  #endif
  #if defined(__SSE3__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_PARALLELISM
    #define PARALLILOS_SSE3
    #include <immintrin.h>
  #endif
  #if defined(__SSE2__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_PARALLELISM
    #define PARALLILOS_SSE2
    #include <immintrin.h>
  #endif
  #if defined(__SSE__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_PARALLELISM
    #define PARALLILOS_SSE
    #include <immintrin.h>
  #endif
  #if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(PARALLILOS_COMPILER_SUPPORTS_NEON)
    #define PARALLILOS_PARALLELISM
    #ifdef __ARM_ARCH_64
      #define PARALLILOS_NEON64
      #include <arm64_neon.h
    #else
      #define PARALLILOS_NEON
      #include <arm_neon.h>
    #endif
  #endif
  #if !(defined(PARALLILOS_AVX512) \
     || defined(PARALLILOS_AVX2)   \
     || defined(PARALLILOS_AVX)    \
     || defined(PARALLILOS_SSE4_2) \
     || defined(PARALLILOS_SSE4_1) \
     || defined(PARALLILOS_SSSE3)  \
     || defined(PARALLILOS_SSE3)   \
     || defined(PARALLILOS_SSE2)   \
     || defined(PARALLILOS_SSE)    \
     || defined(PARALLILOS_NEON64) \
     || defined(PARALLILOS_NEON))
    #define PARALLILOS_SEQUENTIAL
    #if __cplusplus >= 202302L
      #warning "warning: Parallilos: no SIMD instruction set used, sequential fallback used."
    #endif
  #endif
// --Parallilos library: backend--------------------------------------------------------------------
namespace Parallilos
{
// --Parallilos library: frontend-------------------------------------------------------------------
  inline namespace Frontend
  {
    // define the best SIMD intrinsics to use for 32 bit floating numbers
    #if defined(PARALLILOS_AVX512)
      #define PARALLILOS_SET_F32       "AVX512"
      #define PARALLILOS_TYPE_F32      __m512
      #define PARALLILOS_ALIGNMENT_F32 64
      #define PARALLILOS_LOADU_F32     _mm512_loadu_ps(data)
      #define PARALLILOS_LOADA_F32     _mm512_load_ps(data)
      #define PARALLILOS_STOREU_F32    _mm512_storeu_ps((void*)addr, data)
      #define PARALLILOS_STOREA_F32    _mm512_store_ps((void*)addr, data)
      #define PARALLILOS_SETVAL_F32    _mm512_set1_ps(value)
      #define PARALLILOS_SETZERO_F32   _mm512_setzero_ps()
      #define PARALLILOS_MUL_F32       _mm512_mul_ps(a, b)
      #define PARALLILOS_ADD_F32       _mm512_add_ps(a, b)
      #define PARALLILOS_SUB_F32       _mm512_sub_ps(a, b)
      #define PARALLILOS_DIV_F32       _mm512_div_ps(a, b)
      #define PARALLILOS_SQRT_F32      _mm512_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_F32    _mm512_fmadd_ps(b, c, a)
      #define PARALLILOS_SUBMUL_F32    _mm512_fnmadd_ps(a, b, c)
    #elif defined(PARALLILOS_AVX2)
      #define PARALLILOS_SET_F32       "AVX2"
      #define PARALLILOS_TYPE_F32      __m256
      #define PARALLILOS_ALIGNMENT_F32 32
      #define PARALLILOS_LOADU_F32     _mm256_loadu_ps(data)
      #define PARALLILOS_LOADA_F32     _mm256_load_ps(data)
      #define PARALLILOS_STOREU_F32    _mm256_storeu_ps(addr, data)
      #define PARALLILOS_STOREA_F32    _mm256_store_ps(addr, data)
      #define PARALLILOS_SETVAL_F32    _mm256_set1_ps(value)
      #define PARALLILOS_SETZERO_F32   _mm256_setzero_ps()
      #define PARALLILOS_MUL_F32       _mm256_mul_ps(a, b)
      #define PARALLILOS_ADD_F32       _mm256_add_ps(a, b)
      #define PARALLILOS_SUB_F32       _mm256_sub_ps(a, b)
      #define PARALLILOS_DIV_F32       _mm256_div_ps(a, b)
      #define PARALLILOS_SQRT_F32      _mm256_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_F32    _mm256_fmadd_ps(b, c, a)
      #define PARALLILOS_SUBMUL_F32    _mm256_fnmadd_ps(a, b, c)
    #elif defined(PARALLILOS_AVX)
      #define PARALLILOS_SET_F32       "AVX"
      #define PARALLILOS_TYPE_F32      __m256
      #define PARALLILOS_ALIGNMENT_F32 32
      #define PARALLILOS_LOADU_F32     _mm256_loadu_ps(data)
      #define PARALLILOS_LOADA_F32     _mm256_load_ps(data)
      #define PARALLILOS_STOREU_F32    _mm256_storeu_ps(addr, data)
      #define PARALLILOS_STOREA_F32    _mm256_store_ps(addr, data)
      #define PARALLILOS_SETVAL_F32    _mm256_set1_ps(value)
      #define PARALLILOS_SETZERO_F32   _mm256_setzero_ps()
      #define PARALLILOS_MUL_F32       _mm256_mul_ps(a, b)
      #define PARALLILOS_ADD_F32       _mm256_add_ps(a, b)
      #define PARALLILOS_SUB_F32       _mm256_sub_ps(a, b)
      #define PARALLILOS_DIV_F32       _mm256_div_ps(a, b)
      #define PARALLILOS_SQRT_F32      _mm256_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_F32    _mm256_add_ps(a, _mm256_mul_ps(b, c))
      #define PARALLILOS_SUBMUL_F32    _mm256_sub_ps(a, _mm256_mul_ps(b, c))
    //#elif defined(PARALLILOS_SSE4_2) || defined(PARALLILOS_SSE4_1)
    //#elif defined(PARALLILOS_SSSE3)  || defined(PARALLILOS_SSE3)
    #elif defined(PARALLILOS_SSE2)
      #define PARALLILOS_SET_F32       "SSE2"
      #define PARALLILOS_TYPE_F32      __m128
      #define PARALLILOS_ALIGNMENT_F32 16
      #define PARALLILOS_LOADU_F32     _mm_loadu_ps(data)
      #define PARALLILOS_LOADA_F32     _mm_load_ps(data)
      #define PARALLILOS_STOREU_F32    _mm_storeu_ps(addr, data)
      #define PARALLILOS_STOREA_F32    _mm_store_ps(addr, data)
      #define PARALLILOS_SETVAL_F32    _mm_set1_ps(value)
      #define PARALLILOS_SETZERO_F32   _mm_setzero_ps()
      #define PARALLILOS_MUL_F32       _mm_mul_ps(a, b)
      #define PARALLILOS_ADD_F32       _mm_add_ps(a, b)
      #define PARALLILOS_SUB_F32       _mm_sub_ps(a, b)
      #define PARALLILOS_DIV_F32       _mm_div_ps(a, b)
      #define PARALLILOS_SQRT_F32      _mm_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_F32    _mm_add_ps(a, _mm_mul_ps(b, c))
      #define PARALLILOS_SUBMUL_F32    _mm_sub_ps(a, _mm_mul_ps(b, c))
    #elif defined(PARALLILOS_SSE)
      #define PARALLILOS_SET_F32       "SSE"
      #define PARALLILOS_TYPE_F32      __m128
      #define PARALLILOS_ALIGNMENT_F32 16
      #define PARALLILOS_LOADU_F32     _mm_loadu_ps(data)
      #define PARALLILOS_LOADA_F32     _mm_load_ps(data)
      #define PARALLILOS_STOREU_F32    _mm_storeu_ps(addr, data)
      #define PARALLILOS_STOREA_F32    _mm_store_ps(addr, data)
      #define PARALLILOS_SETVAL_F32    _mm_set1_ps(value)
      #define PARALLILOS_SETZERO_F32   _mm_setzero_ps()
      #define PARALLILOS_MUL_F32       _mm_mul_ps(a, b)
      #define PARALLILOS_ADD_F32       _mm_add_ps(a, b)
      #define PARALLILOS_SUB_F32       _mm_sub_ps(a, b)
      #define PARALLILOS_DIV_F32       _mm_div_ps(a, b)
      #define PARALLILOS_SQRT_F32      _mm_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_F32    _mm_add_ps(a, _mm_mul_ps(b, c))
      #define PARALLILOS_SUBMUL_F32    _mm_sub_ps(a, _mm_mul_ps(b, c))
    #elif defined(PARALLILOS_NEON) || defined(PARALLILOS_NEON64)
      #define PARALLILOS_SET_F32       "NEON"
      #define PARALLILOS_TYPE_F32      float32x4_t
      #define PARALLILOS_LOADU_F32     vld1q_f32(data)
      #define PARALLILOS_LOADA_F32     vld1q_f32(data)
      #define PARALLILOS_STOREU_F32    vst1q_f32(addr, data)
      #define PARALLILOS_STOREA_F32    vst1q_f32(addr, data)
      #define PARALLILOS_SETVAL_F32    vdupq_n_f32(value)
      #define PARALLILOS_SETZERO_F32   vdupq_n_f32(0.0f)
      #define PARALLILOS_MUL_F32       vmulq_f32(a, b)
      #define PARALLILOS_ADD_F32       vaddq_f32(a, b)
      #define PARALLILOS_SUB_F32       vsubq_f32(a, b)
      #define PARALLILOS_DIV_F32       vdivq_f32(a, b)
      #define PARALLILOS_SQRT_F32      vsqrtq_f32(a)
      #define PARALLILOS_ADDMUL_F32    vmlaq_f32(a, b, c)
      #define PARALLILOS_SUBMUL_F32    vmlsq_f32(a, b, c)
    #endif

    // define the best SIMD intrinsics to use
    #if defined(PARALLILOS_AVX512)
      #define PARALLILOS_SET_F64       "AVX512"
      #define PARALLILOS_TYPE_F64      __m512d
      #define PARALLILOS_ALIGNMENT_F64 64
      #define PARALLILOS_LOADU_F64     _mm512_loadu_pd(data)
      #define PARALLILOS_LOADA_F64     _mm512_load_pd(data)
      #define PARALLILOS_STOREU_F64    _mm512_storeu_pd((void*)addr, data)
      #define PARALLILOS_STOREA_F64    _mm512_store_pd((void*)addr, data)
      #define PARALLILOS_SETVAL_F64    _mm512_set1_pd(value)
      #define PARALLILOS_SETZERO_F64   _mm512_setzero_pd()
      #define PARALLILOS_MUL_F64       _mm512_mul_pd(a, b)
      #define PARALLILOS_ADD_F64       _mm512_add_pd(a, b)
      #define PARALLILOS_SUB_F64       _mm512_sub_pd(a, b)
      #define PARALLILOS_DIV_F64       _mm512_div_pd(a, b)
      #define PARALLILOS_SQRT_F64      _mm512_sqrt_pd(a)
      #define PARALLILOS_ADDMUL_F64    _mm512_fmadd_pd(b, c, a)
      #define PARALLILOS_SUBMUL_F64    _mm512_fnmadd_pd(a, b, c)
    #elif defined(PARALLILOS_AVX2)
      #define PARALLILOS_SET_F64       "AVX2"
      #define PARALLILOS_TYPE_F64      __m256d
      #define PARALLILOS_ALIGNMENT_F64 32
      #define PARALLILOS_LOADU_F64     _mm256_loadu_pd(data)
      #define PARALLILOS_LOADA_F64     _mm256_load_pd(data)
      #define PARALLILOS_STOREU_F64    _mm256_storeu_pd(addr, data)
      #define PARALLILOS_STOREA_F64    _mm256_store_pd(addr, data)
      #define PARALLILOS_SETVAL_F64    _mm256_set1_pd(value)
      #define PARALLILOS_SETZERO_F64   _mm256_setzero_pd()
      #define PARALLILOS_MUL_F64       _mm256_mul_pd(a, b)
      #define PARALLILOS_ADD_F64       _mm256_add_pd(a, b)
      #define PARALLILOS_SUB_F64       _mm256_sub_pd(a, b)
      #define PARALLILOS_DIV_F64       _mm256_div_pd(a, b)
      #define PARALLILOS_SQRT_F64      _mm256_sqrt_pd(a)
      #define PARALLILOS_ADDMUL_F64    _mm256_fmadd_pd(b, c, a)
      #define PARALLILOS_SUBMUL_F64    _mm256_fnmadd_pd(a, b, c)
    #elif defined(PARALLILOS_AVX)
      #define PARALLILOS_SET_F64       "AVX"
      #define PARALLILOS_TYPE_F64      __m256d
      #define PARALLILOS_ALIGNMENT_F64 32
      #define PARALLILOS_LOADU_F64     _mm256_loadu_pd(data)
      #define PARALLILOS_LOADA_F64     _mm256_load_pd(data)
      #define PARALLILOS_STOREU_F64    _mm256_storeu_pd(addr, data)
      #define PARALLILOS_STOREA_F64    _mm256_store_pd(addr, data)
      #define PARALLILOS_SETVAL_F64    _mm256_set1_pd(value)
      #define PARALLILOS_SETZERO_F64   _mm256_setzero_pd()
      #define PARALLILOS_MUL_F64       _mm256_mul_pd(a, b)
      #define PARALLILOS_ADD_F64       _mm256_add_pd(a, b)
      #define PARALLILOS_SUB_F64       _mm256_sub_pd(a, b)
      #define PARALLILOS_DIV_F64       _mm256_div_pd(a, b)
      #define PARALLILOS_SQRT_F64      _mm256_sqrt_pd(a)
      #define PARALLILOS_ADDMUL_F64    _mm256_add_pd(a, _mm256_mul_pd(b, c))
      #define PARALLILOS_SUBMUL_F64    _mm256_sub_pd(a, _mm256_mul_pd(b, c))
    //#elif defined(PARALLILOS_SSE4_2) || defined(PARALLILOS_SSE4_1)
    //#elif defined(PARALLILOS_SSSE3)  || defined(PARALLILOS_SSE3)
    #elif defined(PARALLILOS_SSE2)
      #define PARALLILOS_SET_F64       "SSE2"
      #define PARALLILOS_TYPE_F64      __m128d
      #define PARALLILOS_ALIGNMENT_F64 16
      #define PARALLILOS_LOADU_F64     _mm_loadu_pd(data)
      #define PARALLILOS_LOADA_F64     _mm_load_pd(data)
      #define PARALLILOS_STOREU_F64    _mm_storeu_pd(addr, data)
      #define PARALLILOS_STOREA_F64    _mm_store_pd(addr, data)
      #define PARALLILOS_SETVAL_F64    _mm_set1_pd(value)
      #define PARALLILOS_SETZERO_F64   _mm_setzero_pd()
      #define PARALLILOS_MUL_F64       _mm_mul_pd(a, b)
      #define PARALLILOS_ADD_F64       _mm_add_pd(a, b)
      #define PARALLILOS_SUB_F64       _mm_sub_pd(a, b)
      #define PARALLILOS_DIV_F64       _mm_div_pd(a, b)
      #define PARALLILOS_SQRT_F64      _mm_sqrt_pd(a)
      #define PARALLILOS_ADDMUL_F64    _mm_add_pd(a, _mm_mul_pd(b, c))
      #define PARALLILOS_SUBMUL_F64    _mm_sub_pd(a, _mm_mul_pd(b, c))
    //#elif defined(PARALLILOS_SSE)
    #elif defined(PARALLILOS_NEON) || defined(PARALLILOS_NEON64)
      #define PARALLILOS_SET_F64       "NEON"
      #define PARALLILOS_TYPE_F64      float64x4_t
      #define PARALLILOS_LOADU_F64     vld1q_f64(data)
      #define PARALLILOS_LOADA_F64     vld1q_f64(data)
      #define PARALLILOS_STOREU_F64    vst1q_f64(addr, data)
      #define PARALLILOS_STOREA_F64    vst1q_f64(addr, data)
      #define PARALLILOS_SETVAL_F64    vdupq_n_f64(value)
      #define PARALLILOS_SETZERO_F64   vdupq_n_f64(0.0)
      #define PARALLILOS_MUL_F64       vmulq_f64(a, b)
      #define PARALLILOS_ADD_F64       vaddq_f64(a, b)
      #define PARALLILOS_SUB_F64       vsubq_f64(a, b)
      #define PARALLILOS_DIV_F64       vdivq_f64(a, b)
      #define PARALLILOS_SQRT_F64      vsqrtq_f64(a)
      #define PARALLILOS_ADDMUL_F64    vmlaq_f64(a, b, c)
      #define PARALLILOS_SUBMUL_F64    vmlsq_f64(a, b, c)
    #endif

    // define the best SIMD intrinsics to use for signed 32 bit integers
    #if defined(PARALLILOS_AVX512)
      #define PARALLILOS_SET_I32       "AVX512"
      #define PARALLILOS_TYPE_I32      __m512i
      #define PARALLILOS_ALIGNMENT_I32 64
      #define PARALLILOS_LOADU_I32     _mm512_loadu_si512(data)
      #define PARALLILOS_LOADA_I32     _mm512_load_si512(data)
      #define PARALLILOS_STOREU_I32    _mm512_storeu_si512((void*)addr, data)
      #define PARALLILOS_STOREA_I32    _mm512_store_si512((void*)addr, data)
      #define PARALLILOS_SETVAL_I32    _mm512_set1_epi32(value)
      #define PARALLILOS_SETZERO_I32   _mm512_setzero_epi32()
      #define PARALLILOS_MUL_I32       _mm512_mul_epi32(a, b)
      #define PARALLILOS_ADD_I32       _mm512_add_epi32(a, b)
      #define PARALLILOS_SUB_I32       _mm512_sub_epi32(a, b)
      #define PARALLILOS_DIV_I32       _mm512_cvtps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(a), _mm512_cvtepi32_ps(b)))
      #define PARALLILOS_SQRT_I32      _mm512_cvtps_epi32(_mm512_sqrt_ps(_mm512_cvtepi32_ps(a)))
      #define PARALLILOS_ADDMUL_I32    _mm512_add_epi32(a, _mm512_mul_epi32(b, c))
      #define PARALLILOS_SUBMUL_I32    _mm512_sub_epi32(a, _mm512_mul_epi32(b, c))
    #elif defined(PARALLILOS_AVX2)
      #define PARALLILOS_SET_I32       "AVX2"
      #define PARALLILOS_TYPE_I32      __m256i
      #define PARALLILOS_ALIGNMENT_I32 32
      #define PARALLILOS_LOADU_I32     _mm256_loadu_si256((const __m256i*)data)
      #define PARALLILOS_LOADA_I32     _mm256_load_si256((const __m256i*)data)
      #define PARALLILOS_STOREU_I32    _mm256_storeu_si256 ((__m256i*)addr, data)
      #define PARALLILOS_STOREA_I32    _mm256_store_si256((__m256i*)addr, data)
      #define PARALLILOS_SETVAL_I32    _mm256_set1_epi32(value)
      #define PARALLILOS_SETZERO_I32   _mm256_setzero_si256()
      #define PARALLILOS_MUL_I32       _mm256_mul_epi32(a, b)
      #define PARALLILOS_ADD_I32       _mm256_add_epi32(a, b)
      #define PARALLILOS_SUB_I32       _mm256_sub_epi32(a, b)
      #define PARALLILOS_DIV_I32       _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(a), _mm256_cvtepi32_ps(b)))
      #define PARALLILOS_SQRT_I32      _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(a)))
      #define PARALLILOS_ADDMUL_I32    _mm256_add_epi32(a, _mm256_mul_epi32(b, c))
      #define PARALLILOS_SUBMUL_I32    _mm256_sub_epi32(a, _mm256_mul_epi32(b, c))
    //#elif defined(PARALLILOS_AVX)
    //#elif defined(PARALLILOS_SSE4_2) || defined(PARALLILOS_SSE4_1)
    //#elif defined(PARALLILOS_SSSE3)  || defined(PARALLILOS_SSE3)
    #elif defined(PARALLILOS_SSE2)
      #define PARALLILOS_SET_I32       "SSE2"
      #define PARALLILOS_TYPE_I32      __m128i
      #define PARALLILOS_ALIGNMENT_I32 16
      #define PARALLILOS_LOADU_I32     _mm_loadu_si128((const __m128i*)data)
      #define PARALLILOS_LOADA_I32     _mm_load_si128((const __m128i*)data)
      #define PARALLILOS_STOREU_I32    _mm_storeu_si128((__m128i*)addr, data)
      #define PARALLILOS_STOREA_I32    _mm_store_si128((__m128i*)addr, data)
      #define PARALLILOS_SETVAL_I32    _mm_set1_epi32(value)
      #define PARALLILOS_SETZERO_I32   _mm_setzero_si128()
      #define PARALLILOS_MUL_I32       _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
      #define PARALLILOS_ADD_I32       _mm_add_epi32(a, b)
      #define PARALLILOS_SUB_I32       _mm_sub_epi32(a, b)
      #define PARALLILOS_DIV_I32       _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
      #define PARALLILOS_SQRT_I32      _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
      #define PARALLILOS_ADDMUL_I32    _mm_add_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
      #define PARALLILOS_SUBMUL_I32    _mm_sub_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
    //#elif defined(PARALLILOS_SSE)
    //#elif defined(PARALLILOS_NEON) || defined(PARALLILOS_NEON64)
    #endif

    #ifdef PARALLILOS_TYPE_F32
    template <>
    struct simd_properties<float> {
      using type = PARALLILOS_TYPE_F32;
      static constexpr const char* set = PARALLILOS_SET_F32;
      static constexpr size_t alignment = PARALLILOS_ALIGNMENT_F32;
      static constexpr size_t size = sizeof(type) / sizeof(float);
      static constexpr size_t inline iterations(const size_t n) { return n / size; }
    };
    template <>
    struct simd_properties<const float> {
      using type = PARALLILOS_TYPE_F32;
      static constexpr const char* set = PARALLILOS_SET_F32;
      static constexpr size_t alignment = PARALLILOS_ALIGNMENT_F32;
      static constexpr size_t size = sizeof(type) / sizeof(float);
      static constexpr size_t inline iterations(const size_t n) { return n / size; }
    };

    template<>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_setzero<float>(void)
    {
      return PARALLILOS_SETZERO_F32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_loadu(const float* data)
    {
      return PARALLILOS_LOADU_F32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_loada(const float* data)
    {
      return PARALLILOS_LOADA_F32;
    }

    template <>
    PARALLILOS_INLINE void simd_storeu(float* addr, PARALLILOS_TYPE_F32 data)
    {
      PARALLILOS_STOREU_F32;
    }

    template <>
    PARALLILOS_INLINE void simd_storea(float* addr, PARALLILOS_TYPE_F32 data)
    {
      PARALLILOS_STOREA_F32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_setval(const float value)
    {
      return PARALLILOS_SETVAL_F32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_add(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b)
    {
      return PARALLILOS_ADD_F32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_mul(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b)
    {
      return PARALLILOS_MUL_F32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_sub(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b)
    {
      return PARALLILOS_SUB_F32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_div(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b)
    {
      return PARALLILOS_DIV_F32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_sqrt(PARALLILOS_TYPE_F32 a)
    {
      return PARALLILOS_SQRT_F32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_addmul(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b, PARALLILOS_TYPE_F32 c)
    {
      return PARALLILOS_ADDMUL_F32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_submul(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b, PARALLILOS_TYPE_F32 c)
    {
      return PARALLILOS_SUBMUL_F32;
    }
    #endif

    // define a standard API to use SIMD intrinsics
    #ifdef PARALLILOS_TYPE_F64
    template<>
    struct simd_properties<double> {
      using type = PARALLILOS_TYPE_F64;
      static constexpr const char* set = PARALLILOS_SET_F64;
      static constexpr size_t alignment = PARALLILOS_ALIGNMENT_F64;
      static constexpr size_t size = sizeof(type) / sizeof(float);
      static constexpr size_t inline iterations(const size_t n) { return n / size; }
    };

    template<>
    struct simd_properties<const double> {
      using type = PARALLILOS_TYPE_F64;
      static constexpr const char* set = PARALLILOS_SET_F64;
      static constexpr size_t alignment = PARALLILOS_ALIGNMENT_F64;
      static constexpr size_t size = sizeof(type) / sizeof(float);
      static constexpr size_t inline iterations(const size_t n) { return n / size; }
    };

    template<>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_setzero<double>(void)
    {
      return PARALLILOS_SETZERO_F64;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_loadu(const double* data)
    {
      return PARALLILOS_LOADU_F64;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_loada(const double* data)
    {
      return PARALLILOS_LOADA_F64;
    }

    template <> 
    PARALLILOS_INLINE void simd_storeu(double* addr, PARALLILOS_TYPE_F64 data)
    {
      PARALLILOS_STOREU_F64;
    }

    template <>
    PARALLILOS_INLINE void simd_storea(double* addr, PARALLILOS_TYPE_F64 data)
    {
      PARALLILOS_STOREA_F64;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_setval(const double value)
    {
      return PARALLILOS_SETVAL_F64;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_add(const PARALLILOS_TYPE_F64 a, const PARALLILOS_TYPE_F64 b)
    {
      return PARALLILOS_ADD_F64;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_mul(PARALLILOS_TYPE_F64 a, PARALLILOS_TYPE_F64 b)
    {
      return PARALLILOS_MUL_F64;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_sub(PARALLILOS_TYPE_F64 a, PARALLILOS_TYPE_F64 b)
    {
      return PARALLILOS_SUB_F64;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_div(PARALLILOS_TYPE_F64 a, PARALLILOS_TYPE_F64 b)
    {
      return PARALLILOS_DIV_F64;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_sqrt(PARALLILOS_TYPE_F64 a)
    {
      return PARALLILOS_SQRT_F64;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_addmul(PARALLILOS_TYPE_F64 a, PARALLILOS_TYPE_F64 b, PARALLILOS_TYPE_F64 c)
    {
      return PARALLILOS_ADDMUL_F64;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_submul(PARALLILOS_TYPE_F64 a, PARALLILOS_TYPE_F64 b, PARALLILOS_TYPE_F64 c)
    {
      return PARALLILOS_SUBMUL_F64;
    }
    #endif
  
    #ifdef PARALLILOS_TYPE_I32
    template <>
    struct simd_properties<int32_t> {
      using type = PARALLILOS_TYPE_I32;
      static constexpr const char* set = PARALLILOS_SET_I32;
      static constexpr size_t alignment = PARALLILOS_ALIGNMENT_I32;
      static constexpr size_t size = sizeof(type) / sizeof(int32_t);
      static constexpr size_t inline iterations(const size_t n) { return n / size; }
    };

    template <>
    struct simd_properties<const int32_t> {
      using type = PARALLILOS_TYPE_I32;
      static constexpr const char* set = PARALLILOS_SET_I32;
      static constexpr size_t alignment = PARALLILOS_ALIGNMENT_I32;
      static constexpr size_t size = sizeof(type) / sizeof(int32_t);
      static constexpr size_t inline iterations(const size_t n) { return n / size; }
    };

    template<>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_setzero<int32_t>(void)
    {
      return PARALLILOS_SETZERO_I32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_loadu(const int32_t* data)
    {
      return PARALLILOS_LOADU_I32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_loada(const int32_t* data)
    {
      return PARALLILOS_LOADA_I32;
    }

    template <>
    PARALLILOS_INLINE void simd_storeu(int32_t* addr, PARALLILOS_TYPE_I32 data)
    {
      PARALLILOS_STOREU_I32;
    }

    template <>
    PARALLILOS_INLINE void simd_storea(int32_t* addr, PARALLILOS_TYPE_I32 data)
    {
      PARALLILOS_STOREA_I32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_setval(const int32_t value)
    {
      return PARALLILOS_SETVAL_I32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_add(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b)
    {
      return PARALLILOS_ADD_I32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_mul(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b)
    {
      return PARALLILOS_MUL_I32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_sub(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b)
    {
      return PARALLILOS_SUB_I32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_div(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b)
    {
      return PARALLILOS_DIV_I32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_sqrt(PARALLILOS_TYPE_I32 a)
    {
      return PARALLILOS_SQRT_I32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_addmul(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b, PARALLILOS_TYPE_I32 c)
    {
      return PARALLILOS_ADDMUL_I32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_I32 simd_submul(PARALLILOS_TYPE_I32 a, PARALLILOS_TYPE_I32 b, PARALLILOS_TYPE_I32 c)
    {
      return PARALLILOS_SUBMUL_I32;
    }
    #endif
    
    template<typename T>
    T* allocate(const size_t number_of_elements)
    {
      // early return
      if (number_of_elements == 0)
        return nullptr;

      constexpr size_t alignment = simd_properties<T>::alignment;

      // allocate
      void* memory_block = std::malloc(number_of_elements * sizeof(T) + alignment);
      
      // allocation failure
      if (memory_block == nullptr)
        return nullptr;

      // no bookeeping needed
      if (alignment == 0)
        return (T*)memory_block;

      // align on alignement boundary
      void* aligned_memory_block = (void*)((size_t(memory_block) + alignment) & ~(alignment - 1));

      // bookkeeping of original memory block
      ((void**)aligned_memory_block)[-1] = memory_block;

      return (T*)aligned_memory_block;
    }

    template<typename T>
    void deallocate(T* addr)
    {
      if (addr != nullptr) {
        if (simd_properties<T>::alignment == 0)
          std::free(addr);
        else std::free(((void**)addr)[-1]);
      }
    }

    struct deleter {
      void operator()(void* ptr)
      {
        deallocate(ptr);
      }
    };

    template<typename T>
    bool is_aligned(const T* addr)
    {
      return (size_t(addr) & (simd_properties<T>::alignment - 1)) == 0;
    }

    template<typename T>
    T* add_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_PARALLELISM
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(a) && is_aligned(b) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_add(simd_loada(a+k), simd_loada(b+k)));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_add(simd_loadu(a+k), simd_loadu(b+k)));
        }
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
      #ifdef PARALLILOS_PARALLELISM
      const typename simd_properties<T>::type a_vector = simd_setval(a);
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(b) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_add(a_vector, simd_loada(b+k)));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_add(a_vector, simd_loadu(b+k)));
        }
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
      #ifdef PARALLILOS_PARALLELISM
      const typename simd_properties<T>::type b_vector = simd_setval(b);
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(a) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_add(simd_loada(a+k), b_vector));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_add(simd_loadu(a+k), b_vector));
        }
      }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a[k] + b;
      }

      return r;
    }

    template<typename T>
    T* mul_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_PARALLELISM
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(a) && is_aligned(b) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_mul(simd_loada(a+k), simd_loada(b+k)));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_mul(simd_loadu(a+k), simd_loadu(b+k)));
        }
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
      #ifdef PARALLILOS_PARALLELISM
      const typename simd_properties<T>::type a_vector = simd_setval(a);
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(b) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_mul(a_vector, simd_loada(b+k)));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_mul(a_vector, simd_loadu(b+k)));
        }
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
      #ifdef PARALLILOS_PARALLELISM
      const typename simd_properties<T>::type b_vector = simd_setval(b);
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(a) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_mul(simd_loada(a+k), b_vector));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_mul(simd_loadu(a+k), b_vector));
        }
      }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a[k] * b;
      }

      return r;
    }

    template<typename T>
    T* sub_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_PARALLELISM
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(a) && is_aligned(b) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_sub(simd_loada(a+k), simd_loada(b+k)));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_sub(simd_loadu(a+k), simd_loadu(b+k)));
        }
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
      #ifdef PARALLILOS_PARALLELISM
      const typename simd_properties<T>::type a_vector = simd_setval(a);
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(b) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_sub(a_vector, simd_loada(b+k)));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_sub(a_vector, simd_loadu(b+k)));
        }
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
      #ifdef PARALLILOS_PARALLELISM
      const typename simd_properties<T>::type b_vector = simd_setval(b);
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(a) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_sub(simd_loada(a+k), b_vector));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_sub(simd_loadu(a+k), b_vector));
        }
      }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = a[k] - b;
      }

      return r;
    }

    template<typename T>
    T* div_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_PARALLELISM
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(a) && is_aligned(b) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_div(simd_loada(a+k), simd_loada(b+k)));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_div(simd_loadu(a+k), simd_loadu(b+k)));
        }
      }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = (b[k] != 0) ? a[k] / b[k] : std::numeric_limits<T>::min();
      }

      return r;
    }

    template<typename T>
    T* div_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_PARALLELISM
      const typename simd_properties<T>::type a_vector = simd_setval(a);
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(b) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_div(a_vector, simd_loada(b+k)));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_div(a_vector, simd_loadu(b+k)));
        }
      }
      #endif

      // sequential fallback
      if (a != 0) {
        for (; k < n; ++k) {
          r[k] = (b[k] != 0) ? a / b[k] : std::numeric_limits<T>::min();
        }
      }
      else {
        for (; k < n; ++k) {
          r[k] = (b[k] == 0);
        }
      }

      return r;
    }

    template<typename T>
    T* div_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_PARALLELISM
      const typename simd_properties<T>::type b_vector = simd_setval(b);
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(a) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_div(simd_loada(a+k), b_vector));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_div(simd_loadu(a+k), b_vector));
        }
      }
      #endif

      // sequential fallback
      if (b != 0) {
        for (; k < n; ++k) {
          r[k] = a[k] / b;
        }
      }
      else {
        for (; k < n; ++k) {
          r[k] = std::numeric_limits<T>::min();
        }
      }

      return r;
    }

    template<typename T>
    T* sqrt_array(const T* a, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_PARALLELISM
      const size_t size = simd_properties<T>::size;
      const size_t iterations = simd_properties<T>::iterations(n);
      if (is_aligned(a) && is_aligned(r)) {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storea(r+k, simd_sqrt(simd_loada(a+k)));
        }
      }
      else {
        for (size_t i = 0; i < iterations; ++i, k+=size) {
          simd_storeu(r+k, simd_sqrt(simd_loadu(a+k)));
        }
      }
      #endif

      // sequential fallback
      for (; k < n; ++k) {
        r[k] = std::sqrt(a[k]);
      }

      return r;
    }
  
    // cleanup namespace
    #undef PARALLILOS_TYPE_F64
    #undef PARALLILOS_LOADU_F64
    #undef PARALLILOS_LOADA_F64
    #undef PARALLILOS_STOREU_F64
    #undef PARALLILOS_STOREA_F64
    #undef PARALLILOS_SETVAL_F64
    #undef PARALLILOS_SETZERO_F64
    #undef PARALLILOS_ADD_F64
    #undef PARALLILOS_MUL_F64
    #undef PARALLILOS_SUB_F64
    #undef PARALLILOS_DIV_F64
    #undef PARALLILOS_SQRT_F64
    #undef PARALLILOS_ADDMUL_F64
    #undef PARALLILOS_SUBMUL_F64
    //
    #undef PARALLILOS_TYPE_F32
    #undef PARALLILOS_LOADU_F32
    #undef PARALLILOS_LOADA_F32
    #undef PARALLILOS_STOREU_F32
    #undef PARALLILOS_STOREA_F32
    #undef PARALLILOS_SETVAL_F32
    #undef PARALLILOS_SETZERO_F32
    #undef PARALLILOS_ADD_F32
    #undef PARALLILOS_MUL_F32
    #undef PARALLILOS_SUB_F32
    #undef PARALLILOS_DIV_F32
    #undef PARALLILOS_SQRT_F32
    #undef PARALLILOS_ADDMUL_F32
    #undef PARALLILOS_SUBMUL_F32
    //
    #undef PARALLILOS_TYPE_I32
    #undef PARALLILOS_LOADU_I32
    #undef PARALLILOS_LOADA_I32
    #undef PARALLILOS_STOREU_I32
    #undef PARALLILOS_STOREA_I32
    #undef PARALLILOS_SETVAL_I32
    #undef PARALLILOS_SETZERO_I32
    #undef PARALLILOS_ADD_I32
    #undef PARALLILOS_MUL_I32
    #undef PARALLILOS_SUB_I32
    #undef PARALLILOS_DIV_I32
    #undef PARALLILOS_DIV_F32
    #undef PARALLILOS_SQRT_I32
    #undef PARALLILOS_ADDMUL_I32
    #undef PARALLILOS_SUBMUL_I32
    //
    #undef PARALLILOS_INLINE
    #undef PARALLILOS_COMPILER_SUPPORTS_SSE_AVX
    #undef PARALLILOS_COMPILER_SUPPORTS_NEON
  }
}
#endif

/*

// Define the predetermined list of types
using TypeList = std::tuple<int, float, double, char>;

// Helper struct to check if a type is in the list
template<typename T, typename... Types>
struct IsOneOf;

// Base case: Type is not found in the list
template<typename T>
struct IsOneOf<T> : std::false_type {};

// Recursive case: Check if the type matches the first type in the list
template<typename T, typename... Types>
struct IsOneOf<T, T, Types...> : std::true_type {};

// Recursive case: Continue checking with the remaining types in the list
template<typename T, typename U, typename... Types>
struct IsOneOf<T, U, Types...> : IsOneOf<T, Types...> {};



*/