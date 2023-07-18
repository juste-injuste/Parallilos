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
#include <cstdint> // for int32_t
#include <limits>  // for std::numeric_limits
#include <memory>  // for std::unique_ptr
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

    template<typename T>
    using unique_array = std::unique_ptr<T[], deleter>;

    // SIMD aligned memory allocation
    template<typename T>
    unique_array<T> allocate(const size_t number_of_elements);

    // SIMD aligned memory deallocation
    template<typename T>
    void deallocate(T* addr);

    // check if an address is aligned for SIMD
    template<typename T>
    bool is_aligned(const T* addr);

    // [r] = [a] + [b]
    template<typename T>
    inline void add_arrays(const T* a, const T* b, T* r, const size_t n);

    // [r] = a + [b]
    template<typename T>
    inline void add_arrays(const T a, const T* b, T* r, const size_t n);

    // [r] = [a] + b
    template<typename T>
    inline void add_arrays(const T* a, const T b, T* r, const size_t n);

    // [r] = [a] - [b]
    template<typename T>
    inline void sub_arrays(const T* a, const T* b, T* r, const size_t n);

    // [r] = a - [b]
    template<typename T>
    inline void sub_arrays(const T a, const T* b, T* r, const size_t n);

    // [r] = [a] - b
    template<typename T>
    inline void sub_arrays(const T* a, const T b, T* r, const size_t n);

    // [r] = [a] * [b]
    template<typename T>
    inline void mul_arrays(const T* a, const T* b, T* r, const size_t n);

    // [r] = a * [b]
    template<typename T>
    inline void mul_arrays(const T a, const T* b, T* r, const size_t n);

    // [r] = [a] * b
    template<typename T>
    inline void mul_arrays(const T* a, const T b, T* r, const size_t n);

    // [r] = [a] / [b]
    template<typename T>
    inline void div_arrays(const T* a, const T* b, T* r, const size_t n);

    // [r] = a / [b]
    template<typename T>
    inline void div_arrays(const T a, const T* b, T* r, const size_t n);

    // [r] = [a] / b
    template<typename T>
    inline void div_arrays(const T* a, const T b, T* r, const size_t n);

    // [r] = sqrt([a])
    template<typename T>
    inline void sqrt_array(const T* a, T* r, const size_t n);

    // define which instruction set is supported and the best way to inline given the compiler
    #if defined(__GNUC__)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE
      #define PARALLILOS_COMPILER_SUPPORTS_AVX
      #define PARALLILOS_COMPILER_SUPPORTS_NEON
      #define PARALLILOS_INLINE __attribute__((always_inline)) inline
    #elif defined(__clang__)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE
      #define PARALLILOS_COMPILER_SUPPORTS_AVX
      #define PARALLILOS_COMPILER_SUPPORTS_NEON
      #define PARALLILOS_INLINE __attribute__((always_inline)) inline
    #elif defined(__MINGW32__) || defined(__MINGW64__)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE
      #define PARALLILOS_COMPILER_SUPPORTS_AVX
      #define PARALLILOS_INLINE __attribute__((always_inline)) inline
    #elif defined(__apple_build_version__)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE
      #define PARALLILOS_COMPILER_SUPPORTS_AVX
      #define PARALLILOS_INLINE __attribute__((always_inline)) inline
    #elif defined(_MSC_VER)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE
      #define PARALLILOS_COMPILER_SUPPORTS_AVX
      #define PARALLILOS_INLINE __forceinline
    #elif defined(__INTEL_COMPILER)
      #define PARALLILOS_COMPILER_SUPPORTS_SSE
      #define PARALLILOS_COMPILER_SUPPORTS_AVX
      #define PARALLILOS_COMPILER_SUPPORTS_SVML
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

    // unsupported type fallback
    template<typename T>
    struct simd_properties {
      using type = T;
      static constexpr const char* set = "no SIMD instruction set used";
      static constexpr size_t alignment = 0;
      static constexpr size_t size = 1;
      static constexpr size_t inline iterations(const size_t) { return 0; }
      static constexpr size_t inline sequential(const size_t n) { return n; }
      simd_properties() = delete;
    };
    
    // treat const T as T
    template <typename T>
    struct simd_properties<const T> : simd_properties<T> {};

    // load a vector from unaligned data
    template<typename T>
    PARALLILOS_INLINE typename simd_properties<T>::type simd_loadu(const T* data)
    {
    #ifdef LOGGING
      std::clog << "non SIMD loadu" << std::endl;
    #endif
      return *data;
    }

    // load a vector from aligned data
    template<typename T>
    PARALLILOS_INLINE typename simd_properties<T>::type simd_loada(const T* data)
    {
    #ifdef LOGGING
      std::clog << "non SIMD loada" << std::endl; 
    #endif
      return *data;
    }

    // store a vector into unaligned memory
    template<typename T>
    PARALLILOS_INLINE void simd_storeu(T* addr, typename simd_properties<T>::type data)
    {
    #ifdef LOGGING
      std::clog << "non SIMD storeu" << std::endl; 
    #endif
      *addr = data;
    }

    // store a vector into aligned memory
    template<typename T>
    PARALLILOS_INLINE void simd_storea(T* addr, typename simd_properties<T>::type data)
    {
    #ifdef LOGGING
      std::clog << "non SIMD storea" << std::endl;
    #endif 
      *addr = data;
    }

    // load a vector with zeros
    template<typename T, typename V>
    PARALLILOS_INLINE V simd_setzero(void)
    {
    #ifdef LOGGING
      std::clog << "non SIMD setzero" << std::endl; 
    #endif
      return 0;
    }

    // load a vector with a specific value
    template<typename T>
    PARALLILOS_INLINE typename simd_properties<T>::type simd_setval(const T value)
    {
    #ifdef LOGGING
      std::clog << "non SIMD setval" << std::endl;
    #endif 
      return value;
    }
    
    // [a] + [b]
    template<typename V>
    PARALLILOS_INLINE V simd_add(V a, V b)
    {
    #ifdef LOGGING
      std::clog << "non SIMD add" << std::endl; 
    #endif
      return a + b;
    }

    // [a] * [b]
    template<typename V>
    PARALLILOS_INLINE V simd_mul(V a, V b)
    {
    #ifdef LOGGING
      std::clog << "non SIMD mul" << std::endl; 
    #endif
      return a * b;
    }

    // [a] - [b]
    template<typename V>
    PARALLILOS_INLINE V simd_sub(V a, V b)
    {
    #ifdef LOGGING
      std::clog << "non SIMD sub" << std::endl; 
    #endif
      return a + b;
    }
    
    // [a] / [b]
    template<typename V>
    PARALLILOS_INLINE V simd_div(V a, V b)
    {
    #ifdef LOGGING
      std::clog << "non SIMD div" << std::endl; 
    #endif
      return a / b;
    }
    
    // sqrt([a])
    template<typename V>
    PARALLILOS_INLINE V simd_sqrt(V a)
    { 
    #ifdef LOGGING
      std::clog << "non SIMD sqrt" << std::endl;
    #endif 
      return std::sqrt(a);
    }

    // [a] + ([b] * [c]) 
    template<typename V>
    PARALLILOS_INLINE V simd_addmul(V a, V b, V c)
    {
    #ifdef LOGGING
      std::clog << "non SIMD addmul" << std::endl; 
    #endif
      return a + b * c;
    }

    // [a] - ([b] * [c])
    template<typename V>
    PARALLILOS_INLINE V simd_submul(V a, V b, V c)
    {
    #ifdef LOGGING
      std::clog << "non SIMD submul" << std::endl;
    #endif 
      return a - b * c;
    }
  }
}
// --Parallilos library: global level---------------------------------------------------------------
  // define parallelism specifiers and include appropriate headers
  #if defined(PARALLILOS_COMPILER_SUPPORTS_AVX)
    #if defined(__AVX512F__)
      #define PARALLILOS_PARALLELISM
      #define PARALLILOS_AVX512F
      #include <immintrin.h>
    #endif
    #if defined(__AVX2__)
      #define PARALLILOS_PARALLELISM
      #define PARALLILOS_AVX2
      #include <immintrin.h>
    #endif
    #if defined(__FMA__)
      #define PARALLILOS_PARALLELISM
      #define PARALLILOS_FMA
      #include <immintrin.h>
    #endif
    #if defined(__AVX__)
      #define PARALLILOS_PARALLELISM
      #define PARALLILOS_AVX
      #include <immintrin.h>
    #endif
  #endif

  #if defined(PARALLILOS_COMPILER_SUPPORTS_SSE)
    #if defined(__SSE4_2__)
      #define PARALLILOS_PARALLELISM
      #define PARALLILOS_SSE4_2
      #include <immintrin.h>
    #endif
    #if defined(__SSE4_1__)
      #define PARALLILOS_PARALLELISM
      #define PARALLILOS_SSE4_1
      #include <immintrin.h>
    #endif
    #if defined(__SSSE3__)
      #define PARALLILOS_PARALLELISM
      #define PARALLILOS_SSSE3
      #include <immintrin.h>
    #endif
    #if defined(__SSE3__)
      #define PARALLILOS_PARALLELISM
      #define PARALLILOS_SSE3
      #include <immintrin.h>
    #endif
    #if defined(__SSE2__)
      #define PARALLILOS_PARALLELISM
      #define PARALLILOS_SSE2
      #include <immintrin.h>
    #endif
    #if defined(__SSE__)
      #define PARALLILOS_PARALLELISM
      #define PARALLILOS_SSE
      #include <immintrin.h>
    #endif
  #endif

  #if defined(PARALLILOS_COMPILER_SUPPORTS_NEON)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
      #define PARALLILOS_PARALLELISM
      #ifdef __ARM_ARCH_64
        #define PARALLILOS_NEON64
        #include <arm64_neon.h
      #else
        #define PARALLILOS_NEON
        #include <arm_neon.h>
      #endif
    #endif
  #endif

  #if defined(PARALLILOS_COMPILER_SUPPORTS_SVML)
    #define PARALLILOS_SVML
  #endif

  #if !defined(PARALLILOS_PARALLELISM)
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
    #if defined(PARALLILOS_AVX512F)
      #define PARALLILOS_SET_F32                   "AVX512F"
      #define PARALLILOS_TYPE_F32                  __m512
      #define PARALLILOS_ALIGNMENT_F32             64
      #define PARALLILOS_LOADU_F32(data)           _mm512_loadu_ps(data)
      #define PARALLILOS_LOADA_F32(data)           _mm512_load_ps(data)
      #define PARALLILOS_STOREU_F32(addr, data)    _mm512_storeu_ps((void*)addr, data)
      #define PARALLILOS_STOREA_F32(addr, data)    _mm512_store_ps((void*)addr, data)
      #define PARALLILOS_SETVAL_F32(value)         _mm512_set1_ps(value)
      #define PARALLILOS_SETZERO_F32()             _mm512_setzero_ps()
      #define PARALLILOS_MUL_F32(a, b)             _mm512_mul_ps(a, b)
      #define PARALLILOS_ADD_F32(a, b)             _mm512_add_ps(a, b)
      #define PARALLILOS_SUB_F32(a, b)             _mm512_sub_ps(a, b)
      #define PARALLILOS_DIV_F32(a, b)             _mm512_div_ps(a, b)
      #define PARALLILOS_SQRT_F32(a)               _mm512_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_F32(a, b, c)       _mm512_fmadd_ps(b, c, a)
      #define PARALLILOS_SUBMUL_F32(a, b, c)       _mm512_fnmadd_ps(a, b, c)
    #elif defined(PARALLILOS_FMA)
      #define PARALLILOS_SET_F32                   "AVX, FMA"
      #define PARALLILOS_TYPE_F32                  __m256
      #define PARALLILOS_ALIGNMENT_F32             32
      #define PARALLILOS_LOADU_F32(data)           _mm256_loadu_ps(data)
      #define PARALLILOS_LOADA_F32(data)           _mm256_load_ps(data)
      #define PARALLILOS_STOREU_F32(addr, data)    _mm256_storeu_ps(addr, data)
      #define PARALLILOS_STOREA_F32(addr, data)    _mm256_store_ps(addr, data)
      #define PARALLILOS_SETVAL_F32(value)         _mm256_set1_ps(value)
      #define PARALLILOS_SETZERO_F32()             _mm256_setzero_ps()
      #define PARALLILOS_MUL_F32(a, b)             _mm256_mul_ps(a, b)
      #define PARALLILOS_ADD_F32(a, b)             _mm256_add_ps(a, b)
      #define PARALLILOS_SUB_F32(a, b)             _mm256_sub_ps(a, b)
      #define PARALLILOS_DIV_F32(a, b)             _mm256_div_ps(a, b)
      #define PARALLILOS_SQRT_F32(a)               _mm256_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_F32(a, b, c)       _mm256_fmadd_ps(b, c, a)
      #define PARALLILOS_SUBMUL_F32(a, b, c)       _mm256_fnmadd_ps(a, b, c)
    #elif defined(PARALLILOS_AVX)
      #define PARALLILOS_SET_F32                   "AVX"
      #define PARALLILOS_TYPE_F32                  __m256
      #define PARALLILOS_ALIGNMENT_F32             32
      #define PARALLILOS_LOADU_F32(data)           _mm256_loadu_ps(data)
      #define PARALLILOS_LOADA_F32(data)           _mm256_load_ps(data)
      #define PARALLILOS_STOREU_F32(addr, data)    _mm256_storeu_ps(addr, data)
      #define PARALLILOS_STOREA_F32(addr, data)    _mm256_store_ps(addr, data)
      #define PARALLILOS_SETVAL_F32(value)         _mm256_set1_ps(value)
      #define PARALLILOS_SETZERO_F32()             _mm256_setzero_ps()
      #define PARALLILOS_MUL_F32(a, b)             _mm256_mul_ps(a, b)
      #define PARALLILOS_ADD_F32(a, b)             _mm256_add_ps(a, b)
      #define PARALLILOS_SUB_F32(a, b)             _mm256_sub_ps(a, b)
      #define PARALLILOS_DIV_F32(a, b)             _mm256_div_ps(a, b)
      #define PARALLILOS_SQRT_F32(a)               _mm256_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_F32(a, b, c)       _mm256_add_ps(a, _mm256_mul_ps(b, c))
      #define PARALLILOS_SUBMUL_F32(a, b, c)       _mm256_sub_ps(a, _mm256_mul_ps(b, c))
    #elif defined(PARALLILOS_SSE)
      #define PARALLILOS_SET_F32                   "SSE"
      #define PARALLILOS_TYPE_F32                  __m128
      #define PARALLILOS_ALIGNMENT_F32             16
      #define PARALLILOS_LOADU_F32(data)           _mm_loadu_ps(data)
      #define PARALLILOS_LOADA_F32(data)           _mm_load_ps(data)
      #define PARALLILOS_STOREU_F32(addr, data)    _mm_storeu_ps(addr, data)
      #define PARALLILOS_STOREA_F32(addr, data)    _mm_store_ps(addr, data)
      #define PARALLILOS_SETVAL_F32(value)         _mm_set1_ps(value)
      #define PARALLILOS_SETZERO_F32()             _mm_setzero_ps()
      #define PARALLILOS_MUL_F32(a, b)             _mm_mul_ps(a, b)
      #define PARALLILOS_ADD_F32(a, b)             _mm_add_ps(a, b)
      #define PARALLILOS_SUB_F32(a, b)             _mm_sub_ps(a, b)
      #define PARALLILOS_DIV_F32(a, b)             _mm_div_ps(a, b)
      #define PARALLILOS_SQRT_F32(a)               _mm_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_F32(a, b, c)       _mm_add_ps(a, _mm_mul_ps(b, c))
      #define PARALLILOS_SUBMUL_F32(a, b, c)       _mm_sub_ps(a, _mm_mul_ps(b, c))
    #elif defined(PARALLILOS_NEON) || defined(PARALLILOS_NEON64)
      #define PARALLILOS_SET_F32                   "NEON"
      #define PARALLILOS_TYPE_F32                  float32x4_t
      #define PARALLILOS_ALIGNMENT_F32             0
      #define PARALLILOS_LOADU_F32(data)           vld1q_f32(data)
      #define PARALLILOS_LOADA_F32(data)           vld1q_f32(data)
      #define PARALLILOS_STOREU_F32(addr, data)    vst1q_f32(addr, data)
      #define PARALLILOS_STOREA_F32(addr, data)    vst1q_f32(addr, data)
      #define PARALLILOS_SETVAL_F32(value)         vdupq_n_f32(value)
      #define PARALLILOS_SETZERO_F32()             vdupq_n_f32(0.0f)
      #define PARALLILOS_MUL_F32(a, b)             vmulq_f32(a, b)
      #define PARALLILOS_ADD_F32(a, b)             vaddq_f32(a, b)
      #define PARALLILOS_SUB_F32(a, b)             vsubq_f32(a, b)
      #define PARALLILOS_DIV_F32(a, b)             vdivq_f32(a, b)
      #define PARALLILOS_SQRT_F32(a)               vsqrtq_f32(a)
      #define PARALLILOS_ADDMUL_F32(a, b, c)       vmlaq_f32(a, b, c)
      #define PARALLILOS_SUBMUL_F32(a, b, c)       vmlsq_f32(a, b, c)
    #endif

    #ifdef PARALLILOS_TYPE_F32
    template <>
    struct simd_properties<float> {
      using type = PARALLILOS_TYPE_F32;
      static constexpr const char* set = PARALLILOS_SET_F32;
      static constexpr size_t alignment = PARALLILOS_ALIGNMENT_F32;
      static constexpr size_t size = sizeof(type) / sizeof(float);
      static constexpr size_t inline iterations(const size_t n) { return n / size; }
      static constexpr size_t inline sequential(const size_t n) { return n - iterations(n)*size; }
      simd_properties() = delete;
    };

    template<>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_setzero<float>(void)
    {
      return PARALLILOS_SETZERO_F32();
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_loadu(const float* data)
    {
      return PARALLILOS_LOADU_F32(data);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_loada(const float* data)
    {
      return PARALLILOS_LOADA_F32(data);
    }

    template <>
    PARALLILOS_INLINE void simd_storeu(float* addr, PARALLILOS_TYPE_F32 data)
    {
      PARALLILOS_STOREU_F32(addr, data);
    }

    template <>
    PARALLILOS_INLINE void simd_storea(float* addr, PARALLILOS_TYPE_F32 data)
    {
      PARALLILOS_STOREA_F32(addr, data);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_setval(const float value)
    {
      return PARALLILOS_SETVAL_F32(value);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_add(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b)
    {
      return PARALLILOS_ADD_F32(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_mul(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b)
    {
      return PARALLILOS_MUL_F32(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_sub(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b)
    {
      return PARALLILOS_SUB_F32(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_div(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b)
    {
      return PARALLILOS_DIV_F32(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_sqrt(PARALLILOS_TYPE_F32 a)
    {
      return PARALLILOS_SQRT_F32(a);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_addmul(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b, PARALLILOS_TYPE_F32 c)
    {
      return PARALLILOS_ADDMUL_F32(a, b, c);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F32 simd_submul(PARALLILOS_TYPE_F32 a, PARALLILOS_TYPE_F32 b, PARALLILOS_TYPE_F32 c)
    {
      return PARALLILOS_SUBMUL_F32(a, b, c);
    }
    #endif

    // define the best SIMD intrinsics to use
    #if defined(PARALLILOS_AVX512F)
      #define PARALLILOS_SET_F64                   "AVX512F"
      #define PARALLILOS_TYPE_F64                  __m512d
      #define PARALLILOS_ALIGNMENT_F64             64
      #define PARALLILOS_LOADU_F64(data)           _mm512_loadu_pd(data)
      #define PARALLILOS_LOADA_F64(data)           _mm512_load_pd(data)
      #define PARALLILOS_STOREU_F64(addr, data)    _mm512_storeu_pd((void*)addr, data)
      #define PARALLILOS_STOREA_F64(addr, data)    _mm512_store_pd((void*)addr, data)
      #define PARALLILOS_SETVAL_F64(value)         _mm512_set1_pd(value)
      #define PARALLILOS_SETZERO_F64()             _mm512_setzero_pd()
      #define PARALLILOS_MUL_F64(a, b)             _mm512_mul_pd(a, b)
      #define PARALLILOS_ADD_F64(a, b)             _mm512_add_pd(a, b)
      #define PARALLILOS_SUB_F64(a, b)             _mm512_sub_pd(a, b)
      #define PARALLILOS_DIV_F64(a, b)             _mm512_div_pd(a, b)
      #define PARALLILOS_SQRT_F64(a)               _mm512_sqrt_pd(a)
      #define PARALLILOS_ADDMUL_F64(a, b, c)       _mm512_fmadd_pd(b, c, a)
      #define PARALLILOS_SUBMUL_F64(a, b, c)       _mm512_fnmadd_pd(a, b, c)
    #elif defined(PARALLILOS_FMA)
      #define PARALLILOS_SET_F64                   "AVX, FMA"
      #define PARALLILOS_TYPE_F64                  __m256d
      #define PARALLILOS_ALIGNMENT_F64             32
      #define PARALLILOS_LOADU_F64(data)           _mm256_loadu_pd(data)
      #define PARALLILOS_LOADA_F64(data)           _mm256_load_pd(data)
      #define PARALLILOS_STOREU_F64(addr, data)    _mm256_storeu_pd(addr, data)
      #define PARALLILOS_STOREA_F64(addr, data)    _mm256_store_pd(addr, data)
      #define PARALLILOS_SETVAL_F64(value)         _mm256_set1_pd(value)
      #define PARALLILOS_SETZERO_F64()             _mm256_setzero_pd()
      #define PARALLILOS_MUL_F64(a, b)             _mm256_mul_pd(a, b)
      #define PARALLILOS_ADD_F64(a, b)             _mm256_add_pd(a, b)
      #define PARALLILOS_SUB_F64(a, b)             _mm256_sub_pd(a, b)
      #define PARALLILOS_DIV_F64(a, b)             _mm256_div_pd(a, b)
      #define PARALLILOS_SQRT_F64(a)               _mm256_sqrt_pd(a)
      #define PARALLILOS_ADDMUL_F64(a, b, c)       _mm256_fmadd_pd(b, c, a)
      #define PARALLILOS_SUBMUL_F64(a, b, c)       _mm256_fnmadd_pd(a, b, c)
    #elif defined(PARALLILOS_AVX)
      #define PARALLILOS_SET_F64                   "AVX"
      #define PARALLILOS_TYPE_F64                  __m256d
      #define PARALLILOS_ALIGNMENT_F64             32
      #define PARALLILOS_LOADU_F64(data)           _mm256_loadu_pd(data)
      #define PARALLILOS_LOADA_F64(data)           _mm256_load_pd(data)
      #define PARALLILOS_STOREU_F64(addr, data)    _mm256_storeu_pd(addr, data)
      #define PARALLILOS_STOREA_F64(addr, data)    _mm256_store_pd(addr, data)
      #define PARALLILOS_SETVAL_F64(value)         _mm256_set1_pd(value)
      #define PARALLILOS_SETZERO_F64()             _mm256_setzero_pd()
      #define PARALLILOS_MUL_F64(a, b)             _mm256_mul_pd(a, b)
      #define PARALLILOS_ADD_F64(a, b)             _mm256_add_pd(a, b)
      #define PARALLILOS_SUB_F64(a, b)             _mm256_sub_pd(a, b)
      #define PARALLILOS_DIV_F64(a, b)             _mm256_div_pd(a, b)
      #define PARALLILOS_SQRT_F64(a)               _mm256_sqrt_pd(a)
      #define PARALLILOS_ADDMUL_F64(a, b, c)       _mm256_add_pd(a, _mm256_mul_pd(b, c))
      #define PARALLILOS_SUBMUL_F64(a, b, c)       _mm256_sub_pd(a, _mm256_mul_pd(b, c))
    #elif defined(PARALLILOS_SSE2)
      #define PARALLILOS_SET_F64                   "SSE2"
      #define PARALLILOS_TYPE_F64                  __m128d
      #define PARALLILOS_ALIGNMENT_F64             16
      #define PARALLILOS_LOADU_F64(data)           _mm_loadu_pd(data)
      #define PARALLILOS_LOADA_F64(data)           _mm_load_pd(data)
      #define PARALLILOS_STOREU_F64(addr, data)    _mm_storeu_pd(addr, data)
      #define PARALLILOS_STOREA_F64(addr, data)    _mm_store_pd(addr, data)
      #define PARALLILOS_SETVAL_F64(value)         _mm_set1_pd(value)
      #define PARALLILOS_SETZERO_F64()             _mm_setzero_pd()
      #define PARALLILOS_MUL_F64(a, b)             _mm_mul_pd(a, b)
      #define PARALLILOS_ADD_F64(a, b)             _mm_add_pd(a, b)
      #define PARALLILOS_SUB_F64(a, b)             _mm_sub_pd(a, b)
      #define PARALLILOS_DIV_F64(a, b)             _mm_div_pd(a, b)
      #define PARALLILOS_SQRT_F64(a)               _mm_sqrt_pd(a)
      #define PARALLILOS_ADDMUL_F64(a, b, c)       _mm_add_pd(a, _mm_mul_pd(b, c))
      #define PARALLILOS_SUBMUL_F64(a, b, c)       _mm_sub_pd(a, _mm_mul_pd(b, c))
    #elif defined(PARALLILOS_NEON) || defined(PARALLILOS_NEON64)
      #define PARALLILOS_SET_F64                   "NEON"
      #define PARALLILOS_TYPE_F64                  float64x4_t
      #define PARALLILOS_ALIGNMENT_F64             0
      #define PARALLILOS_LOADU_F64(data)           vld1q_f64(data)
      #define PARALLILOS_LOADA_F64(data)           vld1q_f64(data)
      #define PARALLILOS_STOREU_F64(addr, data)    vst1q_f64(addr, data)
      #define PARALLILOS_STOREA_F64(addr, data)    vst1q_f64(addr, data)
      #define PARALLILOS_SETVAL_F64(value)         vdupq_n_f64(value)
      #define PARALLILOS_SETZERO_F64()             vdupq_n_f64(0.0)
      #define PARALLILOS_MUL_F64(a, b)             vmulq_f64(a, b)
      #define PARALLILOS_ADD_F64(a, b)             vaddq_f64(a, b)
      #define PARALLILOS_SUB_F64(a, b)             vsubq_f64(a, b)
      #define PARALLILOS_DIV_F64(a, b)             vdivq_f64(a, b)
      #define PARALLILOS_SQRT_F64(a)               vsqrtq_f64(a)
      #define PARALLILOS_ADDMUL_F64(a, b, c)       vmlaq_f64(a, b, c)
      #define PARALLILOS_SUBMUL_F64(a, b, c)       vmlsq_f64(a, b, c)
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
      static constexpr size_t inline sequential(const size_t n) { return n - iterations(n)*size; }
      simd_properties() = delete;
    };

    template<>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_setzero<double>(void)
    {
      return PARALLILOS_SETZERO_F64();
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_loadu(const double* data)
    {
      return PARALLILOS_LOADU_F64(data);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_loada(const double* data)
    {
      return PARALLILOS_LOADA_F64(data);
    }

    template <> 
    PARALLILOS_INLINE void simd_storeu(double* addr, PARALLILOS_TYPE_F64 data)
    {
      PARALLILOS_STOREU_F64(addr, data);
    }

    template <>
    PARALLILOS_INLINE void simd_storea(double* addr, PARALLILOS_TYPE_F64 data)
    {
      PARALLILOS_STOREA_F64(addr, data);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_setval(const double value)
    {
      return PARALLILOS_SETVAL_F64(value);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_add(const PARALLILOS_TYPE_F64 a, const PARALLILOS_TYPE_F64 b)
    {
      return PARALLILOS_ADD_F64(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_mul(PARALLILOS_TYPE_F64 a, PARALLILOS_TYPE_F64 b)
    {
      return PARALLILOS_MUL_F64(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_sub(PARALLILOS_TYPE_F64 a, PARALLILOS_TYPE_F64 b)
    {
      return PARALLILOS_SUB_F64(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_div(PARALLILOS_TYPE_F64 a, PARALLILOS_TYPE_F64 b)
    {
      return PARALLILOS_DIV_F64(a, b);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_sqrt(PARALLILOS_TYPE_F64 a)
    {
      return PARALLILOS_SQRT_F64(a);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_addmul(PARALLILOS_TYPE_F64 a, PARALLILOS_TYPE_F64 b, PARALLILOS_TYPE_F64 c)
    {
      return PARALLILOS_ADDMUL_F64(a, b, c);
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_F64 simd_submul(PARALLILOS_TYPE_F64 a, PARALLILOS_TYPE_F64 b, PARALLILOS_TYPE_F64 c)
    {
      return PARALLILOS_SUBMUL_F64(a, b, c);
    }
    #endif

    // define the best SIMD intrinsics to use for signed 32 bit integers
    #if defined(PARALLILOS_AVX512F)
      #define PARALLILOS_SET_I32                   "AVX512F"
      #define PARALLILOS_TYPE_I32                  __m512i
      #define PARALLILOS_ALIGNMENT_I32             64
      #define PARALLILOS_LOADU_I32(data)           _mm512_loadu_si512(data)
      #define PARALLILOS_LOADA_I32(data)           _mm512_load_si512(data)
      #define PARALLILOS_STOREU_I32(addr, data)    _mm512_storeu_si512((void*)addr, data)
      #define PARALLILOS_STOREA_I32(addr, data)    _mm512_store_si512((void*)addr, data)
      #define PARALLILOS_SETVAL_I32(value)         _mm512_set1_epi32(value)
      #define PARALLILOS_SETZERO_I32()             _mm512_setzero_epi32()
      #define PARALLILOS_MUL_I32(a, b)             _mm512_mul_epi32(a, b)
      #define PARALLILOS_ADD_I32(a, b)             _mm512_add_epi32(a, b)
      #define PARALLILOS_SUB_I32(a, b)             _mm512_sub_epi32(a, b)
      #define PARALLILOS_DIV_I32(a, b)             _mm512_cvtps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(a), _mm512_cvtepi32_ps(b)))
      #define PARALLILOS_SQRT_I32(a)               _mm512_cvtps_epi32(_mm512_sqrt_ps(_mm512_cvtepi32_ps(a)))
      #define PARALLILOS_ADDMUL_I32(a, b, c)       _mm512_add_epi32(a, _mm512_mul_epi32(b, c))
      #define PARALLILOS_SUBMUL_I32(a, b, c)       _mm512_sub_epi32(a, _mm512_mul_epi32(b, c))
      #if defined(PARALLILOS_SVML)
        #undef  PARALLILOS_DIV_I32
        #define PARALLILOS_DIV_I32(a, b)           _mm512_div_epi32(a, b)
      #endif
    #elif defined(PARALLILOS_AVX2)
      #define PARALLILOS_SET_I32                   "AVX2, AVX"
      #define PARALLILOS_TYPE_I32                  __m256i
      #define PARALLILOS_ALIGNMENT_I32             32
      #define PARALLILOS_LOADU_I32(data)           _mm256_loadu_si256((const __m256i*)data)
      #define PARALLILOS_LOADA_I32(data)           _mm256_load_si256((const __m256i*)data)
      #define PARALLILOS_STOREU_I32(addr, data)    _mm256_storeu_si256 ((__m256i*)addr, data)
      #define PARALLILOS_STOREA_I32(addr, data)    _mm256_store_si256((__m256i*)addr, data)
      #define PARALLILOS_SETVAL_I32(value)         _mm256_set1_epi32(value)
      #define PARALLILOS_SETZERO_I32()             _mm256_setzero_si256()
      #define PARALLILOS_MUL_I32(a, b)             _mm256_mul_epi32(a, b)
      #define PARALLILOS_ADD_I32(a, b)             _mm256_add_epi32(a, b)
      #define PARALLILOS_SUB_I32(a, b)             _mm256_sub_epi32(a, b)
      #define PARALLILOS_DIV_I32(a, b)             _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(a), _mm256_cvtepi32_ps(b)))
      #define PARALLILOS_SQRT_I32(a)               _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(a)))
      #define PARALLILOS_ADDMUL_I32(a, b, c)       _mm256_add_epi32(a, _mm256_mul_epi32(b, c))
      #define PARALLILOS_SUBMUL_I32(a, b, c)       _mm256_sub_epi32(a, _mm256_mul_epi32(b, c))
      #if defined(PARALLILOS_SVML)
        #undef  PARALLILOS_DIV_I32
        #define PARALLILOS_DIV_I32(a, b)           _mm256_div_epi32(a, b)
      #endif
    #elif defined(PARALLILOS_SSE4_1)
      #define PARALLILOS_SET_I32                   "SSE4.1, SSE2"
      #define PARALLILOS_TYPE_I32                  __m128i
      #define PARALLILOS_ALIGNMENT_I32             16
      #define PARALLILOS_LOADU_I32(data)           _mm_loadu_si128((const __m128i*)data)
      #define PARALLILOS_LOADA_I32(data)           _mm_load_si128((const __m128i*)data)
      #define PARALLILOS_STOREU_I32(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
      #define PARALLILOS_STOREA_I32(addr, data)    _mm_store_si128((__m128i*)addr, data)
      #define PARALLILOS_SETVAL_I32(value)         _mm_set1_epi32(value)
      #define PARALLILOS_SETZERO_I32()             _mm_setzero_si128()
      #define PARALLILOS_MUL_I32(a, b)             _mm_mul_epi32(a, b)
      #define PARALLILOS_ADD_I32(a, b)             _mm_add_epi32(a, b)
      #define PARALLILOS_SUB_I32(a, b)             _mm_sub_epi32(a, b)
      #define PARALLILOS_DIV_I32(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
      #define PARALLILOS_SQRT_I32(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
      #define PARALLILOS_ADDMUL_I32(a, b, c)       _mm_add_epi32(a, _mm_mul_epi32(b, c))
      #define PARALLILOS_SUBMUL_I32(a, b, c)       _mm_sub_epi32(a, _mm_mul_epi32(b, c))
      #if defined(PARALLILOS_SVML)
        #undef  PARALLILOS_DIV_I32
        #define PARALLILOS_DIV_I32(a, b)           _mm_div_epi32(a, b)
      #endif
    #elif defined(PARALLILOS_SSE2)
      #define PARALLILOS_SET_I32                   "SSE2"
      #define PARALLILOS_TYPE_I32                  __m128i
      #define PARALLILOS_ALIGNMENT_I32             16
      #define PARALLILOS_LOADU_I32(data)           _mm_loadu_si128((const __m128i*)data)
      #define PARALLILOS_LOADA_I32(data)           _mm_load_si128((const __m128i*)data)
      #define PARALLILOS_STOREU_I32(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
      #define PARALLILOS_STOREA_I32(addr, data)    _mm_store_si128((__m128i*)addr, data)
      #define PARALLILOS_SETVAL_I32(value)         _mm_set1_epi32(value)
      #define PARALLILOS_SETZERO_I32()             _mm_setzero_si128()
      #define PARALLILOS_MUL_I32(a, b)             _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
      #define PARALLILOS_ADD_I32(a, b)             _mm_add_epi32(a, b)
      #define PARALLILOS_SUB_I32(a, b)             _mm_sub_epi32(a, b)
      #define PARALLILOS_DIV_I32(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
      #define PARALLILOS_SQRT_I32(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
      #define PARALLILOS_ADDMUL_I32(a, b, c)       _mm_add_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
      #define PARALLILOS_SUBMUL_I32(a, b, c)       _mm_sub_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
      #if defined(PARALLILOS_SVML)
        #undef  PARALLILOS_DIV_I32
        #define PARALLILOS_DIV_I32(a, b)           _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
      #endif
    #endif
  
    #ifdef PARALLILOS_TYPE_I32
    template <>
    struct simd_properties<int32_t> {
      using type = PARALLILOS_TYPE_I32;
      static constexpr const char* set = PARALLILOS_SET_I32;
      static constexpr size_t alignment = PARALLILOS_ALIGNMENT_I32;
      static constexpr size_t size = sizeof(type) / sizeof(int32_t);
      static constexpr size_t inline iterations(const size_t n) { return n / size; }
      static constexpr size_t inline sequential(const size_t n) { return n - iterations(n)*size; }
      simd_properties() = delete;
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
    #endif

    struct deleter {
      template<typename T>
      void operator()(T* ptr) { deallocate(ptr); }
    };
    
    template<typename T>
    unique_array<T> allocate(const size_t number_of_elements)
    {
    // early return
    if (number_of_elements == 0)
      return unique_array<T>(nullptr);

    // alignment requirement for simd
    constexpr size_t alignment = simd_properties<T>::alignment;
    
    #if __cplusplus >= 201703L
      if (simd_properties<T>::alignment == 0)
        return reinterpret_cast<T*>(std::malloc(number_of_elements * sizeof(T)));
      else
        return reinterpret_cast<T*>(std::aligned_alloc(alignment, number_of_elements * sizeof(T)));
    #else
      // allocate
      void* memory_block = std::malloc(number_of_elements * sizeof(T) + alignment);
      
      // allocation failure
      if (memory_block == nullptr)
        return unique_array<T>(nullptr);

      // no bookeeping needed
      if (alignment == 0)
        return unique_array<T>(reinterpret_cast<T*>(memory_block));

      // align on alignement boundary
      void* aligned_memory_block = reinterpret_cast<void*>((size_t(memory_block) + alignment) & ~(alignment - 1));

      // bookkeeping of original memory block
      reinterpret_cast<void**>(aligned_memory_block)[-1] = memory_block;

      return unique_array<T>(reinterpret_cast<T*>(aligned_memory_block));
    #endif
    }

    template<typename T>
    void deallocate(T* addr)
    {
    #if __cplusplus >= 201703L
      std::free(addr);
    #else
      if (addr != nullptr) {
        if (simd_properties<T>::alignment != 0)
          std::free(reinterpret_cast<void**>(addr)[-1]);
        else std::free(addr);
      }
    #endif
    }

    template<typename T>
    bool is_aligned(const T* addr)
    {
      return (size_t(addr) & (simd_properties<T>::alignment - 1)) == 0;
    }

    template<typename T>
    void add_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(a) && is_aligned(b) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_add(simd_loada(a+k), simd_loada(b+k)));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_add(simd_loadu(a+k), simd_loadu(b+k)));
      }

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a[k] + b[k];
    }

    template<typename T>
    void add_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        const typename simd_properties<T>::type a_vector = simd_setval(a);
        if (is_aligned(a) && is_aligned(b) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_add(a_vector, simd_loada(b+k)));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_add(a_vector, simd_loadu(b+k)));
      }

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a + b[k];
    }

    template<typename T>
    void add_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const typename simd_properties<T>::type b_vector = simd_setval(b);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(a) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_add(simd_loada(a+k), b_vector));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_add(simd_loadu(a+k), b_vector));
      }

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a[k] + b;
    }

    template<typename T>
    void mul_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(a) && is_aligned(b) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_mul(simd_loada(a+k), simd_loada(b+k)));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_mul(simd_loadu(a+k), simd_loadu(b+k)));
      }

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a[k] * b[k];
    }

    template<typename T>
    void mul_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const typename simd_properties<T>::type a_vector = simd_setval(a);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(b) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_mul(a_vector, simd_loada(b+k)));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_mul(a_vector, simd_loadu(b+k)));
      }

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a * b[k];
    }

    template<typename T>
    void mul_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const typename simd_properties<T>::type b_vector = simd_setval(b);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(a) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_mul(simd_loada(a+k), b_vector));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_mul(simd_loadu(a+k), b_vector));
      }

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a[k] * b;
    }

    template<typename T>
    void sub_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(a) && is_aligned(b) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_sub(simd_loada(a+k), simd_loada(b+k)));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_sub(simd_loadu(a+k), simd_loadu(b+k)));
      }

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a[k] - b[k];
    }

    template<typename T>
    void sub_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const typename simd_properties<T>::type a_vector = simd_setval(a);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(b) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_sub(a_vector, simd_loada(b+k)));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_sub(a_vector, simd_loadu(b+k)));
      }

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a - b[k];
    }

    template<typename T>
    void sub_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const typename simd_properties<T>::type b_vector = simd_setval(b);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(a) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_sub(simd_loada(a+k), b_vector));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_sub(simd_loadu(a+k), b_vector));
      }

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a[k] - b;
    }

    template<typename T>
    void div_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(a) && is_aligned(b) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_div(simd_loada(a+k), simd_loada(b+k)));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_div(simd_loadu(a+k), simd_loadu(b+k)));
      }

      // sequential fallback
      for (; k < n; ++k)
        r[k] = (b[k] != 0) ? a[k] / b[k] : std::numeric_limits<T>::min();
    }

    template<typename T>
    void div_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const typename simd_properties<T>::type a_vector = simd_setval(a);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(b) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_div(a_vector, simd_loada(b+k)));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_div(a_vector, simd_loadu(b+k)));
      }

      // sequential fallback
      if (a != 0)
        for (; k < n; ++k)
          r[k] = (b[k] != 0) ? a / b[k] : std::numeric_limits<T>::min();
      else
        for (; k < n; ++k)
          r[k] = (b[k] == 0);
    }

    template<typename T>
    void div_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const typename simd_properties<T>::type b_vector = simd_setval(b);
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(a) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_div(simd_loada(a+k), b_vector));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_div(simd_loadu(a+k), b_vector));
      }

      // sequential fallback
      if (b != 0)
        for (; k < n; ++k)
          r[k] = a[k] / b;
      else
        for (; k < n; ++k)
          r[k] = std::numeric_limits<T>::min();
    }

    template<typename T>
    void sqrt_array(const T* a, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (simd_properties<T>::alignment != 0) {
        const size_t size = simd_properties<T>::size;
        const size_t iterations = simd_properties<T>::iterations(n);
        if (is_aligned(a) && is_aligned(r))
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storea(r+k, simd_sqrt(simd_loada(a+k)));
        else
          for (size_t i = 0; i < iterations; ++i, k+=size)
            simd_storeu(r+k, simd_sqrt(simd_loadu(a+k)));
      }

      // sequential fallback
      for (; k < n; ++k)
        r[k] = std::sqrt(a[k]);
    }
  
    // cleanup namespace
    #undef PARALLILOS_TYPE_F32
    #undef PARALLILOS_ALIGNMENT_F32
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
    #undef PARALLILOS_TYPE_F64
    #undef PARALLILOS_ALIGNMENT_F64
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
    #undef PARALLILOS_TYPE_I32
    #undef PARALLILOS_ALIGNMENT_I32
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
    #undef PARALLILOS_DIV_I32
    #undef PARALLILOS_SQRT_I32
    #undef PARALLILOS_ADDMUL_I32
    #undef PARALLILOS_SUBMUL_I32
    //
    #undef PARALLILOS_INLINE
    #undef PARALLILOS_COMPILER_SUPPORTS_SSE
    #undef PARALLILOS_COMPILER_SUPPORTS_AVX
    #undef PARALLILOS_COMPILER_SUPPORTS_SVML
    #undef PARALLILOS_COMPILER_SUPPORTS_NEON
  }
}
#endif