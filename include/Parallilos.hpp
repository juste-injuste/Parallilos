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
    // SIMD aligned memory allocation
    template<typename T>
    T* allocate(const size_t number_of_elements);

    // SIMD aligned memory deallocation
    template<typename T>
    void deallocate(T* addr);

    // custom deleter which invokes deallocate
    struct deleter;

    // check if an address is aligned for SIMD
    bool is_aligned(const void* addr);

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
      #define PARALLILOS_INLINE inline
      #if __cplusplus >= 202302L
        #warning "warning: Parallilos: your compiler is not supported."
      #endif
    #endif

    template<typename T>
    struct simd_properties;

    // load a vector from unaligned data
    template<typename T, typename V = typename simd_properties<T>::type>
    PARALLILOS_INLINE V simd_loadu(const T* data);
    
    // load a vector from aligned data
    template<typename T, typename V = typename simd_properties<T>::type>
    PARALLILOS_INLINE V simd_loada(const T* data);

    // store a vector into unaligned memory
    template<typename T, typename V = typename simd_properties<T>::type>
    PARALLILOS_INLINE void simd_storeu(T* addr, V data);

    // store a vector into aligned memory
    template<typename T, typename V = typename simd_properties<T>::type>
    PARALLILOS_INLINE void simd_storea(T* addr, V data);

    // load a vector with zeros
    template<typename T, typename V = typename simd_properties<T>::type>
    PARALLILOS_INLINE V simd_setzero(void);

    // load a vector with a specific value
    template<typename T, typename V = typename simd_properties<T>::type>
    PARALLILOS_INLINE V simd_setval(const T value);

    // [a] * [b]
    template<typename V>
    PARALLILOS_INLINE V simd_mul(V a, V b);

    // [a] + [b]
    template<typename V>
    PARALLILOS_INLINE V simd_add(V a, V b);

    // [a] - [b]
    template<typename V>
    PARALLILOS_INLINE V simd_sub(V a, V b);

    // [a] / [b]
    template<typename V>
    PARALLILOS_INLINE V simd_div(V a, V b);

    // sqrt([a])
    template<typename V>
    PARALLILOS_INLINE V simd_sqrt(V a);

    // [a] + ([b] * [c]) 
    template<typename V>
    PARALLILOS_INLINE V simd_addmul(V a, V b, V c);

    // [a] - ([b] * [c])
    template<typename V>
    PARALLILOS_INLINE V simd_submul(V a, V b, V c);
  }
}
// --Parallilos library: global level---------------------------------------------------------------
  // define parallelism specifiers and include appropriate headers
  #if defined(__AVX512F__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_AVX512
    #include <immintrin.h>
  #elif defined(__AVX2__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_AVX2
    #include <immintrin.h>
  #elif defined(__AVX__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_AVX
    #include <immintrin.h>
  #elif defined(__SSE4_2__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_SSE4_2
    #include <immintrin.h>
  #elif defined(__SSE4_1__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_SSE4_1
    #include <immintrin.h>
  #elif defined(__SSE3__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_SSE3
    #include <immintrin.h>
  #elif defined(__SSSE3__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_SSSE3
    #include <immintrin.h>
  #elif defined(__SSE2__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_SSE2
    #include <immintrin.h>
  #elif defined(__SSE__) && defined(PARALLILOS_COMPILER_SUPPORTS_SSE_AVX)
    #define PARALLILOS_USE_PARALLELISM
    #define PARALLILOS_USE_SSE
    #include <immintrin.h>
  #elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(PARALLILOS_COMPILER_SUPPORTS_NEON)
    #define PARALLILOS_USE_PARALLELISM
    #ifdef __ARM_ARCH_64
      #define PARALLILOS_USE_NEON64
      #include <arm64_neon.h
    #else
      #define PARALLILOS_USE_NEON
      #include <arm_neon.h>
    #endif
  #else
    #define PARALLILOS_USE_SEQUENTIAL
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
    // define the used SIMD instruction set and the best SIMD alignment
    #if defined(PARALLILOS_USE_AVX512)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "AVX512"
      #define PARALLILOS_ALIGNMENT 64
    #elif defined(PARALLILOS_USE_AVX2)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "AVX2"
      #define PARALLILOS_ALIGNMENT 32
    #elif defined(PARALLILOS_USE_AVX)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "AVX"
      #define PARALLILOS_ALIGNMENT 32
    #elif defined(PARALLILOS_USE_SSE4_2)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "SSE4.2"
      #define PARALLILOS_ALIGNMENT 16
    #elif defined(PARALLILOS_USE_SSE4_1)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "SSE4.1"
      #define PARALLILOS_ALIGNMENT 16
    #elif defined(PARALLILOS_USE_SSSE3)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "SSE3"
      #define PARALLILOS_ALIGNMENT 16
    #elif defined(PARALLILOS_USE_SSE3)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "SSE3"
      #define PARALLILOS_ALIGNMENT 16
    #elif defined(PARALLILOS_USE_SSE2)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "SSE2"
      #define PARALLILOS_ALIGNMENT 16
    #elif defined(PARALLILOS_USE_SSE)
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "SSE"
      #define PARALLILOS_ALIGNMENT 16
    #else
      #define PARALLILOS_EXTENDED_INSTRUCTION_SET "no SIMD set available"
      #define PARALLILOS_ALIGNMENT 0
    #endif
    
    template<typename T>
    T* allocate(const size_t number_of_elements)
    {
      // early return
      if (number_of_elements == 0)
        return nullptr;

      // allocate
      void* memory_block = std::malloc(number_of_elements * sizeof(T) + PARALLILOS_ALIGNMENT);
      
      // allocation failure
      if (memory_block == nullptr)
        return nullptr;

      // align on alignement boundary
      void* aligned_memory_block = (void*)((size_t(memory_block) + PARALLILOS_ALIGNMENT) & ~(size_t(PARALLILOS_ALIGNMENT) - 1));

      // bookkeeping of original memory block
      ((void**)aligned_memory_block)[-1] = memory_block;

      return (T*)aligned_memory_block;
    }

    template<typename T>
    void deallocate(T* addr)
    {
      if (addr != nullptr)
        std::free(((void**)addr)[-1]);
    }

    struct deleter {
      void operator()(void* ptr)
      {
        deallocate(ptr);
      }
    };

    bool is_aligned(const void* addr)
    {
      return (size_t(addr) & (PARALLILOS_ALIGNMENT - 1)) == 0;
    }

    // define the best SIMD intrinsics to use
    #if defined(PARALLILOS_USE_AVX512)
      #define PARALLILOS_TYPE_PS    __m512
      #define PARALLILOS_LOADU_PS   _mm512_loadu_ps(data)
      #define PARALLILOS_LOADA_PS   _mm512_load_ps(data)
      #define PARALLILOS_STOREU_PS  _mm512_storeu_ps((void*)addr, data)
      #define PARALLILOS_STOREA_PS  _mm512_store_ps((void*)addr, data)
      #define PARALLILOS_SETVAL_PS  _mm512_set1_ps(value)
      #define PARALLILOS_SETZERO_PS _mm512_setzero_ps()
      #define PARALLILOS_MUL_PS     _mm512_mul_ps(a, b)
      #define PARALLILOS_ADD_PS     _mm512_add_ps(a, b)
      #define PARALLILOS_SUB_PS     _mm512_sub_ps(a, b)
      #define PARALLILOS_DIV_PS     _mm512_div_ps(a, b)
      #define PARALLILOS_SQRT_PS    _mm512_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_PS  _mm512_fmadd_ps(b, c, a)
      #define PARALLILOS_SUBMUL_PS  _mm512_fnmadd_ps(a, b, c)
      #define PARALLILOS_TYPE_PD    __m512d
      #define PARALLILOS_LOADU_PD   _mm512_loadu_pd(data)
      #define PARALLILOS_LOADA_PD   _mm512_load_pd(data)
      #define PARALLILOS_STOREU_PD  _mm512_storeu_pd((void*)addr, data)
      #define PARALLILOS_STOREA_PD  _mm512_store_pd((void*)addr, data)
      #define PARALLILOS_SETVAL_PD  _mm512_set1_pd(value)
      #define PARALLILOS_SETZERO_PD _mm512_setzero_pd()
      #define PARALLILOS_MUL_PD     _mm512_mul_pd(a, b)
      #define PARALLILOS_ADD_PD     _mm512_add_pd(a, b)
      #define PARALLILOS_SUB_PD     _mm512_sub_pd(a, b)
      #define PARALLILOS_DIV_PD     _mm512_div_pd(a, b)
      #define PARALLILOS_SQRT_PD    _mm512_sqrt_pd(a)
      #define PARALLILOS_ADDMUL_PD  _mm512_fmadd_pd(b, c, a)
      #define PARALLILOS_SUBMUL_PD  _mm512_fnmadd_pd(a, b, c)
    #elif defined(PARALLILOS_USE_AVX2) || defined(PARALLILOS_USE_AVX)
      #define PARALLILOS_TYPE_PD    __m256d
      #define PARALLILOS_TYPE_PS    __m256
      #define PARALLILOS_LOADU_PD   _mm256_loadu_pd(data)
      #define PARALLILOS_LOADU_PS   _mm256_loadu_ps(data)
      #define PARALLILOS_LOADA_PD   _mm256_load_pd(data)
      #define PARALLILOS_LOADA_PS   _mm256_load_ps(data)
      #define PARALLILOS_STOREU_PD  _mm256_storeu_pd(addr, data)
      #define PARALLILOS_STOREU_PS  _mm256_storeu_ps(addr, data)
      #define PARALLILOS_STOREA_PD  _mm256_store_pd(addr, data)
      #define PARALLILOS_STOREA_PS  _mm256_store_ps(addr, data)
      #define PARALLILOS_SETVAL_PD  _mm256_set1_pd(value)
      #define PARALLILOS_SETVAL_PS  _mm256_set1_ps(value)
      #define PARALLILOS_SETZERO_PD _mm256_setzero_pd()
      #define PARALLILOS_SETZERO_PS _mm256_setzero_ps()
      #define PARALLILOS_MUL_PD     _mm256_mul_pd(a, b)
      #define PARALLILOS_MUL_PS     _mm256_mul_ps(a, b)
      #define PARALLILOS_ADD_PD     _mm256_add_pd(a, b)
      #define PARALLILOS_ADD_PS     _mm256_add_ps(a, b)
      #define PARALLILOS_SUB_PD     _mm256_sub_pd(a, b)
      #define PARALLILOS_SUB_PS     _mm256_sub_ps(a, b)
      #define PARALLILOS_DIV_PD     _mm256_div_pd(a, b)
      #define PARALLILOS_DIV_PS     _mm256_div_ps(a, b)
      #define PARALLILOS_SQRT_PD    _mm256_sqrt_pd(a)
      #define PARALLILOS_SQRT_PS    _mm256_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_PD  _mm256_fmadd_pd(b, c, a)
      #define PARALLILOS_ADDMUL_PS  _mm256_fmadd_ps(b, c, a)
      #define PARALLILOS_SUBMUL_PD  _mm256_fnmadd_pd(a, b, c)
      #define PARALLILOS_SUBMUL_PS  _mm256_fnmadd_ps(a, b, c)
    #elif defined(PARALLILOS_USE_SSE)
      #define PARALLILOS_TYPE_PS      __m128
      #define PARALLILOS_LOADU_PS     _mm_loadu_ps(data)
      #define PARALLILOS_LOADA_PS     _mm_load_ps(data)
      #define PARALLILOS_STOREU_PS    _mm_storeu_ps(addr, data)
      #define PARALLILOS_STOREA_PS    _mm_store_ps(addr, data)
      #define PARALLILOS_SETVAL_PS    _mm_set1_ps(value)
      #define PARALLILOS_SETZERO_PS   _mm_setzero_ps()
      #define PARALLILOS_MUL_PS       _mm_mul_ps(a, b)
      #define PARALLILOS_ADD_PS       _mm_add_ps(a, b)
      #define PARALLILOS_SUB_PS       _mm_sub_ps(a, b)
      #define PARALLILOS_DIV_PS       _mm_div_ps(a, b)
      #define PARALLILOS_SQRT_PS      _mm_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_PS    _mm_add_ps(a, _mm_mul_ps(b, c))
      #define PARALLILOS_SUBMUL_PS    _mm_sub_ps(a, _mm_mul_ps(b, c))
    #elif defined(PARALLILOS_USE_SSE2)
      #define PARALLILOS_TYPE_PS      __m128
      #define PARALLILOS_LOADU_PS     _mm_loadu_ps(data)
      #define PARALLILOS_LOADA_PS     _mm_load_ps(data)
      #define PARALLILOS_STOREU_PS    _mm_storeu_ps(addr, data)
      #define PARALLILOS_STOREA_PS    _mm_store_ps(addr, data)
      #define PARALLILOS_SETVAL_PS    _mm_set1_ps(value)
      #define PARALLILOS_SETZERO_PS   _mm_setzero_ps()
      #define PARALLILOS_MUL_PS       _mm_mul_ps(a, b)
      #define PARALLILOS_ADD_PS       _mm_add_ps(a, b)
      #define PARALLILOS_SUB_PS       _mm_sub_ps(a, b)
      #define PARALLILOS_DIV_PS       _mm_div_ps(a, b)
      #define PARALLILOS_SQRT_PS      _mm_sqrt_ps(a)
      #define PARALLILOS_ADDMUL_PS    _mm_add_ps(a, _mm_mul_ps(b, c))
      #define PARALLILOS_SUBMUL_PS    _mm_sub_ps(a, _mm_mul_ps(b, c))
      #define PARALLILOS_TYPE_PD      __m128d
      #define PARALLILOS_LOADU_PD     _mm_loadu_pd(data)
      #define PARALLILOS_LOADA_PD     _mm_load_pd(data)
      #define PARALLILOS_STOREU_PD    _mm_storeu_pd(addr, data)
      #define PARALLILOS_STOREA_PD    _mm_store_pd(addr, data)
      #define PARALLILOS_SETVAL_PD    _mm_set1_pd(value)
      #define PARALLILOS_SETZERO_PD   _mm_setzero_pd()
      #define PARALLILOS_MUL_PD       _mm_mul_pd(a, b)
      #define PARALLILOS_ADD_PD       _mm_add_pd(a, b)
      #define PARALLILOS_SUB_PD       _mm_sub_pd(a, b)
      #define PARALLILOS_DIV_PD       _mm_div_pd(a, b)
      #define PARALLILOS_SQRT_PD      _mm_sqrt_pd(a)
      #define PARALLILOS_ADDMUL_PD    _mm_add_pd(a, _mm_mul_pd(b, c))
      #define PARALLILOS_SUBMUL_PD    _mm_sub_pd(a, _mm_mul_pd(b, c))
      #define PARALLILOS_TYPE_PI32    __m128i
      #define PARALLILOS_LOADU_PI32   _mm_loadu_si128((const __m128i*)data)
      #define PARALLILOS_LOADA_PI32   _mm_load_si128((const __m128i*)data)
      #define PARALLILOS_STOREU_PI32  _mm_storeu_si128((__m128i*)addr, data)
      #define PARALLILOS_STOREA_PI32  _mm_store_si128((__m128i*)addr, data)
      #define PARALLILOS_SETVAL_PI32  _mm_set1_epi32(value)
      #define PARALLILOS_SETZERO_PI32 _mm_setzero_si128()
      #define PARALLILOS_MUL_PI32     _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
      #define PARALLILOS_ADD_PI32     _mm_add_epi32(a, b)
      #define PARALLILOS_SUB_PI32     _mm_sub_epi32(a, b)
      #define PARALLILOS_DIV_PI32     _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
      #define PARALLILOS_SQRT_PI32    _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
      #define PARALLILOS_ADDMUL_PI32  _mm_add_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
      #define PARALLILOS_SUBMUL_PI32  _mm_sub_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
    #elif defined(PARALLILOS_USE_SSE3)   || defined(PARALLILOS_USE_SSSE3) \
       || defined(PARALLILOS_USE_SSE4_1) || defined(PARALLILOS_USE_SSE4_2)
    #elif defined(PARALLILOS_USE_NEON) || defined(PARALLILOS_USE_NEON64)
      #define PARALLILOS_TYPE_PS    float32x4_t
      #define PARALLILOS_LOADU_PS   vld1q_f32(data)
      #define PARALLILOS_LOADA_PS   vld1q_f32(data)
      #define PARALLILOS_STOREU_PS  vst1q_f32(addr, data)
      #define PARALLILOS_STOREA_PS  vst1q_f32(addr, data)
      #define PARALLILOS_SETVAL_PS  vdupq_n_f32(value)
      #define PARALLILOS_SETZERO_PS vdupq_n_f32(0.0f)
      #define PARALLILOS_MUL_PS     vmulq_f32(a, b)
      #define PARALLILOS_ADD_PS     vaddq_f32(a, b)
      #define PARALLILOS_SUB_PS     vsubq_f32(a, b)
      #define PARALLILOS_DIV_PS     vdivq_f32(a, b)
      #define PARALLILOS_SQRT_PS    vsqrtq_f32(a)
      #define PARALLILOS_ADDMUL_PS  vmlaq_f32(a, b, c)
      #define PARALLILOS_SUBMUL_PS  vmlsq_f32(a, b, c)
      #define PARALLILOS_TYPE_PD    float64x4_t
      #define PARALLILOS_LOADU_PD   vld1q_f64(data)
      #define PARALLILOS_LOADA_PD   vld1q_f64(data)
      #define PARALLILOS_STOREU_PD  vst1q_f64(addr, data)
      #define PARALLILOS_STOREA_PD  vst1q_f64(addr, data)
      #define PARALLILOS_SETVAL_PD  vdupq_n_f64(value)
      #define PARALLILOS_SETZERO_PD vdupq_n_f64(0.0)
      #define PARALLILOS_MUL_PD     vmulq_f64(a, b)
      #define PARALLILOS_ADD_PD     vaddq_f64(a, b)
      #define PARALLILOS_SUB_PD     vsubq_f64(a, b)
      #define PARALLILOS_DIV_PD     vdivq_f64(a, b)
      #define PARALLILOS_SQRT_PD    vsqrtq_f64(a)
      #define PARALLILOS_ADDMUL_PD  vmlaq_f64(a, b, c)
      #define PARALLILOS_SUBMUL_PD  vmlsq_f64(a, b, c)
    #endif

    // define a standard API to use SIMD intrinsics
    #ifdef PARALLILOS_USE_PARALLELISM
    #ifdef PARALLILOS_TYPE_PD
    template<>
    struct simd_properties<double> {
      using type = PARALLILOS_TYPE_PD;
      static constexpr size_t size = sizeof(type) / sizeof(float);
      static size_t iterations(const size_t n) {
        return n / size;
      }
    };

    template<>
    PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_setzero<double>(void)
    {
      return PARALLILOS_SETZERO_PD;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_loadu(const double* data)
    {
      return PARALLILOS_LOADU_PD;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_loada(const double* data)
    {
      return PARALLILOS_LOADA_PD;
    }

    template <> 
    PARALLILOS_INLINE void simd_storeu(double* addr, PARALLILOS_TYPE_PD data)
    {
      PARALLILOS_STOREU_PD;
    }

    template <>
    PARALLILOS_INLINE void simd_storea(double* addr, PARALLILOS_TYPE_PD data)
    {
      PARALLILOS_STOREA_PD;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_setval(const double value)
    {
      return PARALLILOS_SETVAL_PD;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_mul(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b)
    {
      return PARALLILOS_MUL_PD;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_add(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b)
    {
      return PARALLILOS_ADD_PD;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_sub(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b)
    {
      return PARALLILOS_SUB_PD;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_div(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b)
    {
      return PARALLILOS_DIV_PD;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_sqrt(PARALLILOS_TYPE_PD a)
    {
      return PARALLILOS_SQRT_PD;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_addmul(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b, PARALLILOS_TYPE_PD c)
    {
      return PARALLILOS_ADDMUL_PD;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PD simd_submul(PARALLILOS_TYPE_PD a, PARALLILOS_TYPE_PD b, PARALLILOS_TYPE_PD c)
    {
      return PARALLILOS_SUBMUL_PD;
    }
    #endif

    #ifdef PARALLILOS_TYPE_PS
    template <>
    struct simd_properties<float> {
      using type = PARALLILOS_TYPE_PS;
      static constexpr size_t size = sizeof(type) / sizeof(float);
      static size_t iterations(const size_t n) {
        return n / size;
      }
    };

    template<>
    PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_setzero<float>(void)
    {
      return PARALLILOS_SETZERO_PS;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_loadu(const float* data)
    {
      return PARALLILOS_LOADU_PS;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_loada(const float* data)
    {
      return PARALLILOS_LOADA_PS;
    }

    template <>
    PARALLILOS_INLINE void simd_storeu(float* addr, PARALLILOS_TYPE_PS data)
    {
      PARALLILOS_STOREU_PS;
    }

    template <>
    PARALLILOS_INLINE void simd_storea(float* addr, PARALLILOS_TYPE_PS data)
    {
      PARALLILOS_STOREA_PS;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_setval(const float value)
    {
      return PARALLILOS_SETVAL_PS;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_mul(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b)
    {
      return PARALLILOS_MUL_PS;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_add(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b)
    {
      return PARALLILOS_ADD_PS;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_sub(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b)
    {
      return PARALLILOS_SUB_PS;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_div(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b)
    {
      return PARALLILOS_DIV_PS;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_sqrt(PARALLILOS_TYPE_PS a)
    {
      return PARALLILOS_SQRT_PS;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_addmul(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b, PARALLILOS_TYPE_PS c)
    {
      return PARALLILOS_ADDMUL_PS;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PS simd_submul(PARALLILOS_TYPE_PS a, PARALLILOS_TYPE_PS b, PARALLILOS_TYPE_PS c)
    {
      return PARALLILOS_SUBMUL_PS;
    }
    #endif
  
    #ifdef PARALLILOS_TYPE_PI32
    template <>
    struct simd_properties<int32_t> {
      using type = PARALLILOS_TYPE_PI32;
      static constexpr size_t size = sizeof(type) / sizeof(int32_t);
      static size_t iterations(const size_t n) {
        return n / size;
      }
    };

    template<>
    PARALLILOS_INLINE PARALLILOS_TYPE_PI32 simd_setzero<int32_t>(void)
    {
      return PARALLILOS_SETZERO_PI32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PI32 simd_loadu(const int32_t* data)
    {
      return PARALLILOS_LOADU_PI32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PI32 simd_loada(const int32_t* data)
    {
      return PARALLILOS_LOADA_PI32;
    }

    template <>
    PARALLILOS_INLINE void simd_storeu(int32_t* addr, PARALLILOS_TYPE_PI32 data)
    {
      PARALLILOS_STOREU_PI32;
    }

    template <>
    PARALLILOS_INLINE void simd_storea(int32_t* addr, PARALLILOS_TYPE_PI32 data)
    {
      PARALLILOS_STOREA_PI32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PI32 simd_setval(const int32_t value)
    {
      return PARALLILOS_SETVAL_PI32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PI32 simd_mul(PARALLILOS_TYPE_PI32 a, PARALLILOS_TYPE_PI32 b)
    {
      return PARALLILOS_MUL_PI32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PI32 simd_add(PARALLILOS_TYPE_PI32 a, PARALLILOS_TYPE_PI32 b)
    {
      return PARALLILOS_ADD_PI32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PI32 simd_sub(PARALLILOS_TYPE_PI32 a, PARALLILOS_TYPE_PI32 b)
    {
      return PARALLILOS_SUB_PI32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PI32 simd_div(PARALLILOS_TYPE_PI32 a, PARALLILOS_TYPE_PI32 b)
    {
      return PARALLILOS_DIV_PI32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PI32 simd_sqrt(PARALLILOS_TYPE_PI32 a)
    {
      return PARALLILOS_SQRT_PI32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PI32 simd_addmul(PARALLILOS_TYPE_PI32 a, PARALLILOS_TYPE_PI32 b, PARALLILOS_TYPE_PI32 c)
    {
      return PARALLILOS_ADDMUL_PI32;
    }

    template <>
    PARALLILOS_INLINE PARALLILOS_TYPE_PI32 simd_submul(PARALLILOS_TYPE_PI32 a, PARALLILOS_TYPE_PI32 b, PARALLILOS_TYPE_PI32 c)
    {
      return PARALLILOS_SUBMUL_PI32;
    }
    #endif
    #endif

    template<typename T>
    T* add_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
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
      #ifdef PARALLILOS_USE_PARALLELISM
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
      #ifdef PARALLILOS_USE_PARALLELISM
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
    T* sub_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
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
      #ifdef PARALLILOS_USE_PARALLELISM
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
      #ifdef PARALLILOS_USE_PARALLELISM
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
    T* mul_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
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
      #ifdef PARALLILOS_USE_PARALLELISM
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
      #ifdef PARALLILOS_USE_PARALLELISM
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
    T* div_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      #ifdef PARALLILOS_USE_PARALLELISM
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
    #undef PARALLILOS_TYPE_PD
    #undef PARALLILOS_LOADU_PD
    #undef PARALLILOS_LOADA_PD
    #undef PARALLILOS_STOREU_PD
    #undef PARALLILOS_STOREA_PD
    #undef PARALLILOS_SETVAL_PD
    #undef PARALLILOS_SETZERO_PD
    #undef PARALLILOS_MUL_PD
    #undef PARALLILOS_ADD_PD
    #undef PARALLILOS_SUB_PD
    #undef PARALLILOS_DIV_PD
    #undef PARALLILOS_DIV_PS
    #undef PARALLILOS_SQRT_PD
    #undef PARALLILOS_ADDMUL_PD
    #undef PARALLILOS_SUBMUL_PD
    //
    #undef PARALLILOS_TYPE_PS
    #undef PARALLILOS_LOADU_PS
    #undef PARALLILOS_LOADA_PS
    #undef PARALLILOS_STOREU_PS
    #undef PARALLILOS_STOREA_PS
    #undef PARALLILOS_SETVAL_PS
    #undef PARALLILOS_SETZERO_PS
    #undef PARALLILOS_MUL_PS
    #undef PARALLILOS_ADD_PS
    #undef PARALLILOS_SUB_PS
    #undef PARALLILOS_SQRT_PS
    #undef PARALLILOS_ADDMUL_PS
    #undef PARALLILOS_SUBMUL_PS
    //
    #undef PARALLILOS_TYPE_PI32
    #undef PARALLILOS_LOADU_PI32
    #undef PARALLILOS_LOADA_PI32
    #undef PARALLILOS_STOREU_PI32
    #undef PARALLILOS_STOREA_PI32
    #undef PARALLILOS_SETVAL_PI32
    #undef PARALLILOS_SETZERO_PI32
    #undef PARALLILOS_MUL_PI32
    #undef PARALLILOS_ADD_PI32
    #undef PARALLILOS_SUB_PI32
    #undef PARALLILOS_DIV_PI32
    #undef PARALLILOS_DIV_PS
    #undef PARALLILOS_SQRT_PI32
    #undef PARALLILOS_ADDMUL_PI32
    #undef PARALLILOS_SUBMUL_PI32
    //
    #undef PARALLILOS_INLINE
    #undef PARALLILOS_COMPILER_SUPPORTS_SSE_AVX
    #undef PARALLILOS_COMPILER_SUPPORTS_NEON
  }
}
#endif