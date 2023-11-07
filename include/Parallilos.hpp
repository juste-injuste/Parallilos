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
#include <cstddef>  // for size_t
#include <cmath>    // for std::sqrt
#include <cstdlib>  // for std::malloc, std::free
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::unique_ptr
#include <ostream>  // for std::ostream
#include <iostream> // for std::cerr
# include <type_traits> // for std::is_arithmetic
#if defined(PARALLILOS_WARNINGS)
# include <string>      // for std::string, std::to_string
# include <type_traits> // for std::is_integral, std::is_unsigned, std::is_floating_point
# include <typeinfo>    // to use operator typeid
#endif
// --Parallilos library-----------------------------------------------------------------------------
namespace Parallilos
{
  namespace _Version
  {
    // library version
    constexpr long NUMBER = 000001000;
    constexpr long MAJOR  = 000      ;
    constexpr long MINOR  =    001   ;
    constexpr long PATCH  =       000;
  }

  namespace Global
  {
    std::ostream wrn{std::cerr.rdbuf()};
  }

  // define which instruction sets are supported and the best way to inline given the compiler
#if defined(__GNUC__)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_COMPILER_SUPPORTS_NEON
# define PARALLILOS_INLINE __attribute__((always_inline)) inline
#elif defined(__clang__)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_COMPILER_SUPPORTS_NEON
# define PARALLILOS_INLINE __attribute__((always_inline)) inline
#elif defined(__MINGW32__) || defined(__MINGW64__)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_INLINE __attribute__((always_inline)) inline
#elif defined(__apple_build_version__)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_INLINE __forceinline
#elif defined(__INTEL_COMPILER)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_COMPILER_SUPPORTS_SVML
# define PARALLILOS_INLINE __forceinline
#elif defined(__ARMCC_VERSION)
# define PARALLILOS_COMPILER_SUPPORTS_NEON
# define PARALLILOS_INLINE __forceinline
#else
# if __cplusplus >= 202302L
#   warning "warning: Parallilos: your compiler is not supported."
# endif
# define PARALLILOS_INLINE inline
#endif

// Arithmetic "concept"
# define PARALLILOS_ARITHMETIC(T) typename T, typename = typename std::is_arithmetic<T>::type

// Vector type associated with T
# define PARALLILOS_VECTOR_OF(T)  typename simd<T>::vector_type

  // logging utilities
#if defined(PARALLILOS_WARNINGS)
  namespace Logging
  {
    template<PARALLILOS_ARITHMETIC(T)>
    inline std::string type_name()
    {
      if (std::is_floating_point<T>::value)
      {
        return "float" + std::to_string(sizeof(T) * 8);
      }

      if (std::is_unsigned<T>::value)
      {
        return "uint" + std::to_string(sizeof(T) * 8);
      }

      return "int" + std::to_string(sizeof(T) * 8);
    }

    template<typename T, typename... Tn>
    inline auto get_type_name() -> typename std::enable_if<sizeof...(Tn) == 0, std::string>::type
    {
      return type_name<T>();
    }

    template<typename T, typename... Tn>
    inline auto get_type_name() -> typename std::enable_if<sizeof...(Tn) != 0, std::string>::type
    {
      return type_name<T>() + ", " + get_type_name<Tn...>();
    }
  }
# define PARALLILOS_TYPE_WARNING(...)                                      \
  Global::wrn << "warning: Parallilos: " << __func__ << '(' <<             \
  Logging::get_type_name<__VA_ARGS__>() << "): SIMD not used" << std::endl
#else
# define PARALLILOS_TYPE_WARNING(...) /* to enable warnings #define PARALLILOS_WARNINGS */
#endif
// --Parallilos library: frontend forward declarations----------------------------------------------
  inline namespace Frontend
  {
    // custom deleter which invokes free_array
    struct Deleter;

    template<PARALLILOS_ARITHMETIC(T)>
    using Array = std::unique_ptr<T[], Deleter>;

    // SIMD aligned memory allocation
    template<PARALLILOS_ARITHMETIC(T)>
    inline Array<T> make_array(const size_t number_of_elements);

    // SIMD aligned memory deallocation
    template<PARALLILOS_ARITHMETIC(T)>
    inline void free_array(T* addr);

    // check if an address is aligned for SIMD
    template<PARALLILOS_ARITHMETIC(T)>
    inline bool is_aligned(const T* addr);

    // unsupported type fallback
    template<PARALLILOS_ARITHMETIC(T)>
    struct simd
    {
      using vector_type = T;
      static constexpr const char* set  = "no SIMD instruction set used for this type";
      static constexpr size_t alignment = 0;
      static constexpr size_t size      = 1;

      static constexpr size_t inline passes(const size_t)
      {
        return 0;
      }

      static constexpr size_t inline sequential(const size_t n_elements)
      {
        return n_elements;
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
    
    // treat const T as T
    template <typename T>
    struct simd<const T> : simd<T>
    {};

    // load a vector from unaligned data
    template<PARALLILOS_ARITHMETIC(T)>
    PARALLILOS_INLINE auto simd_loadu(const T* data) -> PARALLILOS_VECTOR_OF(T)
    {
      PARALLILOS_TYPE_WARNING(T);
      return *data;
    }

    // load a vector from aligned data
    template<PARALLILOS_ARITHMETIC(T)>
    PARALLILOS_INLINE auto simd_loada(const T* data) -> PARALLILOS_VECTOR_OF(T)
    {
      PARALLILOS_TYPE_WARNING(T);
      return *data;
    }

    // store a vector into unaligned memory
    template<PARALLILOS_ARITHMETIC(T)>
    PARALLILOS_INLINE void simd_storeu(T* addr, PARALLILOS_VECTOR_OF(T) data)
    {
      PARALLILOS_TYPE_WARNING(T, PARALLILOS_VECTOR_OF(T));
      *addr = data;
    }

    // store a vector into aligned memory
    template<PARALLILOS_ARITHMETIC(T)>
    PARALLILOS_INLINE void simd_storea(T* addr, PARALLILOS_VECTOR_OF(T) data)
    {
      PARALLILOS_TYPE_WARNING(T, PARALLILOS_VECTOR_OF(T));
      *addr = data;
    }

    // load a vector with zeros
    template<PARALLILOS_ARITHMETIC(T)>
    PARALLILOS_INLINE auto simd_setzero() -> PARALLILOS_VECTOR_OF(T)
    {
      PARALLILOS_TYPE_WARNING(T);
      return 0;
    }

    // load a vector with a specific value
    template<PARALLILOS_ARITHMETIC(T)>
    PARALLILOS_INLINE auto simd_setval(const T value) -> PARALLILOS_VECTOR_OF(T)
    {
      PARALLILOS_TYPE_WARNING(T);
      return value;
    }
    
    // [a] + [b]
    template<typename V>
    PARALLILOS_INLINE V simd_add(V a, V b)
    {
      PARALLILOS_TYPE_WARNING(V, V);
      return a + b;
    }

    // [a] * [b]
    template<typename V>
    PARALLILOS_INLINE V simd_mul(V a, V b)
    {
      PARALLILOS_TYPE_WARNING(V, V);
      return a * b;
    }

    // [a] - [b]
    template<typename V>
    PARALLILOS_INLINE V simd_sub(V a, V b)
    {
      PARALLILOS_TYPE_WARNING(V, V);
      return a - b;
    }
    
    // [a] / [b]
    template<typename V>
    PARALLILOS_INLINE V simd_div(V a, V b)
    {
      PARALLILOS_TYPE_WARNING(V, V);
      return a / b;
    }
    
    // sqrt([a])
    template<typename V>
    PARALLILOS_INLINE V simd_sqrt(V a)
    { 
      PARALLILOS_TYPE_WARNING(V);
      return std::sqrt(a);
    }

    // [a] + ([b] * [c]) 
    template<typename V>
    PARALLILOS_INLINE V simd_addmul(V a, V b, V c)
    {
      PARALLILOS_TYPE_WARNING(V, V);
      return a + b * c;
    }

    // [a] - ([b] * [c])
    template<typename V>
    PARALLILOS_INLINE V simd_submul(V a, V b, V c)
    {
      PARALLILOS_TYPE_WARNING(V, V, V);
      return a - b * c;
    }
  }
}
// --Parallilos library: global level---------------------------------------------------------------
  // define parallelism specifiers and include appropriate headers
# if defined(PARALLILOS_COMPILER_SUPPORTS_AVX)
#   if defined(__AVX512F__)
#     define PARALLILOS_PARALLELISM
#     define PARALLILOS_AVX512F
#     if not defined(__OPTIMIZE__)
#       define __OPTIMIZE__
#       include <immintrin.h>
#       undef  __OPTIMIZE__
#     else
#       include <immintrin.h>
#     endif
#   endif
#   if defined(__AVX2__)
#     define PARALLILOS_PARALLELISM
#     define PARALLILOS_AVX2
#     if not defined(__OPTIMIZE__)
#       define __OPTIMIZE__
#       include <immintrin.h>
#       undef  __OPTIMIZE__
#     else
#       include <immintrin.h>
#     endif
#   endif
#   if defined(__FMA__)
#     define PARALLILOS_PARALLELISM
#     define PARALLILOS_FMA
#     if not defined(__OPTIMIZE__)
#       define __OPTIMIZE__
#       include <immintrin.h>
#       undef  __OPTIMIZE__
#     else
#       include <immintrin.h>
#     endif
#   endif
#   if defined(__AVX__)
#     define PARALLILOS_PARALLELISM
#     define PARALLILOS_AVX
#     if not defined(__OPTIMIZE__)
#       define __OPTIMIZE__
#       include <immintrin.h>
#       undef  __OPTIMIZE__
#     else
#       include <immintrin.h>
#     endif
#   endif
# endif

# if defined(PARALLILOS_COMPILER_SUPPORTS_SSE)
#   if defined(__SSE4_2__)
#     define PARALLILOS_PARALLELISM
#     define PARALLILOS_SSE4_2
#     if not defined(__OPTIMIZE__)
#       define __OPTIMIZE__
#       include <immintrin.h>
#       undef  __OPTIMIZE__
#     else
#       include <immintrin.h>
#     endif
#   endif
#   if defined(__SSE4_1__)
#     define PARALLILOS_PARALLELISM
#     define PARALLILOS_SSE4_1
#     if not defined(__OPTIMIZE__)
#       define __OPTIMIZE__
#       include <immintrin.h>
#       undef  __OPTIMIZE__
#     else
#       include <immintrin.h>
#     endif
#   endif
#   if defined(__SSSE3__)
#     define PARALLILOS_PARALLELISM
#     define PARALLILOS_SSSE3
#     if not defined(__OPTIMIZE__)
#       define __OPTIMIZE__
#       include <immintrin.h>
#       undef  __OPTIMIZE__
#     else
#       include <immintrin.h>
#     endif
#   endif
#   if defined(__SSE3__)
#     define PARALLILOS_PARALLELISM
#     define PARALLILOS_SSE3
#     if not defined(__OPTIMIZE__)
#       define __OPTIMIZE__
#       include <immintrin.h>
#       undef  __OPTIMIZE__
#     else
#       include <immintrin.h>
#     endif
#   endif
#   if defined(__SSE2__)
#     define PARALLILOS_PARALLELISM
#     define PARALLILOS_SSE2
#     if not defined(__OPTIMIZE__)
#       define __OPTIMIZE__
#       include <immintrin.h>
#       undef  __OPTIMIZE__
#     else
#       include <immintrin.h>
#     endif
#   endif
#   if defined(__SSE__)
#     define PARALLILOS_PARALLELISM
#     define PARALLILOS_SSE
#     if not defined(__OPTIMIZE__)
#       define __OPTIMIZE__
#       include <immintrin.h>
#       undef  __OPTIMIZE__
#     else
#       include <immintrin.h>
#     endif
#   endif
# endif

# if defined(PARALLILOS_COMPILER_SUPPORTS_NEON)
#   if defined(__ARM_NEON) || defined(__ARM_NEON__)
#     define PARALLILOS_PARALLELISM
#     ifdef __ARM_ARCH_64
#       define PARALLILOS_NEON64
#       include <arm64_neon.h
#     else
#       define PARALLILOS_NEON
#       include <arm_neon.h>
#     endif
#   endif
  #endif

# if defined(PARALLILOS_COMPILER_SUPPORTS_SVML)
#   define PARALLILOS_SVML
# endif

# if not defined(PARALLILOS_PARALLELISM)
#   define PARALLILOS_SEQUENTIAL
#   if __cplusplus >= 202302L
#     warning "warning: Parallilos: no SIMD instruction set used, sequential fallback used."
#   endif
# endif
// --Parallilos library: backend--------------------------------------------------------------------
namespace Parallilos
{
// --Parallilos library: frontend-------------------------------------------------------------------
  inline namespace Frontend
  {
    // define the best SIMD intrinsics to use for 32 bit floating numbers
# if defined(PARALLILOS_AVX512F)
#   define PARALLILOS_SET_F32                   "AVX512F"
#   define PARALLILOS_TYPE_F32                  __m512
#   define PARALLILOS_ALIGNMENT_F32             64
#   define PARALLILOS_LOADU_F32(data)           _mm512_loadu_ps(data)
#   define PARALLILOS_LOADA_F32(data)           _mm512_load_ps(data)
#   define PARALLILOS_STOREU_F32(addr, data)    _mm512_storeu_ps((void*)addr, data)
#   define PARALLILOS_STOREA_F32(addr, data)    _mm512_store_ps((void*)addr, data)
#   define PARALLILOS_SETVAL_F32(value)         _mm512_set1_ps(value)
#   define PARALLILOS_SETZERO_F32()             _mm512_setzero_ps()
#   define PARALLILOS_MUL_F32(a, b)             _mm512_mul_ps(a, b)
#   define PARALLILOS_ADD_F32(a, b)             _mm512_add_ps(a, b)
#   define PARALLILOS_SUB_F32(a, b)             _mm512_sub_ps(a, b)
#   define PARALLILOS_DIV_F32(a, b)             _mm512_div_ps(a, b)
#   define PARALLILOS_SQRT_F32(a)               _mm512_sqrt_ps(a)
#   define PARALLILOS_ADDMUL_F32(a, b, c)       _mm512_fmadd_ps(b, c, a)
#   define PARALLILOS_SUBMUL_F32(a, b, c)       _mm512_fnmadd_ps(a, b, c)
# elif defined(PARALLILOS_FMA)
#   define PARALLILOS_SET_F32                   "AVX, FMA"
#   define PARALLILOS_TYPE_F32                  __m256
#   define PARALLILOS_ALIGNMENT_F32             32
#   define PARALLILOS_LOADU_F32(data)           _mm256_loadu_ps(data)
#   define PARALLILOS_LOADA_F32(data)           _mm256_load_ps(data)
#   define PARALLILOS_STOREU_F32(addr, data)    _mm256_storeu_ps(addr, data)
#   define PARALLILOS_STOREA_F32(addr, data)    _mm256_store_ps(addr, data)
#   define PARALLILOS_SETVAL_F32(value)         _mm256_set1_ps(value)
#   define PARALLILOS_SETZERO_F32()             _mm256_setzero_ps()
#   define PARALLILOS_MUL_F32(a, b)             _mm256_mul_ps(a, b)
#   define PARALLILOS_ADD_F32(a, b)             _mm256_add_ps(a, b)
#   define PARALLILOS_SUB_F32(a, b)             _mm256_sub_ps(a, b)
#   define PARALLILOS_DIV_F32(a, b)             _mm256_div_ps(a, b)
#   define PARALLILOS_SQRT_F32(a)               _mm256_sqrt_ps(a)
#   define PARALLILOS_ADDMUL_F32(a, b, c)       _mm256_fmadd_ps(b, c, a)
#   define PARALLILOS_SUBMUL_F32(a, b, c)       _mm256_fnmadd_ps(a, b, c)
#elif defined(PARALLILOS_AVX)
#   define PARALLILOS_SET_F32                   "AVX"
#   define PARALLILOS_TYPE_F32                  __m256
#   define PARALLILOS_ALIGNMENT_F32             32
#   define PARALLILOS_LOADU_F32(data)           _mm256_loadu_ps(data)
#   define PARALLILOS_LOADA_F32(data)           _mm256_load_ps(data)
#   define PARALLILOS_STOREU_F32(addr, data)    _mm256_storeu_ps(addr, data)
#   define PARALLILOS_STOREA_F32(addr, data)    _mm256_store_ps(addr, data)
#   define PARALLILOS_SETVAL_F32(value)         _mm256_set1_ps(value)
#   define PARALLILOS_SETZERO_F32()             _mm256_setzero_ps()
#   define PARALLILOS_MUL_F32(a, b)             _mm256_mul_ps(a, b)
#   define PARALLILOS_ADD_F32(a, b)             _mm256_add_ps(a, b)
#   define PARALLILOS_SUB_F32(a, b)             _mm256_sub_ps(a, b)
#   define PARALLILOS_DIV_F32(a, b)             _mm256_div_ps(a, b)
#   define PARALLILOS_SQRT_F32(a)               _mm256_sqrt_ps(a)
#   define PARALLILOS_ADDMUL_F32(a, b, c)       _mm256_add_ps(a, _mm256_mul_ps(b, c))
#   define PARALLILOS_SUBMUL_F32(a, b, c)       _mm256_sub_ps(a, _mm256_mul_ps(b, c))
# elif defined(PARALLILOS_SSE)
#   define PARALLILOS_SET_F32                   "SSE"
#   define PARALLILOS_TYPE_F32                  __m128
#   define PARALLILOS_ALIGNMENT_F32             16
#   define PARALLILOS_LOADU_F32(data)           _mm_loadu_ps(data)
#   define PARALLILOS_LOADA_F32(data)           _mm_load_ps(data)
#   define PARALLILOS_STOREU_F32(addr, data)    _mm_storeu_ps(addr, data)
#   define PARALLILOS_STOREA_F32(addr, data)    _mm_store_ps(addr, data)
#   define PARALLILOS_SETVAL_F32(value)         _mm_set1_ps(value)
#   define PARALLILOS_SETZERO_F32()             _mm_setzero_ps()
#   define PARALLILOS_MUL_F32(a, b)             _mm_mul_ps(a, b)
#   define PARALLILOS_ADD_F32(a, b)             _mm_add_ps(a, b)
#   define PARALLILOS_SUB_F32(a, b)             _mm_sub_ps(a, b)
#   define PARALLILOS_DIV_F32(a, b)             _mm_div_ps(a, b)
#   define PARALLILOS_SQRT_F32(a)               _mm_sqrt_ps(a)
#   define PARALLILOS_ADDMUL_F32(a, b, c)       _mm_add_ps(a, _mm_mul_ps(b, c))
#   define PARALLILOS_SUBMUL_F32(a, b, c)       _mm_sub_ps(a, _mm_mul_ps(b, c))
# elif defined(PARALLILOS_NEON) || defined(PARALLILOS_NEON64)
#   define PARALLILOS_SET_F32                   "NEON"
#   define PARALLILOS_TYPE_F32                  float32x4_t
#   define PARALLILOS_ALIGNMENT_F32             0
#   define PARALLILOS_LOADU_F32(data)           vld1q_f32(data)
#   define PARALLILOS_LOADA_F32(data)           vld1q_f32(data)
#   define PARALLILOS_STOREU_F32(addr, data)    vst1q_f32(addr, data)
#   define PARALLILOS_STOREA_F32(addr, data)    vst1q_f32(addr, data)
#   define PARALLILOS_SETVAL_F32(value)         vdupq_n_f32(value)
#   define PARALLILOS_SETZERO_F32()             vdupq_n_f32(0.0f)
#   define PARALLILOS_MUL_F32(a, b)             vmulq_f32(a, b)
#   define PARALLILOS_ADD_F32(a, b)             vaddq_f32(a, b)
#   define PARALLILOS_SUB_F32(a, b)             vsubq_f32(a, b)
#   define PARALLILOS_DIV_F32(a, b)             vdivq_f32(a, b)
#   define PARALLILOS_SQRT_F32(a)               vsqrtq_f32(a)
#   define PARALLILOS_ADDMUL_F32(a, b, c)       vmlaq_f32(a, b, c)
#   define PARALLILOS_SUBMUL_F32(a, b, c)       vmlsq_f32(a, b, c)
# endif

# ifdef PARALLILOS_TYPE_F32
    template <>
    struct simd<float>
    {
      using vector_type = PARALLILOS_TYPE_F32;
      static constexpr const char* set  = PARALLILOS_SET_F32;
      static constexpr size_t alignment = PARALLILOS_ALIGNMENT_F32;
      static constexpr size_t size      = sizeof(vector_type) / sizeof(float);

      static constexpr size_t inline passes(const size_t n_elements)
      {
        return n_elements / size;
      }
      
      static constexpr size_t inline sequential(const size_t n_elements)
      {
        return n_elements - passes(n_elements)*size;
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
# endif

    // define the best SIMD intrinsics to use
# if defined(PARALLILOS_AVX512F)
#   define PARALLILOS_SET_F64                   "AVX512F"
#   define PARALLILOS_TYPE_F64                  __m512d
#   define PARALLILOS_ALIGNMENT_F64             64
#   define PARALLILOS_LOADU_F64(data)           _mm512_loadu_pd(data)
#   define PARALLILOS_LOADA_F64(data)           _mm512_load_pd(data)
#   define PARALLILOS_STOREU_F64(addr, data)    _mm512_storeu_pd((void*)addr, data)
#   define PARALLILOS_STOREA_F64(addr, data)    _mm512_store_pd((void*)addr, data)
#   define PARALLILOS_SETVAL_F64(value)         _mm512_set1_pd(value)
#   define PARALLILOS_SETZERO_F64()             _mm512_setzero_pd()
#   define PARALLILOS_MUL_F64(a, b)             _mm512_mul_pd(a, b)
#   define PARALLILOS_ADD_F64(a, b)             _mm512_add_pd(a, b)
#   define PARALLILOS_SUB_F64(a, b)             _mm512_sub_pd(a, b)
#   define PARALLILOS_DIV_F64(a, b)             _mm512_div_pd(a, b)
#   define PARALLILOS_SQRT_F64(a)               _mm512_sqrt_pd(a)
#   define PARALLILOS_ADDMUL_F64(a, b, c)       _mm512_fmadd_pd(b, c, a)
#   define PARALLILOS_SUBMUL_F64(a, b, c)       _mm512_fnmadd_pd(a, b, c)
# elif defined(PARALLILOS_FMA)
#   define PARALLILOS_SET_F64                   "AVX, FMA"
#   define PARALLILOS_TYPE_F64                  __m256d
#   define PARALLILOS_ALIGNMENT_F64             32
#   define PARALLILOS_LOADU_F64(data)           _mm256_loadu_pd(data)
#   define PARALLILOS_LOADA_F64(data)           _mm256_load_pd(data)
#   define PARALLILOS_STOREU_F64(addr, data)    _mm256_storeu_pd(addr, data)
#   define PARALLILOS_STOREA_F64(addr, data)    _mm256_store_pd(addr, data)
#   define PARALLILOS_SETVAL_F64(value)         _mm256_set1_pd(value)
#   define PARALLILOS_SETZERO_F64()             _mm256_setzero_pd()
#   define PARALLILOS_MUL_F64(a, b)             _mm256_mul_pd(a, b)
#   define PARALLILOS_ADD_F64(a, b)             _mm256_add_pd(a, b)
#   define PARALLILOS_SUB_F64(a, b)             _mm256_sub_pd(a, b)
#   define PARALLILOS_DIV_F64(a, b)             _mm256_div_pd(a, b)
#   define PARALLILOS_SQRT_F64(a)               _mm256_sqrt_pd(a)
#   define PARALLILOS_ADDMUL_F64(a, b, c)       _mm256_fmadd_pd(b, c, a)
#   define PARALLILOS_SUBMUL_F64(a, b, c)       _mm256_fnmadd_pd(a, b, c)
# elif defined(PARALLILOS_AVX)
#   define PARALLILOS_SET_F64                   "AVX"
#   define PARALLILOS_TYPE_F64                  __m256d
#   define PARALLILOS_ALIGNMENT_F64             32
#   define PARALLILOS_LOADU_F64(data)           _mm256_loadu_pd(data)
#   define PARALLILOS_LOADA_F64(data)           _mm256_load_pd(data)
#   define PARALLILOS_STOREU_F64(addr, data)    _mm256_storeu_pd(addr, data)
#   define PARALLILOS_STOREA_F64(addr, data)    _mm256_store_pd(addr, data)
#   define PARALLILOS_SETVAL_F64(value)         _mm256_set1_pd(value)
#   define PARALLILOS_SETZERO_F64()             _mm256_setzero_pd()
#   define PARALLILOS_MUL_F64(a, b)             _mm256_mul_pd(a, b)
#   define PARALLILOS_ADD_F64(a, b)             _mm256_add_pd(a, b)
#   define PARALLILOS_SUB_F64(a, b)             _mm256_sub_pd(a, b)
#   define PARALLILOS_DIV_F64(a, b)             _mm256_div_pd(a, b)
#   define PARALLILOS_SQRT_F64(a)               _mm256_sqrt_pd(a)
#   define PARALLILOS_ADDMUL_F64(a, b, c)       _mm256_add_pd(a, _mm256_mul_pd(b, c))
#   define PARALLILOS_SUBMUL_F64(a, b, c)       _mm256_sub_pd(a, _mm256_mul_pd(b, c))
# elif defined(PARALLILOS_SSE2)
#   define PARALLILOS_SET_F64                   "SSE2"
#   define PARALLILOS_TYPE_F64                  __m128d
#   define PARALLILOS_ALIGNMENT_F64             16
#   define PARALLILOS_LOADU_F64(data)           _mm_loadu_pd(data)
#   define PARALLILOS_LOADA_F64(data)           _mm_load_pd(data)
#   define PARALLILOS_STOREU_F64(addr, data)    _mm_storeu_pd(addr, data)
#   define PARALLILOS_STOREA_F64(addr, data)    _mm_store_pd(addr, data)
#   define PARALLILOS_SETVAL_F64(value)         _mm_set1_pd(value)
#   define PARALLILOS_SETZERO_F64()             _mm_setzero_pd()
#   define PARALLILOS_MUL_F64(a, b)             _mm_mul_pd(a, b)
#   define PARALLILOS_ADD_F64(a, b)             _mm_add_pd(a, b)
#   define PARALLILOS_SUB_F64(a, b)             _mm_sub_pd(a, b)
#   define PARALLILOS_DIV_F64(a, b)             _mm_div_pd(a, b)
#   define PARALLILOS_SQRT_F64(a)               _mm_sqrt_pd(a)
#   define PARALLILOS_ADDMUL_F64(a, b, c)       _mm_add_pd(a, _mm_mul_pd(b, c))
#   define PARALLILOS_SUBMUL_F64(a, b, c)       _mm_sub_pd(a, _mm_mul_pd(b, c))
# elif defined(PARALLILOS_NEON) || defined(PARALLILOS_NEON64)
#   define PARALLILOS_SET_F64                   "NEON"
#   define PARALLILOS_TYPE_F64                  float64x4_t
#   define PARALLILOS_ALIGNMENT_F64             0
#   define PARALLILOS_LOADU_F64(data)           vld1q_f64(data)
#   define PARALLILOS_LOADA_F64(data)           vld1q_f64(data)
#   define PARALLILOS_STOREU_F64(addr, data)    vst1q_f64(addr, data)
#   define PARALLILOS_STOREA_F64(addr, data)    vst1q_f64(addr, data)
#   define PARALLILOS_SETVAL_F64(value)         vdupq_n_f64(value)
#   define PARALLILOS_SETZERO_F64()             vdupq_n_f64(0.0)
#   define PARALLILOS_MUL_F64(a, b)             vmulq_f64(a, b)
#   define PARALLILOS_ADD_F64(a, b)             vaddq_f64(a, b)
#   define PARALLILOS_SUB_F64(a, b)             vsubq_f64(a, b)
#   define PARALLILOS_DIV_F64(a, b)             vdivq_f64(a, b)
#   define PARALLILOS_SQRT_F64(a)               vsqrtq_f64(a)
#   define PARALLILOS_ADDMUL_F64(a, b, c)       vmlaq_f64(a, b, c)
#   define PARALLILOS_SUBMUL_F64(a, b, c)       vmlsq_f64(a, b, c)
# endif

    // define a standard API to use SIMD intrinsics
# ifdef PARALLILOS_TYPE_F64
    template<>
    struct simd<double>
    {
      using vector_type = PARALLILOS_TYPE_F64;
      static constexpr const char* set  = PARALLILOS_SET_F64;
      static constexpr size_t alignment = PARALLILOS_ALIGNMENT_F64;
      static constexpr size_t size      = sizeof(vector_type) / sizeof(float);

      static constexpr size_t inline passes(const size_t n_elements)
      {
        return n_elements / size;
      }

      static constexpr size_t inline sequential(const size_t n_elements)
      {
        return n_elements - passes(n_elements)*size;
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
# endif

    struct Deleter 
    {
      template<PARALLILOS_ARITHMETIC(T)>
      void operator()(T* ptr)
      {
        free_array(ptr);
      }
    };
    
    template<PARALLILOS_ARITHMETIC(T)>
    Array<T> make_array(const size_t number_of_elements)
    {
      // early return
      if (number_of_elements == 0)
      {
        return Array<T>(nullptr);
      }

      // alignment requirement for simd
      constexpr size_t alignment = simd<T>::alignment;

      // nothing special needed
      if (alignment == 0)
      {
        return Array<T>(reinterpret_cast<T*>(std::malloc(number_of_elements * sizeof(T))));
      }
#   if __cplusplus >= 201703L
      else
      {
        return Array<T>(reinterpret_cast<T*>(std::aligned_alloc(alignment, number_of_elements * sizeof(T))));
      }
#   else

      // allocate
      void* memory_block = std::malloc(number_of_elements * sizeof(T) + alignment);
      
      // allocation failure
      if (memory_block == nullptr)
      {
        return Array<T>(nullptr);
      }

      // align on alignement boundary
      void* aligned_memory_block = reinterpret_cast<void*>((uintptr_t(memory_block) + alignment) & ~(alignment - 1));

      // bookkeeping of original memory block
      reinterpret_cast<void**>(aligned_memory_block)[-1] = memory_block;

      return Array<T>(reinterpret_cast<T*>(aligned_memory_block));
#   endif
    }

    template<PARALLILOS_ARITHMETIC(T)>
    void free_array(T* addr)
    {
#   if __cplusplus < 201703L
      if (simd<T>::alignment && addr)
      {
        std::free(reinterpret_cast<void**>(addr)[-1]);
        return;
      }
#   endif

      std::free(addr);
    }

    template<PARALLILOS_ARITHMETIC(T)>
    bool is_aligned(const T* addr)
    {
      return (uintptr_t(addr) & (simd<T>::alignment - 1)) == 0;
    }

# undef PARALLILOS_TYPE_WARNING
# undef PARALLILOS_ARITHMETIC
# undef PARALLILOS_VECTOR_OF

# undef PARALLILOS_TYPE_F32
# undef PARALLILOS_ALIGNMENT_F32
# undef PARALLILOS_LOADU_F32
# undef PARALLILOS_LOADA_F32
# undef PARALLILOS_STOREU_F32
# undef PARALLILOS_STOREA_F32
# undef PARALLILOS_SETVAL_F32
# undef PARALLILOS_SETZERO_F32
# undef PARALLILOS_ADD_F32
# undef PARALLILOS_MUL_F32
# undef PARALLILOS_SUB_F32
# undef PARALLILOS_DIV_F32
# undef PARALLILOS_SQRT_F32
# undef PARALLILOS_ADDMUL_F32
# undef PARALLILOS_SUBMUL_F32
  //
# undef PARALLILOS_TYPE_F64
# undef PARALLILOS_ALIGNMENT_F64
# undef PARALLILOS_LOADU_F64
# undef PARALLILOS_LOADA_F64
# undef PARALLILOS_STOREU_F64
# undef PARALLILOS_STOREA_F64
# undef PARALLILOS_SETVAL_F64
# undef PARALLILOS_SETZERO_F64
# undef PARALLILOS_ADD_F64
# undef PARALLILOS_MUL_F64
# undef PARALLILOS_SUB_F64
# undef PARALLILOS_DIV_F64
# undef PARALLILOS_SQRT_F64
# undef PARALLILOS_ADDMUL_F64
# undef PARALLILOS_SUBMUL_F64
  //
# undef PARALLILOS_COMPILER_SUPPORTS_SSE
# undef PARALLILOS_COMPILER_SUPPORTS_AVX
# undef PARALLILOS_COMPILER_SUPPORTS_SVML
# undef PARALLILOS_COMPILER_SUPPORTS_NEON
  }
}
#endif