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
  namespace Version
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

#if defined(PARALLILOS_WARNINGS)
  namespace Logging
  {
    template<typename T>
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

  namespace Backend
  {
    template<typename T, size_t VS, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    class SIMD
    {
    public:
      SIMD() = delete;
      static constexpr size_t size = VS / sizeof(T);
    private:
      class Parallel
      {
      public:
        inline explicit Parallel(const size_t n_elements) noexcept :
          current_index{0},
          passes_left{size ? (n_elements / size) : 0},
          passes{passes_left}
        {}
        inline size_t operator*() noexcept               {return current_index;}
        inline void operator++() noexcept                {--passes_left, current_index += size;}
        inline bool operator!=(const Parallel&) noexcept {return passes_left;}
        inline Parallel& begin() noexcept                {return *this;}
        inline Parallel end() noexcept                   {return Parallel{0};}
      private:
        size_t current_index;
        size_t passes_left;
      public:
        const size_t passes;
      };

      class Sequential
      {
      public:
        inline explicit Sequential(const size_t n_elements) noexcept :
          current_index{size ? ((n_elements / size) * size) : 0},
          passes_left{n_elements - current_index},
          passes{passes_left}
        {}
        inline size_t operator*() noexcept                 {return current_index;}
        inline void operator++() noexcept                  {--passes_left, ++current_index;}
        inline bool operator!=(const Sequential&) noexcept {return passes_left;}
        inline Sequential& begin() noexcept                {return *this;}
        inline Sequential end() noexcept                   {return Sequential{0};}
      private:
        size_t current_index;
        size_t passes_left;
      public:
        const size_t passes;
      };
    public:
      static inline Parallel parallel(const size_t n_elements)
      {
        return Parallel(n_elements);
      }
      
      static inline Sequential sequential(const size_t n_elements)
      {
        return Sequential(n_elements);
      }
    };
  }


// --Parallilos library: frontend forward declarations----------------------------------------------
  inline namespace Frontend
  {
    // custom deleter which invokes free_array
    struct Deleter;

    template<typename T>
    using Array = std::unique_ptr<T[], Deleter>;

    // SIMD aligned memory allocation
    template<typename T>
    inline Array<T> make_array(const size_t number_of_elements);

    // SIMD aligned memory deallocation
    template<typename T>
    inline void free_array(T* addr);

    // check if an address is aligned for SIMD
    template<typename T, typename... Tn>
    inline bool is_aligned(const T* addr, const Tn*... addrn);
    inline bool is_aligned();

    // unsupported type fallback
    template<typename T>
    struct SIMD : public Backend::SIMD<T, 0>
    {
      using type = T;
      static constexpr const char* const set = "no SIMD instruction set used for this type";
      static constexpr size_t alignment = 0;
    };
 
    // T = type, V = vector type, A = alignment, S = sets used
    #define PARALLILOS_MAKE_SIMD(T, V, A, S)              \
      template <>                                         \
      struct SIMD<T> : public Backend::SIMD<T, sizeof(V)> \
      {                                                   \
        using type = V;                                   \
        static constexpr const char* const set = S;       \
        static constexpr size_t alignment = A;            \
      }

    // load a vector from unaligned data
    template<typename T>
    PARALLILOS_INLINE auto simd_loadu(const T* data) -> typename SIMD<T>::type
    {
      PARALLILOS_TYPE_WARNING(T);
      return *data;
    }

    // load a vector from aligned data
    template<typename T>
    PARALLILOS_INLINE auto simd_loada(const T* data) -> typename SIMD<T>::type
    {
      PARALLILOS_TYPE_WARNING(T);
      return *data;
    }

    // store a vector into unaligned memory
    template<typename T>
    PARALLILOS_INLINE void simd_storeu(T* addr, typename SIMD<T>::type data)
    {
      PARALLILOS_TYPE_WARNING(T, typename SIMD<T>::type);
      *addr = data;
    }

    // store a vector into aligned memory
    template<typename T>
    PARALLILOS_INLINE void simd_storea(T* addr, typename SIMD<T>::type data)
    {
      PARALLILOS_TYPE_WARNING(T, typename SIMD<T>::type);
      *addr = data;
    }

    // load a vector with zeros
    template<typename T>
    PARALLILOS_INLINE auto simd_setzero() -> typename SIMD<T>::type
    {
      PARALLILOS_TYPE_WARNING(T);
      return 0;
    }

    // load a vector with a specific value
    template<typename T>
    PARALLILOS_INLINE auto simd_setval(const T value) -> typename SIMD<T>::type
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
# if defined(PARALLILOS_COMPILER_SUPPORTS_AVX)
#   undef PARALLILOS_COMPILER_SUPPORTS_AVX
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
#   undef PARALLILOS_COMPILER_SUPPORTS_SSE
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
#   undef PARALLILOS_COMPILER_SUPPORTS_NEON
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
#   undef PARALLILOS_COMPILER_SUPPORTS_SVML
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
# if defined(PARALLILOS_AVX512F)
    static_assert(sizeof(float) == 4, "float must be 32 bit");
#   define PARALLILOS_F32
    PARALLILOS_MAKE_SIMD(float, __m512, 64, "AVX512F");
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
    static_assert(sizeof(float) == 4, "float must be 32 bit");
#   define PARALLILOS_F32
    PARALLILOS_MAKE_SIMD(float, __m256, 32, "AVX, FMA");
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
    static_assert(sizeof(float) == 4, "float must be 32 bit");
#   define PARALLILOS_F32
    PARALLILOS_MAKE_SIMD(float, __m256, 32, "AVX");
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
    static_assert(sizeof(float) == 4, "float must be 32 bit");
#   define PARALLILOS_F32
    PARALLILOS_MAKE_SIMD(float, __m128, 16, "SSE");
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
    static_assert(sizeof(float) == 4, "float must be 32 bit");
#   define PARALLILOS_F32
    PARALLILOS_MAKE_SIMD(float, float32x4_t, 0, "NEON");
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

# ifdef PARALLILOS_F32
    template<>
    PARALLILOS_INLINE auto simd_setzero<float>() -> SIMD<float>::type
    {
      return PARALLILOS_SETZERO_F32();
#   undef PARALLILOS_SETZERO_F32
    }

    template <>
    PARALLILOS_INLINE auto simd_loadu(const float* data) -> SIMD<float>::type
    {
      return PARALLILOS_LOADU_F32(data);
#   undef PARALLILOS_LOADU_F32
    }

    template <>
    PARALLILOS_INLINE auto simd_loada(const float* data) -> SIMD<float>::type
    {
      return PARALLILOS_LOADA_F32(data);
#   undef PARALLILOS_LOADA_F32
    }

    template <>
    PARALLILOS_INLINE void simd_storeu(float* addr, SIMD<float>::type data)
    {
      PARALLILOS_STOREU_F32(addr, data);
#   undef PARALLILOS_STOREU_F32
    }

    template <>
    PARALLILOS_INLINE void simd_storea(float* addr, SIMD<float>::type data)
    {
      PARALLILOS_STOREA_F32(addr, data);
#   undef PARALLILOS_STOREA_F32
    }

    template <>
    PARALLILOS_INLINE auto simd_setval(const float value) -> SIMD<float>::type
    {
      return PARALLILOS_SETVAL_F32(value);
#   undef PARALLILOS_SETVAL_F32
    }

    template <>
    PARALLILOS_INLINE auto simd_add(SIMD<float>::type a, SIMD<float>::type b) -> SIMD<float>::type
    {
      return PARALLILOS_ADD_F32(a, b);
#   undef PARALLILOS_ADD_F32
    }

    template <>
    PARALLILOS_INLINE auto simd_mul(SIMD<float>::type a, SIMD<float>::type b) -> SIMD<float>::type
    {
      return PARALLILOS_MUL_F32(a, b);
#   undef PARALLILOS_MUL_F32
    }

    template <>
    PARALLILOS_INLINE auto simd_sub(SIMD<float>::type a, SIMD<float>::type b) -> SIMD<float>::type
    {
      return PARALLILOS_SUB_F32(a, b);
#   undef PARALLILOS_SUB_F32
    }

    template <>
    PARALLILOS_INLINE auto simd_div(SIMD<float>::type a, SIMD<float>::type b) -> SIMD<float>::type
    {
      return PARALLILOS_DIV_F32(a, b);
#   undef PARALLILOS_DIV_F32
    }

    template <>
    PARALLILOS_INLINE auto simd_sqrt(SIMD<float>::type a) -> SIMD<float>::type
    {
      return PARALLILOS_SQRT_F32(a);
#   undef PARALLILOS_SQRT_F32
    }

    template <>
    PARALLILOS_INLINE auto simd_addmul(SIMD<float>::type a, SIMD<float>::type b, SIMD<float>::type c) -> SIMD<float>::type
    {
      return PARALLILOS_ADDMUL_F32(a, b, c);
#   undef PARALLILOS_ADDMUL_F32
    }

    template <>
    PARALLILOS_INLINE auto simd_submul(SIMD<float>::type a, SIMD<float>::type b, SIMD<float>::type c) -> SIMD<float>::type
    {
      return PARALLILOS_SUBMUL_F32(a, b, c);
#   undef PARALLILOS_SUBMUL_F32
    }
# endif

# if defined(PARALLILOS_AVX512F)
#   define PARALLILOS_F64
    static_assert(sizeof(double) == 8, "float must be 64 bit");
    PARALLILOS_MAKE_SIMD(double, __m512d, 64, "AVX512F");
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
#   define PARALLILOS_F64
    static_assert(sizeof(float) == 8, "float must be 64 bit");
    PARALLILOS_MAKE_SIMD(double, __m256d, 32, "AVX, FMA");
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
#   define PARALLILOS_F64
    static_assert(sizeof(float) == 8, "float must be 64 bit");
    PARALLILOS_MAKE_SIMD(double, __m256d, 32, "AVX");
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
#   define PARALLILOS_F64
    static_assert(sizeof(float) == 8, "float must be 64 bit");
    PARALLILOS_MAKE_SIMD(double, __m128d, 16, "SSE2");
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
#   define PARALLILOS_F64
    static_assert(sizeof(float) == 8, "float must be 64 bit");
    PARALLILOS_MAKE_SIMD(double, float64x4_t, 0, "NEON");
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

# ifdef PARALLILOS_F64
    template<>
    PARALLILOS_INLINE auto simd_setzero<double>() -> SIMD<double>::type
    {
      return PARALLILOS_SETZERO_F64();
#   undef PARALLILOS_SETZERO_F64
    }

    template <>
    PARALLILOS_INLINE auto simd_loadu(const double* data) -> SIMD<double>::type
    {
      return PARALLILOS_LOADU_F64(data);
#   undef PARALLILOS_LOADU_F64
    }

    template <>
    PARALLILOS_INLINE auto simd_loada(const double* data) -> SIMD<double>::type
    {
      return PARALLILOS_LOADA_F64(data);
#   undef PARALLILOS_LOADA_F64
    }

    template <> 
    PARALLILOS_INLINE void simd_storeu(double* addr, SIMD<double>::type data)
    {
      PARALLILOS_STOREU_F64(addr, data);
#   undef PARALLILOS_STOREU_F64
    }

    template <>
    PARALLILOS_INLINE void simd_storea(double* addr, SIMD<double>::type data)
    {
      PARALLILOS_STOREA_F64(addr, data);
#   undef PARALLILOS_STOREA_F64
    }

    template <>
    PARALLILOS_INLINE auto simd_setval(const double value) -> SIMD<double>::type
    {
      return PARALLILOS_SETVAL_F64(value);
#   undef PARALLILOS_SETVAL_F64
    }

    template <>
    PARALLILOS_INLINE auto simd_add(const SIMD<double>::type a, const SIMD<double>::type b) -> SIMD<double>::type
    {
      return PARALLILOS_ADD_F64(a, b);
#   undef PARALLILOS_ADD_F64
    }

    template <>
    PARALLILOS_INLINE auto simd_mul(SIMD<double>::type a, SIMD<double>::type b) -> SIMD<double>::type
    {
      return PARALLILOS_MUL_F64(a, b);
#   undef PARALLILOS_MUL_F64
    }

    template <>
    PARALLILOS_INLINE auto simd_sub(SIMD<double>::type a, SIMD<double>::type b) -> SIMD<double>::type
    {
      return PARALLILOS_SUB_F64(a, b);
#   undef PARALLILOS_SUB_F64
    }

    template <>
    PARALLILOS_INLINE auto simd_div(SIMD<double>::type a, SIMD<double>::type b) -> SIMD<double>::type
    {
      return PARALLILOS_DIV_F64(a, b);
#   undef PARALLILOS_DIV_F64
    }

    template <>
    PARALLILOS_INLINE auto simd_sqrt(SIMD<double>::type a) -> SIMD<double>::type
    {
      return PARALLILOS_SQRT_F64(a);
#   undef PARALLILOS_SQRT_F64
    }

    template <>
    PARALLILOS_INLINE auto simd_addmul(SIMD<double>::type a, SIMD<double>::type b, SIMD<double>::type c) -> SIMD<double>::type
    {
      return PARALLILOS_ADDMUL_F64(a, b, c);
#   undef PARALLILOS_ADDMUL_F64
    }

    template <>
    PARALLILOS_INLINE auto simd_submul(SIMD<double>::type a, SIMD<double>::type b, SIMD<double>::type c) -> SIMD<double>::type
    {
      return PARALLILOS_SUBMUL_F64(a, b, c);
#   undef PARALLILOS_SUBMUL_F64
    }
# endif

# if defined(PARALLILOS_AVX512F)
#   define PARALLILOS_I32
    static_assert(sizeof(int32_t) == 4, "int32_t must be 32 bit");
    PARALLILOS_MAKE_SIMD(int32_t, __m512i, 64, "AVX512F");
#   define PARALLILOS_LOADU_I32(data)           _mm512_loadu_si512(data)
#   define PARALLILOS_LOADA_I32(data)           _mm512_load_si512(data)
#   define PARALLILOS_STOREU_I32(addr, data)    _mm512_storeu_si512((void*)addr, data)
#   define PARALLILOS_STOREA_I32(addr, data)    _mm512_store_si512((void*)addr, data)
#   define PARALLILOS_SETVAL_I32(value)         _mm512_set1_epi32(value)
#   define PARALLILOS_SETZERO_I32()             _mm512_setzero_si512()
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
#   define PARALLILOS_I32
    PARALLILOS_MAKE_SIMD(int32_t, __m256i, 32, "AVX2, AVX");
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
#   define PARALLILOS_I32
    PARALLILOS_MAKE_SIMD(int32_t, __m128i, 16, "SSE4.1, SSE2, SSE");
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
#   define PARALLILOS_I32
    PARALLILOS_MAKE_SIMD(int32_t, __m128i, 16, "SSE2, SSE");
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
  
# ifdef PARALLILOS_I32
    template<>
    PARALLILOS_INLINE auto simd_setzero<int32_t>(void) -> SIMD<int32_t>::type
    {
      return PARALLILOS_SETZERO_I32();
#   undef PARALLILOS_SETZERO_I32
    }

    template <>
    PARALLILOS_INLINE auto simd_loadu(const int32_t* data) -> SIMD<int32_t>::type
    {
      return PARALLILOS_LOADU_I32(data);
#   undef PARALLILOS_LOADU_I32
    }

    template <>
    PARALLILOS_INLINE auto simd_loada(const int32_t* data) -> SIMD<int32_t>::type
    {
      return PARALLILOS_LOADA_I32(data);
#   undef PARALLILOS_LOADA_I32
    }

    template <>
    PARALLILOS_INLINE void simd_storeu(int32_t* addr, SIMD<int32_t>::type data)
    {
      PARALLILOS_STOREU_I32(addr, data);
#   undef PARALLILOS_STOREU_I32
    }

    template <>
    PARALLILOS_INLINE void simd_storea(int32_t* addr, SIMD<int32_t>::type data)
    {
      PARALLILOS_STOREA_I32(addr, data);
#   undef PARALLILOS_STOREA_I32
    }

    template <>
    PARALLILOS_INLINE auto simd_setval(const int32_t value) -> SIMD<int32_t>::type
    {
      return PARALLILOS_SETVAL_I32(value);
#   undef PARALLILOS_SETVAL_I32
    }

    template <>
    PARALLILOS_INLINE auto simd_add(SIMD<int32_t>::type a, SIMD<int32_t>::type b) -> SIMD<int32_t>::type
    {
      return PARALLILOS_ADD_I32(a, b);
#   undef PARALLILOS_ADD_I32
    }

    template <>
    PARALLILOS_INLINE auto simd_mul(SIMD<int32_t>::type a, SIMD<int32_t>::type b) -> SIMD<int32_t>::type
    {
      return PARALLILOS_MUL_I32(a, b);
#   undef PARALLILOS_MUL_I32
    }

    template <>
    PARALLILOS_INLINE auto simd_sub(SIMD<int32_t>::type a, SIMD<int32_t>::type b) -> SIMD<int32_t>::type
    {
      return PARALLILOS_SUB_I32(a, b);
#   undef PARALLILOS_SUB_I32
    }

    template <>
    PARALLILOS_INLINE auto simd_div(SIMD<int32_t>::type a, SIMD<int32_t>::type b) -> SIMD<int32_t>::type
    {
      return PARALLILOS_DIV_I32(a, b);
#   undef PARALLILOS_DIV_I32
    }

    template <>
    PARALLILOS_INLINE auto simd_sqrt(SIMD<int32_t>::type a) -> SIMD<int32_t>::type
    {
      return PARALLILOS_SQRT_I32(a);
#   undef PARALLILOS_SQRT_I32
    }

    template <>
    PARALLILOS_INLINE auto simd_addmul(SIMD<int32_t>::type a, SIMD<int32_t>::type b, SIMD<int32_t>::type c) -> SIMD<int32_t>::type
    {
      return PARALLILOS_ADDMUL_I32(a, b, c);
#   undef PARALLILOS_ADDMUL_I32
    }

    template <>
    PARALLILOS_INLINE auto simd_submul(SIMD<int32_t>::type a, SIMD<int32_t>::type b, SIMD<int32_t>::type c) -> SIMD<int32_t>::type
    {
      return PARALLILOS_SUBMUL_I32(a, b, c);
#   undef PARALLILOS_SUBMUL_I32
    }
# endif

    struct Deleter 
    {
      template<typename T>
      void operator()(T* ptr)
      {
        free_array(ptr);
      }
    };
    
    template<typename T>
    Array<T> make_array(const size_t number_of_elements)
    {
      // early return
      if (number_of_elements == 0)
      {
        return Array<T>(nullptr);
      }

      // alignment requirement for simd
      constexpr size_t alignment = SIMD<T>::alignment;

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

    template<typename T>
    void free_array(T* addr)
    {
#   if __cplusplus < 201703L
      if (SIMD<T>::alignment && addr)
      {
        std::free(reinterpret_cast<void**>(addr)[-1]);
        return;
      }
#   endif

      std::free(addr);
    }

    template<typename T, typename... Tn>
    bool is_aligned(const T* addr, const Tn*... addrn)
    {
      return ((uintptr_t(addr) & (SIMD<T>::alignment - 1)) == 0)  && is_aligned(addrn...);
    }

    bool is_aligned()
    {
      return true;
    }

# undef PARALLILOS_TYPE_WARNING
  }
}
#endif