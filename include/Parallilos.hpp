/*---author-------------------------------------------------------------------------------------------------------------

Justin Asselin (juste-injuste)
justin.asselin@usherbrooke.ca
https://github.com/juste-injuste/Parallilos

-----licence------------------------------------------------------------------------------------------------------------

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

-----description--------------------------------------------------------------------------------------------------------

Parallilos is a simple and lightweight C++11 (and newer) library that abstracts away SIMD usage to facilitate generic
parallelism.

-----inclusion guard--------------------------------------------------------------------------------------------------*/
#if not defined(PARALLILOS_HPP)
#if defined(__cplusplus) and (__cplusplus >= 201103L)
#define PARALLILOS_HPP
// --necessary standard libraries---------------------------------------------------------------------------------------
#include <cstddef>      // for size_t
#include <cstdint>      // for fixed-sized integers
#include <cmath>        // for std::sqrt
#include <cstdlib>      // for std::malloc, std::free, std::aligned_alloc
#include <ostream>      // for std::ostream
#include <iostream>     // for std::cerr
#include <type_traits>  // for std::is_arithmetic, std::is_integral, std::enable_if
// --supplementary standard libraries-----------------------------------------------------------------------------------
#if defined(__STDCPP_THREADS__) and not defined(PARALLILOS_NOT_THREADSAFE)
# define PARALLILOS_THREADSAFE
# include <mutex>       // for std::mutex, std::lock_guard
#endif
#if defined(PARALLILOS_LOGGING)
# include <type_traits> // for std::is_floating_point, std::is_unsigned, std::is_pointer, std::remove_pointer
# include <typeinfo>    // for typeid
# include <cstdio>      // for std::sprintf
#endif
// --Parallilos library-------------------------------------------------------------------------------------------------
#if defined(__AVX512F__)
# define PARALLILOS_AVX512F
# define PARALLILOS_SIMD_HEADER <immintrin.h>
#endif
#if defined(__AVX2__)
# define PARALLILOS_AVX2
# define PARALLILOS_SIMD_HEADER <immintrin.h>
#endif
#if defined(__FMA__)
# define PARALLILOS_FMA
# define PARALLILOS_SIMD_HEADER <immintrin.h>
#endif
#if defined(__AVX__)
# define PARALLILOS_AVX
# define PARALLILOS_SIMD_HEADER <immintrin.h>
#endif

#if defined(__SSE4_2__)
# define PARALLILOS_SSE4_2
# define PARALLILOS_SIMD_HEADER <immintrin.h>
#endif
#if defined(__SSE4_1__)
# define PARALLILOS_SSE4_1
# define PARALLILOS_SIMD_HEADER <immintrin.h>
#endif
#if defined(__SSSE3__)
# define PARALLILOS_SSSE3
# define PARALLILOS_SIMD_HEADER <immintrin.h>
#endif
#if defined(__SSE3__)
# define PARALLILOS_SSE3
# define PARALLILOS_SIMD_HEADER <immintrin.h>
#endif
#if defined(__SSE2__)
# define PARALLILOS_SSE2
# define PARALLILOS_SIMD_HEADER <immintrin.h>
#endif
#if defined(__SSE__)
# define PARALLILOS_SSE
# define PARALLILOS_SIMD_HEADER <immintrin.h>
#endif

// #if defined(__ARM_NEON) or defined(__ARM_NEON__)
// # if defined(__ARM_ARCH_64)
// #   define PARALLILOS_NEON64
// #   define PARALLILOS_SIMD_HEADER <arm64_neon.h>
// # else
// #   define PARALLILOS_NEON
// #   define PARALLILOS_SIMD_HEADER <arm_neon.h>
// # endif
// #endif

#if defined(PARALLILOS_SIMD_HEADER)
# if not defined(__OPTIMIZE__)
#   define __OPTIMIZE__
#   include PARALLILOS_SIMD_HEADER
# else
#   include PARALLILOS_SIMD_HEADER
# endif
#endif

namespace Parallilos
{  
  template<typename T>
  class SIMD;

  template<typename T>
  class Array;

  namespace Global
  {
    static std::ostream wrn{std::cerr.rdbuf()}; // warning ostream
    static std::ostream log{std::clog.rdbuf()}; // logging ostream
  }

  namespace Version
  {
    constexpr unsigned long MAJOR  = 000;
    constexpr unsigned long MINOR  = 001;
    constexpr unsigned long PATCH  = 000;
    constexpr unsigned long NUMBER = (MAJOR * 1000 + MINOR) * 1000 + PATCH;
  }
// ---------------------------------------------------------------------------------------------------------------------
  namespace _backend
  {
# if defined(__clang__)
#   define PARALLILOS_INLINE __attribute__((always_inline)) inline
#   if (__cplusplus >= 201703L) and defined(_LIBCPP_HAS_C11_FEATURES)
#     define PARALLILOS_HAS_ALIGNED_ALLOC
#   endif
# elif defined(__GNUC__)
#   define PARALLILOS_INLINE __attribute__((always_inline)) inline
#   if (__cplusplus >= 201703L) and defined(_GLIBCXX_HAVE_ALIGNED_ALLOC)
#     define PARALLILOS_HAS_ALIGNED_ALLOC
#   endif
# elif defined(__MINGW32__) or defined(__MINGW64__)
#   define PARALLILOS_INLINE __attribute__((always_inline)) inline
# elif defined(__apple_build_version__)
#   define PARALLILOS_INLINE __attribute__((always_inline)) inline
# elif defined(_MSC_VER)
#   define PARALLILOS_INLINE __forceinline
# elif defined(__INTEL_COMPILER)
#   define PARALLILOS_SVML
#   define PARALLILOS_INLINE __forceinline
# elif defined(__ARMCC_VERSION)
#   define PARALLILOS_INLINE __forceinline
# else
#   define PARALLILOS_INLINE inline
# endif

# if defined(__clang__) and (__clang_major__ >= 12)
#   define PARALLILOS_COLD [[unlikely]]
#   define PARALLILOS_HOT  [[likely]]
# elif defined(__GNUC__) and (__GNUC__ >= 9)
#   define PARALLILOS_COLD [[unlikely]]
#   define PARALLILOS_HOT  [[likely]]
# else
#   define PARALLILOS_COLD
#   define PARALLILOS_HOT
# endif

# if defined(PARALLILOS_THREADSAFE)
#   define PARALLILOS_THREADLOCAL     thread_local
#   define PARALLILOS_MAKE_MUTEX(...) static std::mutex __VA_ARGS__
#   define PARALLILOS_LOCK(MUTEX)     std::lock_guard<decltype(MUTEX)> _lock{MUTEX}
# else
#   define PARALLILOS_THREADLOCAL
#   define PARALLILOS_MAKE_MUTEX(...)
#   define PARALLILOS_LOCK(MUTEX)     void(0)
# endif

    template<size_t size>
    class _parallel
    {
    public:
      _parallel(const size_t n_elements) noexcept :
        current_index{0},
        passes_left{size ? (n_elements / size) : 0}
      {}
      size_t     operator*()                  noexcept {return current_index;}
      void       operator++()                 noexcept {--passes_left, current_index += size;}
      bool       operator!=(const _parallel&) noexcept {return passes_left;}
      _parallel& begin()                      noexcept {return *this;}
      _parallel  end()                        noexcept {return _parallel{0};}
    private:
      size_t current_index;
      size_t passes_left;
    public:
      const size_t passes = passes_left;
    };

    template<size_t size>
    class _sequential
    {
    public:
      _sequential(const size_t n_elements) noexcept :
        current_index{size ? ((n_elements / size) * size) : 0},
        passes_left{n_elements - current_index}
      {}
      size_t       operator*()                    noexcept {return current_index;}
      void         operator++()                   noexcept {--passes_left, ++current_index;}
      bool         operator!=(const _sequential&) noexcept {return passes_left;}
      _sequential& begin()                        noexcept {return *this;}
      _sequential  end()                          noexcept {return _sequential{0};}
    private:
      size_t current_index;
      size_t passes_left;
    public:
      const size_t passes = passes_left;
    };

# if defined(PARALLILOS_LOGGING)
    PARALLILOS_THREADLOCAL static char _typename_buffer[16];

    template<typename T> inline
    const char* _type_name()
    {
      if (std::is_floating_point<T>::value)
      {
        std::sprintf(_typename_buffer, "float%u", static_cast<unsigned>(sizeof(T) * 8));
      }
      else if (std::is_unsigned<T>::value)
      {
        std::sprintf(_typename_buffer, "uint%u", static_cast<unsigned>(sizeof(T) * 8));
      }
      else if (std::is_pointer<T>::value)
      {
        _type_name<typename std::remove_pointer<T>::type>();
        char* char_ptr = _typename_buffer;
        while (*char_ptr != '\0') { ++char_ptr; }
        char_ptr[0] = '*';
        char_ptr[1] = '\0';
      }
      else
      {
        std::sprintf(_typename_buffer, "int%u", static_cast<unsigned>(sizeof(T) * 8));
      }

      return _typename_buffer;
    }

    PARALLILOS_THREADLOCAL static char _log_buffer[256];
    PARALLILOS_MAKE_MUTEX(_log_mtx);
    
#   define PARALLILOS_LOG(...)                                               \
      [&](const char* caller){                                               \
        sprintf(_backend::_log_buffer, __VA_ARGS__);                         \
        PARALLILOS_LOCK(_backend::_log_mtx);                                 \
        Global::log << caller << ": " << _backend::_log_buffer << std::endl; \
      }(__func__)
# else
#   define PARALLILOS_LOG(...) void(0)
# endif

    template<typename T>
    using _if_integral = typename std::enable_if<std::is_integral<T>::value>::type;
  }
// ---------------------------------------------------------------------------------------------------------------------
  template<typename T>
  class SIMD
  {
  static_assert(std::is_arithmetic<T>::value, "T in SIMD<T> must be an arithmetic type");
  public:
    static constexpr size_t size      = 0;
    static constexpr size_t alignment = 0;
    using Type = T;
    using Mask = bool;
    static constexpr const char* set = "no SIMD instruction set used for this type";

    static _backend::_parallel<size> parallel(const size_t n_elements) noexcept
    {
      return _backend::_parallel<size>{n_elements};
    }

    static _backend::_sequential<size> sequential(const size_t n_elements) noexcept
    {
      return _backend::_sequential<size>{n_elements};
    }
    
    SIMD() = delete;
  };

  // load a vector from unaligned data
  template<typename T> inline
  typename SIMD<T>::Type simd_loadu(const T data[]) noexcept
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T*>());
    return *data;
  }

  // load a vector from aligned data
  template<typename T> inline
  typename SIMD<T>::Type simd_loada(const T data[]) noexcept
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T*>());
    return *data;
  }

  // store a vector into unaligned memory
  template<typename T> inline
  void simd_storeu(T addr[], const typename SIMD<T>::Type& data)
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T*>());
    *addr = data;
  }

  // store a vector into aligned memory
  template<typename T> inline
  void simd_storea(T addr[], const typename SIMD<T>::Type& data)
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T*>());
    *addr = data;
  }

  // load a vector with zeros
  template<typename T> inline
  typename SIMD<T>::Type simd_setzero() noexcept
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return 0;
  }

  // load a vector with a specific value
  template<typename T> inline
  typename SIMD<T>::Type simd_setval(const T value) noexcept
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return value;
  }

  // [a] + [b]
  template<typename T> inline
  T simd_add(T a, T b)
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a + b;
  }

  // [a] * [b]
  template<typename T> inline
  T simd_mul(T a, T b)
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a * b;
  }

  // [a] - [b]
  template<typename T> inline
  T simd_sub(T a, T b)
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a - b;
  }

  // [a] / [b]
  template<typename T> inline
  T simd_div(T a, T b)
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a / b;
  }

  // sqrt([a])
  template<typename T> inline
  T simd_sqrt(T a)
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return std::sqrt(a);
  }

  // [a] + ([b] * [c])
  template<typename T> inline
  T simd_addmul(T a, T b, T c)
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a + b * c;
  }

  // [a] - ([b] * [c])
  template<typename T> inline
  T simd_submul(T a, T b, T c)
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a - b * c;
  }

  // [a] == [b]
  template<typename T> inline
  bool simd_eq(T a, T b) noexcept
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a == b;
  }

  // [a] != [b]
  template<typename T> inline
  bool simd_neq(T a, T b) noexcept
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a != b;
  }

  // [a] > [b]
  template<typename T> inline
  bool simd_gt(T a, T b) noexcept
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a > b;
  }

  // [a] >= [b]
  template<typename T> inline
  bool simd_gte(T a, T b) noexcept
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a >= b;
  }

  // [a] < [b]
  template<typename T> inline
  bool simd_lt(T a, T b) noexcept
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a < b;
  }

  // [a] <= [b]
  template<typename T> inline
  bool simd_lte(T a, T b) noexcept
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a <= b;
  }

  inline namespace Bitwise
  {
    // ![a]
    template<typename T, typename = _backend::_if_integral<T>> inline
    T simd_not(T a) noexcept
    {
      PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return !a;
    }

    // [a] & [b]
    template<typename T, typename = _backend::_if_integral<T>> inline
    T simd_and(T a, T b) noexcept
    {
      PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return a & b;
    }

    // !([a] & [b])
    template<typename T, typename = _backend::_if_integral<T>> inline
    T simd_nand(T a, T b) noexcept
    {
      PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return !(a & b);
    }

    // [a] | [b]
    template<typename T, typename = _backend::_if_integral<T>> inline
    T simd_or(T a, T b) noexcept
    {
      PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return a | b;
    }

    // !([a] | [b])
    template<typename T, typename = _backend::_if_integral<T>> inline
    T simd_nor(T a, T b) noexcept
    {
      PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return !(a | b);
    }

    // [a] ^ [b]
    template<typename T, typename = _backend::_if_integral<T>> inline
    T simd_xor(T a, T b) noexcept
    {
      PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return a ^ b;
    }

    // !([a] ^ [b])
    template<typename T, typename = _backend::_if_integral<T>> inline
    T simd_xnor(T a, T b) noexcept
    {
      PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return !(a ^ b);
    }
  }

  // abs([a])
  template<typename T> inline
  T simd_abs(T a) noexcept
  {
    PARALLILOS_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return std::abs(a);
  }

  template<typename T>
  class Array final
  {
  static_assert(std::is_arithmetic<T>::value, "T in Array<T> must be an arithmetic type");
  public:
    const T* data(const size_t k = 0)   const noexcept { return array + k; }
    T*       data(const size_t k = 0)         noexcept { return array + k; }
    size_t   size()                     const noexcept { return numel; }
    T        operator[](const size_t k) const noexcept { return array[k]; }
    T&       operator[](const size_t k)       noexcept { return array[k]; }
    operator T*()                             noexcept { return array; }
    T*       release()                        noexcept
    {
      T* data = array;
      array   = nullptr;
      numel   = 0;
      return data;
    }

    Array(const size_t number_of_elements) noexcept :
      array([number_of_elements]() -> T* {
        if ((number_of_elements == 0)) PARALLILOS_COLD
        {
          return nullptr;
        }

        if (SIMD<T>::alignment == 0) PARALLILOS_COLD
        {
          return reinterpret_cast<T*>(std::malloc(number_of_elements * sizeof(T)));
        }

#     if defined(PARALLILOS_HAS_ALIGNED_ALLOC)
        return reinterpret_cast<T*>(std::aligned_alloc(SIMD<T>::alignment, number_of_elements * sizeof(T)));
#     else
        void* memory_block = std::malloc(number_of_elements * sizeof(T) + SIMD<T>::alignment);

        if (memory_block == nullptr) PARALLILOS_COLD
        {
          return nullptr;
        }

        // align on alignement boundary
        auto  aligned_address      = (reinterpret_cast<uintptr_t>(memory_block) + SIMD<T>::alignment) & ~(SIMD<T>::alignment - 1);
        void* aligned_memory_block = reinterpret_cast<void*>(aligned_address);

        // bookkeeping of original memory block
        reinterpret_cast<void**>(aligned_memory_block)[-1] = memory_block;

        return reinterpret_cast<T*>(aligned_memory_block);
#     endif
      }()),
      numel(array == nullptr ? 0 : number_of_elements)
    {
      PARALLILOS_LOG("created array of size %u", unsigned(numel));
    }

    Array(const std::initializer_list<T> initializer_list) noexcept :
      Array(initializer_list.size())
    {
      if (numel != 0) PARALLILOS_HOT
      {
        size_t k = 0;
        for (T value : initializer_list)
        {
          array[k++] = value;
        }
      }
    }
    
    Array(Array&& other) noexcept :
      array(other.array),
      numel(other.numel)
    {
      PARALLILOS_LOG("moved array of size %u", unsigned(numel));
      other.array = nullptr;
      other.numel = 0;
    }

    Array(const Array&) = delete; // copying is disallowed

    ~Array()
    {
      if (array != nullptr) PARALLILOS_HOT
      {
#   if defined(PARALLILOS_HAS_ALIGNED_ALLOC)
        std::free(array);
#   else
        if (SIMD<T>::alignment != 0) PARALLILOS_HOT
        {
          std::free(reinterpret_cast<void**>(array)[-1]);
        }
        else std::free(array);
#   endif
      }
      PARALLILOS_LOG("freed used memory");
    }

    // interpret aligned array as a vector, k is an offset in elements into the array
    typename SIMD<T>::Type& as_vector(const size_t k) noexcept
    {
      return reinterpret_cast<typename SIMD<T>::Type&>(array[k]);
    }
  private:
    T*     array;
    size_t numel;
  };
// ---------------------------------------------------------------------------------------------------------------------
  namespace _backend
  {
    // T = type, V = vector type, M = mask type, A = alignment, S = sets used
#   define PARALLILOS_MAKE_SIMD_SPECIALIZATION(T, V, M, A, S)                                  \
      template<>                                                                               \
      class SIMD<T>                                                                            \
      {                                                                                        \
      static_assert(std::is_arithmetic<T>::value, "T in SIMD<T> must be an arithmetic type");  \
      public:                                                                                  \
        static constexpr size_t size      = sizeof(V)/sizeof(T);                               \
        static constexpr size_t alignment = A;                                                 \
        using Type = V;                                                                        \
        using Mask = M;                                                                        \
        static constexpr const char* set = S;                                                  \
        static _backend::_parallel<size> parallel(const size_t n_elements) noexcept            \
        { return _backend::_parallel<size>{n_elements}; }                                      \
        static _backend::_sequential<size> sequential(const size_t n_elements) noexcept        \
        { return _backend::_sequential<size>{n_elements}; }                                    \
        SIMD() = delete;                                                                       \
      };
  }
#if defined(PARALLILOS_AVX512F)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define PARALLILOS_F32
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(float, __m512, __mmask16, 64, "AVX512F");
# define PARALLILOS_F32_LOADU(data)           _mm512_loadu_ps(data)
# define PARALLILOS_F32_LOADA(data)           _mm512_load_ps(data)
# define PARALLILOS_F32_STOREU(addr, data)    _mm512_storeu_ps((Toid*)addr, data)
# define PARALLILOS_F32_STOREA(addr, data)    _mm512_store_ps((Toid*)addr, data)
# define PARALLILOS_F32_SETVAL(Talue)         _mm512_set1_ps(Talue)
# define PARALLILOS_F32_SETZERO()             _mm512_setzero_ps()
# define PARALLILOS_F32_MUL(a, b)             _mm512_mul_ps(a, b)
# define PARALLILOS_F32_ADD(a, b)             _mm512_add_ps(a, b)
# define PARALLILOS_F32_SUB(a, b)             _mm512_sub_ps(a, b)
# define PARALLILOS_F32_DIV(a, b)             _mm512_div_ps(a, b)
# define PARALLILOS_F32_SQRT(a)               _mm512_sqrt_ps(a)
# define PARALLILOS_F32_ADDMUL(a, b, c)       _mm512_fmadd_ps(b, c, a)
# define PARALLILOS_F32_SUBMUL(a, b, c)       _mm512_fnmadd_ps(a, b, c)
# define PARALLILOS_F32_EQ(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_EQ_UQ)
# define PARALLILOS_F32_NEQ(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_NEQ_UQ)
# define PARALLILOS_F32_GT(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ)
# define PARALLILOS_F32_GTE(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ)
# define PARALLILOS_F32_LT(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ)
# define PARALLILOS_F32_LTE(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ)

# define PARALLILOS_F32_ABS(a)                _mm512_abs_ps(a)
#elif defined(PARALLILOS_FMA)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define PARALLILOS_F32
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(float, __m256, __m256, 32, "AVX, FMA");
# define PARALLILOS_F32_LOADU(data)           _mm256_loadu_ps(data)
# define PARALLILOS_F32_LOADA(data)           _mm256_load_ps(data)
# define PARALLILOS_F32_STOREU(addr, data)    _mm256_storeu_ps(addr, data)
# define PARALLILOS_F32_STOREA(addr, data)    _mm256_store_ps(addr, data)
# define PARALLILOS_F32_SETVAL(Talue)         _mm256_set1_ps(Talue)
# define PARALLILOS_F32_SETZERO()             _mm256_setzero_ps()
# define PARALLILOS_F32_MUL(a, b)             _mm256_mul_ps(a, b)
# define PARALLILOS_F32_ADD(a, b)             _mm256_add_ps(a, b)
# define PARALLILOS_F32_SUB(a, b)             _mm256_sub_ps(a, b)
# define PARALLILOS_F32_DIV(a, b)             _mm256_div_ps(a, b)
# define PARALLILOS_F32_SQRT(a)               _mm256_sqrt_ps(a)
# define PARALLILOS_F32_ADDMUL(a, b, c)       _mm256_fmadd_ps(b, c, a)
# define PARALLILOS_F32_SUBMUL(a, b, c)       _mm256_fnmadd_ps(a, b, c)
# define PARALLILOS_F32_EQ(a, b)              _mm256_cmp_ps(a, b, _CMP_EQ_UQ)
# define PARALLILOS_F32_NEQ(a, b)             _mm256_cmp_ps(a, b, _CMP_NEQ_UQ)
# define PARALLILOS_F32_GT(a, b)              _mm256_cmp_ps(a, b, _CMP_GT_OQ)
# define PARALLILOS_F32_GTE(a, b)             _mm256_cmp_ps(a, b, _CMP_GE_OQ)
# define PARALLILOS_F32_LT(a, b)              _mm256_cmp_ps(a, b, _CMP_LT_OQ)
# define PARALLILOS_F32_LTE(a, b)             _mm256_cmp_ps(a, b, _CMP_LE_OQ)
# define PARALLILOS_F32_ABS(a)                _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a)
#elif defined(PARALLILOS_AVX)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define PARALLILOS_F32
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(float, __m256, __m256, 32, "AVX");
# define PARALLILOS_F32_LOADU(data)           _mm256_loadu_ps(data)
# define PARALLILOS_F32_LOADA(data)           _mm256_load_ps(data)
# define PARALLILOS_F32_STOREU(addr, data)    _mm256_storeu_ps(addr, data)
# define PARALLILOS_F32_STOREA(addr, data)    _mm256_store_ps(addr, data)
# define PARALLILOS_F32_SETVAL(Talue)         _mm256_set1_ps(Talue)
# define PARALLILOS_F32_SETZERO()             _mm256_setzero_ps()
# define PARALLILOS_F32_MUL(a, b)             _mm256_mul_ps(a, b)
# define PARALLILOS_F32_ADD(a, b)             _mm256_add_ps(a, b)
# define PARALLILOS_F32_SUB(a, b)             _mm256_sub_ps(a, b)
# define PARALLILOS_F32_DIV(a, b)             _mm256_div_ps(a, b)
# define PARALLILOS_F32_SQRT(a)               _mm256_sqrt_ps(a)
# define PARALLILOS_F32_ADDMUL(a, b, c)       _mm256_add_ps(a, _mm256_mul_ps(b, c))
# define PARALLILOS_F32_SUBMUL(a, b, c)       _mm256_sub_ps(a, _mm256_mul_ps(b, c))
# define PARALLILOS_F32_EQ(a, b)              _mm256_cmp_ps(a, b, _CMP_EQ_UQ)
# define PARALLILOS_F32_NEQ(a, b)             _mm256_cmp_ps(a, b, _CMP_NEQ_UQ)
# define PARALLILOS_F32_GT(a, b)              _mm256_cmp_ps(a, b, _CMP_GT_OQ)
# define PARALLILOS_F32_GTE(a, b)             _mm256_cmp_ps(a, b, _CMP_GE_OQ)
# define PARALLILOS_F32_LT(a, b)              _mm256_cmp_ps(a, b, _CMP_LT_OQ)
# define PARALLILOS_F32_LTE(a, b)             _mm256_cmp_ps(a, b, _CMP_LE_OQ)
# define PARALLILOS_F32_ABS(a)                _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a)
#elif defined(PARALLILOS_SSE)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define PARALLILOS_F32
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(float, __m128, __m128, 16, "SSE");
# define PARALLILOS_F32_LOADU(data)           _mm_loadu_ps(data)
# define PARALLILOS_F32_LOADA(data)           _mm_load_ps(data)
# define PARALLILOS_F32_STOREU(addr, data)    _mm_storeu_ps(addr, data)
# define PARALLILOS_F32_STOREA(addr, data)    _mm_store_ps(addr, data)
# define PARALLILOS_F32_SETVAL(Talue)         _mm_set1_ps(Talue)
# define PARALLILOS_F32_SETZERO()             _mm_setzero_ps()
# define PARALLILOS_F32_MUL(a, b)             _mm_mul_ps(a, b)
# define PARALLILOS_F32_ADD(a, b)             _mm_add_ps(a, b)
# define PARALLILOS_F32_SUB(a, b)             _mm_sub_ps(a, b)
# define PARALLILOS_F32_DIV(a, b)             _mm_div_ps(a, b)
# define PARALLILOS_F32_SQRT(a)               _mm_sqrt_ps(a)
# define PARALLILOS_F32_ADDMUL(a, b, c)       _mm_add_ps(a, _mm_mul_ps(b, c))
# define PARALLILOS_F32_SUBMUL(a, b, c)       _mm_sub_ps(a, _mm_mul_ps(b, c))
# define PARALLILOS_F32_EQ(a, b)              _mm_cmpeq_ps (a, b)
# define PARALLILOS_F32_NEQ(a, b)             _mm_cmpneq_ps (a, b)
# define PARALLILOS_F32_GT(a, b)              _mm_cmpgt_ps(a, b)
# define PARALLILOS_F32_GTE(a, b)             _mm_cmpge_ps(a, b)
# define PARALLILOS_F32_LT(a, b)              _mm_cmplt_ps(a, b)
# define PARALLILOS_F32_LTE(a, b)             _mm_cmple_ps(a, b)
# define PARALLILOS_F32_ABS(a)                _mm_andnot_ps(_mm_set1_ps(-0.0f), a)
// #elif defined(PARALLILOS_NEON) or defined(PARALLILOS_NEON64)
//   static_assert(sizeof(float) == 4, "float must be 32 bit");
// # define PARALLILOS_F32
//   PARALLILOS_MAKE_SIMD_SPECIALIZATION(float, float32x4_t, uint32x4_t, 0, "NEON");
// # define PARALLILOS_F32_LOADU(data)           vld1q_f32(data)
// # define PARALLILOS_F32_LOADA(data)           vld1q_f32(data)
// # define PARALLILOS_F32_STOREU(addr, data)    vst1q_f32(addr, data)
// # define PARALLILOS_F32_STOREA(addr, data)    vst1q_f32(addr, data)
// # define PARALLILOS_F32_SETVAL(Talue)         vdupq_n_f32(Talue)
// # define PARALLILOS_F32_SETZERO()             vdupq_n_f32(0.0f)
// # define PARALLILOS_F32_MUL(a, b)             vmulq_f32(a, b)
// # define PARALLILOS_F32_ADD(a, b)             vaddq_f32(a, b)
// # define PARALLILOS_F32_SUB(a, b)             vsubq_f32(a, b)
// # define PARALLILOS_F32_DIV(a, b)             vdivq_f32(a, b)
// # define PARALLILOS_F32_SQRT(a)               vsqrtq_f32(a)
// # define PARALLILOS_F32_ADDMUL(a, b, c)       vmlaq_f32(a, b, c)
// # define PARALLILOS_F32_SUBMUL(a, b, c)       vmlsq_f32(a, b, c)
// # define PARALLILOS_F32_EQ(a, b)              vceqq_f32(a, b)
// # define PARALLILOS_F32_NEQ(a, b)             vmvnq_u32(Tceqq_f32(a, b))
// # define PARALLILOS_F32_GT(a, b)              vcgtq_f32(a, b)
// # define PARALLILOS_F32_GTE(a, b)             vcgeq_f32(a, b)
// # define PARALLILOS_F32_LT(a, b)              vcltq_f32(a, b)
// # define PARALLILOS_F32_LTE(a, b)             vcleq_f32(a, b)

// # define PARALLILOS_F32_ABS(a)                vabsq_f32(a)
#endif

#ifdef PARALLILOS_F32
  // load a vector with zeros
  template<> PARALLILOS_INLINE
  SIMD<float>::Type simd_setzero<float>() noexcept
  {
    return PARALLILOS_F32_SETZERO();
#   undef  PARALLILOS_F32_SETZERO
  }

  // load a vector from unaligned data
  PARALLILOS_INLINE
  SIMD<float>::Type simd_loadu(const float data[]) noexcept
  {
    return PARALLILOS_F32_LOADU(data);
#   undef  PARALLILOS_F32_LOADU
  }

  // load a vector from aligned data
  PARALLILOS_INLINE
  SIMD<float>::Type simd_loada(const float data[]) noexcept
  {
    return PARALLILOS_F32_LOADA(data);
#   undef  PARALLILOS_F32_LOADA
  }

  // store a vector into unaligned memory
  PARALLILOS_INLINE
  void simd_storeu(float addr[], const SIMD<float>::Type& data)
  {
    PARALLILOS_F32_STOREU(addr, data);
#   undef PARALLILOS_F32_STOREU
  }

  // store a vector into aligned memory
  PARALLILOS_INLINE
  void simd_storea(float addr[], const SIMD<float>::Type& data)
  {
    PARALLILOS_F32_STOREA(addr, data);
#   undef PARALLILOS_F32_STOREA
  }

  // load a vector with a specific value
  template<> PARALLILOS_INLINE
  SIMD<float>::Type simd_setval(const float value) noexcept
  {
    return PARALLILOS_F32_SETVAL(value);
#   undef  PARALLILOS_F32_SETVAL
  }

  // [a] + [b]
  PARALLILOS_INLINE
  SIMD<float>::Type simd_add(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return PARALLILOS_F32_ADD(a, b);
#   undef  PARALLILOS_F32_ADD
  }

  // [a] * [b]
  PARALLILOS_INLINE
  SIMD<float>::Type simd_mul(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return PARALLILOS_F32_MUL(a, b);
#   undef  PARALLILOS_F32_MUL
  }

  // [a] - [b]
  PARALLILOS_INLINE
  SIMD<float>::Type simd_sub(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return PARALLILOS_F32_SUB(a, b);
#   undef  PARALLILOS_F32_SUB
  }

  // [a] / [b]
  PARALLILOS_INLINE
  SIMD<float>::Type simd_div(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return PARALLILOS_F32_DIV(a, b);
#   undef  PARALLILOS_F32_DIV
  }

  // sqrt([a])
  PARALLILOS_INLINE
  SIMD<float>::Type simd_sqrt(const SIMD<float>::Type& a) noexcept
  {
    return PARALLILOS_F32_SQRT(a);
#   undef  PARALLILOS_F32_SQRT
  }

  // [a] + ([b] * [c])
  PARALLILOS_INLINE
  SIMD<float>::Type simd_addmul(const SIMD<float>::Type& a, const SIMD<float>::Type& b, const SIMD<float>::Type& c) noexcept
  {
    return PARALLILOS_F32_ADDMUL(a, b, c);
#   undef  PARALLILOS_F32_ADDMUL
  }

  // [a] - ([b] * [c])
  PARALLILOS_INLINE
  SIMD<float>::Type simd_submul(const SIMD<float>::Type& a, const SIMD<float>::Type& b, const SIMD<float>::Type& c) noexcept
  {
    return PARALLILOS_F32_SUBMUL(a, b, c);
#   undef  PARALLILOS_F32_SUBMUL
  }

  // [a] == [b]
  PARALLILOS_INLINE
  SIMD<float>::Mask simd_eq(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return PARALLILOS_F32_EQ(a, b);
#   undef  PARALLILOS_F32_EQ
  }

  // [a] != [b]
  PARALLILOS_INLINE
  SIMD<float>::Mask simd_neq(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return PARALLILOS_F32_NEQ(a, b);
#   undef  PARALLILOS_F32_NEQ
  }

  // [a] > [b]
  PARALLILOS_INLINE
  SIMD<float>::Mask simd_gt(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return PARALLILOS_F32_GT(a, b);
#   undef  PARALLILOS_F32_GT
  }

  // [a] >= [b]
  PARALLILOS_INLINE
  SIMD<float>::Mask simd_gte(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return PARALLILOS_F32_GTE(a, b);
#   undef  PARALLILOS_F32_GTE
  }

  // [a] < [b]
  PARALLILOS_INLINE
  SIMD<float>::Mask simd_lt(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return PARALLILOS_F32_LT(a, b);
#   undef  PARALLILOS_F32_LT
  }

  // [a] <= [b]
  PARALLILOS_INLINE
  SIMD<float>::Mask simd_lte(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return PARALLILOS_F32_LTE(a, b);
#   undef  PARALLILOS_F32_LTE
  }

  // abs([a])
  PARALLILOS_INLINE
  SIMD<float>::Type simd_abs(const SIMD<float>::Type& a) noexcept
  {
    return PARALLILOS_F32_ABS(a);
#   undef  PARALLILOS_F32_ABS
  }

  inline
  std::ostream& operator<<(std::ostream& ostream, const SIMD<float>::Type& vector) noexcept
  {
    for (unsigned k = 0; k < SIMD<float>::size; ++k)
    {
      if (k != 0) PARALLILOS_HOT
      {
        ostream << ' ';
      }

      ostream << reinterpret_cast<const float*>(&vector)[k];
    }

    return ostream;
  }
#endif

#if defined(PARALLILOS_AVX512F)
# define PARALLILOS_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(double, __m512d, __mmask8, 64, "AVX512F");
# define PARALLILOS_F64_LOADU(data)           _mm512_loadu_pd(data)
# define PARALLILOS_F64_LOADA(data)           _mm512_load_pd(data)
# define PARALLILOS_F64_STOREU(addr, data)    _mm512_storeu_pd((Toid*)addr, data)
# define PARALLILOS_F64_STOREA(addr, data)    _mm512_store_pd((Toid*)addr, data)
# define PARALLILOS_F64_SETVAL(Talue)         _mm512_set1_pd(Talue)
# define PARALLILOS_F64_SETZERO()             _mm512_setzero_pd()
# define PARALLILOS_F64_MUL(a, b)             _mm512_mul_pd(a, b)
# define PARALLILOS_F64_ADD(a, b)             _mm512_add_pd(a, b)
# define PARALLILOS_F64_SUB(a, b)             _mm512_sub_pd(a, b)
# define PARALLILOS_F64_DIV(a, b)             _mm512_div_pd(a, b)
# define PARALLILOS_F64_SQRT(a)               _mm512_sqrt_pd(a)
# define PARALLILOS_F64_ADDMUL(a, b, c)       _mm512_fmadd_pd(b, c, a)
# define PARALLILOS_F64_SUBMUL(a, b, c)       _mm512_fnmadd_pd(a, b, c)
# define PARALLILOS_F64_EQ(a, b)              _mm512_cmp_pd_mask(a, b, _CMP_EQ_UQ)
# define PARALLILOS_F64_NEQ(a, b)             _mm512_cmp_pd_mask(a, b, _CMP_NEQ_UQ)
# define PARALLILOS_F64_GT(a, b)              _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ)
# define PARALLILOS_F64_GTE(a, b)             _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ)
# define PARALLILOS_F64_LT(a, b)              _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ)
# define PARALLILOS_F64_LTE(a, b)             _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ)

# define PARALLILOS_F64_ABS(a)                _mm512_abs_pd(a)
#elif defined(PARALLILOS_FMA)
# define PARALLILOS_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(double, __m256d, __m256d, 32, "AVX, FMA");
# define PARALLILOS_F64_LOADU(data)           _mm256_loadu_pd(data)
# define PARALLILOS_F64_LOADA(data)           _mm256_load_pd(data)
# define PARALLILOS_F64_STOREU(addr, data)    _mm256_storeu_pd(addr, data)
# define PARALLILOS_F64_STOREA(addr, data)    _mm256_store_pd(addr, data)
# define PARALLILOS_F64_SETVAL(Talue)         _mm256_set1_pd(Talue)
# define PARALLILOS_F64_SETZERO()             _mm256_setzero_pd()
# define PARALLILOS_F64_MUL(a, b)             _mm256_mul_pd(a, b)
# define PARALLILOS_F64_ADD(a, b)             _mm256_add_pd(a, b)
# define PARALLILOS_F64_SUB(a, b)             _mm256_sub_pd(a, b)
# define PARALLILOS_F64_DIV(a, b)             _mm256_div_pd(a, b)
# define PARALLILOS_F64_SQRT(a)               _mm256_sqrt_pd(a)
# define PARALLILOS_F64_ADDMUL(a, b, c)       _mm256_fmadd_pd(b, c, a)
# define PARALLILOS_F64_SUBMUL(a, b, c)       _mm256_fnmadd_pd(a, b, c)
# define PARALLILOS_F64_EQ(a, b)              _mm256_cmp_pd(a, b, _CMP_EQ_UQ)
# define PARALLILOS_F64_NEQ(a, b)             _mm256_cmp_pd(a, b, _CMP_NEQ_UQ)
# define PARALLILOS_F64_GT(a, b)              _mm256_cmp_pd(a, b, _CMP_GT_OQ)
# define PARALLILOS_F64_GTE(a, b)             _mm256_cmp_pd(a, b, _CMP_GE_OQ)
# define PARALLILOS_F64_LT(a, b)              _mm256_cmp_pd(a, b, _CMP_LT_OQ)
# define PARALLILOS_F64_LTE(a, b)             _mm256_cmp_pd(a, b, _CMP_LE_OQ)

# define PARALLILOS_F64_ABS(a)                _mm256_andnot_pd(_mm256_set1_pd(-0.0), a)
#elif defined(PARALLILOS_AVX)
# define PARALLILOS_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(double, __m256d, __m256d, 32, "AVX");
# define PARALLILOS_F64_LOADU(data)           _mm256_loadu_pd(data)
# define PARALLILOS_F64_LOADA(data)           _mm256_load_pd(data)
# define PARALLILOS_F64_STOREU(addr, data)    _mm256_storeu_pd(addr, data)
# define PARALLILOS_F64_STOREA(addr, data)    _mm256_store_pd(addr, data)
# define PARALLILOS_F64_SETVAL(Talue)         _mm256_set1_pd(Talue)
# define PARALLILOS_F64_SETZERO()             _mm256_setzero_pd()
# define PARALLILOS_F64_MUL(a, b)             _mm256_mul_pd(a, b)
# define PARALLILOS_F64_ADD(a, b)             _mm256_add_pd(a, b)
# define PARALLILOS_F64_SUB(a, b)             _mm256_sub_pd(a, b)
# define PARALLILOS_F64_DIV(a, b)             _mm256_div_pd(a, b)
# define PARALLILOS_F64_SQRT(a)               _mm256_sqrt_pd(a)
# define PARALLILOS_F64_ADDMUL(a, b, c)       _mm256_add_pd(a, _mm256_mul_pd(b, c))
# define PARALLILOS_F64_SUBMUL(a, b, c)       _mm256_sub_pd(a, _mm256_mul_pd(b, c))
# define PARALLILOS_F64_EQ(a, b)              _mm256_cmp_pd(a, b, _CMP_EQ_UQ)
# define PARALLILOS_F64_NEQ(a, b)             _mm256_cmp_pd(a, b, _CMP_NEQ_UQ)
# define PARALLILOS_F64_GT(a, b)              _mm256_cmp_pd(a, b, _CMP_GT_OQ)
# define PARALLILOS_F64_GTE(a, b)             _mm256_cmp_pd(a, b, _CMP_GE_OQ)
# define PARALLILOS_F64_LT(a, b)              _mm256_cmp_pd(a, b, _CMP_LT_OQ)
# define PARALLILOS_F64_LTE(a, b)             _mm256_cmp_pd(a, b, _CMP_LE_OQ)

# define PARALLILOS_F64_ABS(a)                _mm256_andnot_pd(_mm256_set1_pd(-0.0), a)
#elif defined(PARALLILOS_SSE2)
# define PARALLILOS_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(double, __m128d, __m128d, 16, "SSE2");
# define PARALLILOS_F64_LOADU(data)           _mm_loadu_pd(data)
# define PARALLILOS_F64_LOADA(data)           _mm_load_pd(data)
# define PARALLILOS_F64_STOREU(addr, data)    _mm_storeu_pd(addr, data)
# define PARALLILOS_F64_STOREA(addr, data)    _mm_store_pd(addr, data)
# define PARALLILOS_F64_SETVAL(Talue)         _mm_set1_pd(Talue)
# define PARALLILOS_F64_SETZERO()             _mm_setzero_pd()
# define PARALLILOS_F64_MUL(a, b)             _mm_mul_pd(a, b)
# define PARALLILOS_F64_ADD(a, b)             _mm_add_pd(a, b)
# define PARALLILOS_F64_SUB(a, b)             _mm_sub_pd(a, b)
# define PARALLILOS_F64_DIV(a, b)             _mm_div_pd(a, b)
# define PARALLILOS_F64_SQRT(a)               _mm_sqrt_pd(a)
# define PARALLILOS_F64_ADDMUL(a, b, c)       _mm_add_pd(a, _mm_mul_pd(b, c))
# define PARALLILOS_F64_SUBMUL(a, b, c)       _mm_sub_pd(a, _mm_mul_pd(b, c))
# define PARALLILOS_F64_EQ(a, b)              _mm_cmpeq_pd(a, b)
# define PARALLILOS_F64_NEQ(a, b)             _mm_cmpneq_pd(a, b)
# define PARALLILOS_F64_GT(a, b)              _mm_cmpgt_pd(a, b)
# define PARALLILOS_F64_GTE(a, b)             _mm_cmpge_pd(a, b)
# define PARALLILOS_F64_LT(a, b)              _mm_cmplt_pd(a, b)
# define PARALLILOS_F64_LTE(a, b)             _mm_cmple_pd(a, b)

# define PARALLILOS_F64_ABS(a)                _mm_andnot_pd(_mm_set1_pd(-0.0), a)
// #elif defined(PARALLILOS_NEON) or defined(PARALLILOS_NEON64)
// # define PARALLILOS_F64
//   static_assert(sizeof(double) == 8, "float must be 64 bit");
//   PARALLILOS_MAKE_SIMD_SPECIALIZATION(double, float64x4_t, float64x4_t, 0, "NEON");
// # define PARALLILOS_F64_LOADU(data)           vld1q_f64(data)
// # define PARALLILOS_F64_LOADA(data)           vld1q_f64(data)
// # define PARALLILOS_F64_STOREU(addr, data)    vst1q_f64(addr, data)
// # define PARALLILOS_F64_STOREA(addr, data)    vst1q_f64(addr, data)
// # define PARALLILOS_F64_SETVAL(Talue)         vdupq_n_f64(Talue)
// # define PARALLILOS_F64_SETZERO()             vdupq_n_f64(0.0)
// # define PARALLILOS_F64_MUL(a, b)             vmulq_f64(a, b)
// # define PARALLILOS_F64_ADD(a, b)             vaddq_f64(a, b)
// # define PARALLILOS_F64_SUB(a, b)             vsubq_f64(a, b)
// # define PARALLILOS_F64_DIV(a, b)             vdivq_f64(a, b)
// # define PARALLILOS_F64_SQRT(a)               vsqrtq_f64(a)
// # define PARALLILOS_F64_ADDMUL(a, b, c)       vmlaq_f64(a, b, c)
// # define PARALLILOS_F64_SUBMUL(a, b, c)       vmlsq_f64(a, b, c)
// # define PARALLILOS_F32_EQ(a, b)              vceqq_f64(a, b)
// # define PARALLILOS_F32_NEQ(a, b)             vmvnq_u64(Tceqq_f64(a, b))
// # define PARALLILOS_F32_GT(a, b)              vcgtq_f64(a, b)
// # define PARALLILOS_F32_GTE(a, b)             vcgeq_f64(a, b)
// # define PARALLILOS_F32_LT(a, b)              vcltq_f64(a, b)
// # define PARALLILOS_F32_LTE(a, b)             vcleq_f64(a, b)

// # define PARALLILOS_F32_ABS(a)                vabsq_f64(a)
#endif

#ifdef PARALLILOS_F64
  // load a vector with zeros
  template<> PARALLILOS_INLINE
  SIMD<double>::Type simd_setzero<double>() noexcept
  {
    return PARALLILOS_F64_SETZERO();
#   undef  PARALLILOS_F64_SETZERO
  }

  // load a vector from unaligned data
  PARALLILOS_INLINE
  SIMD<double>::Type simd_loadu(const double data[]) noexcept
  {
    return PARALLILOS_F64_LOADU(data);
#   undef  PARALLILOS_F64_LOADU
  }

  // load a vector from aligned data
  PARALLILOS_INLINE
  SIMD<double>::Type simd_loada(const double data[]) noexcept
  {
    return PARALLILOS_F64_LOADA(data);
#   undef  PARALLILOS_F64_LOADA
  }

  // store a vector into unaligned memory
  PARALLILOS_INLINE
  void simd_storeu(double addr[], const SIMD<double>::Type& data)
  {
    PARALLILOS_F64_STOREU(addr, data);
#   undef PARALLILOS_F64_STOREU
  }

  // store a vector into aligned memory
  PARALLILOS_INLINE
  void simd_storea(double addr[], const SIMD<double>::Type& data)
  {
    PARALLILOS_F64_STOREA(addr, data);
#   undef PARALLILOS_F64_STOREA
  }

  // load a vector with a specific value
  template<> PARALLILOS_INLINE
  SIMD<double>::Type simd_setval(const double value) noexcept
  {
    return PARALLILOS_F64_SETVAL(value);
#   undef  PARALLILOS_F64_SETVAL
  }

  // [a] + [b]
  PARALLILOS_INLINE
  SIMD<double>::Type simd_add(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return PARALLILOS_F64_ADD(a, b);
#   undef  PARALLILOS_F64_ADD
  }

  // [a] * [b]
  PARALLILOS_INLINE
  SIMD<double>::Type simd_mul(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return PARALLILOS_F64_MUL(a, b);
#   undef  PARALLILOS_F64_MUL
  }

  // [a] - [b]
  PARALLILOS_INLINE
  SIMD<double>::Type simd_sub(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return PARALLILOS_F64_SUB(a, b);
#   undef  PARALLILOS_F64_SUB
  }

  // [a] / [b]
  PARALLILOS_INLINE
  SIMD<double>::Type simd_div(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return PARALLILOS_F64_DIV(a, b);
#   undef  PARALLILOS_F64_DIV
  }

  // sqrt([a])
  PARALLILOS_INLINE
  SIMD<double>::Type simd_sqrt(const SIMD<double>::Type& a) noexcept
  {
    return PARALLILOS_F64_SQRT(a);
#   undef  PARALLILOS_F64_SQRT
  }

  // [a] + ([b] * [c])
  PARALLILOS_INLINE
  SIMD<double>::Type simd_addmul(const SIMD<double>::Type& a, const SIMD<double>::Type& b, const SIMD<double>::Type& c) noexcept
  {
    return PARALLILOS_F64_ADDMUL(a, b, c);
#   undef  PARALLILOS_F64_ADDMUL
  }

  // [a] - ([b] * [c])
  PARALLILOS_INLINE
  SIMD<double>::Type simd_submul(const SIMD<double>::Type& a, const SIMD<double>::Type& b, const SIMD<double>::Type& c) noexcept
  {
    return PARALLILOS_F64_SUBMUL(a, b, c);
#   undef  PARALLILOS_F64_SUBMUL
  }

  // [a] == [b]
  PARALLILOS_INLINE
  SIMD<double>::Mask simd_eq(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return PARALLILOS_F64_EQ(a, b);
#   undef  PARALLILOS_F64_EQ
  }

  // [a] != [b]
  PARALLILOS_INLINE
  SIMD<double>::Mask simd_neq(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return PARALLILOS_F64_NEQ(a, b);
#   undef  PARALLILOS_F64_NEQ
  }

  // [a] > [b]
  PARALLILOS_INLINE
  SIMD<double>::Mask simd_gt(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return PARALLILOS_F64_GT(a, b);
#   undef  PARALLILOS_F64_GT
  }

  // [a] >= [b]
  PARALLILOS_INLINE
  SIMD<double>::Mask simd_gte(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return PARALLILOS_F64_GTE(a, b);
#   undef  PARALLILOS_F64_GTE
  }

  // [a] < [b]
  PARALLILOS_INLINE
  SIMD<double>::Mask simd_lt(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return PARALLILOS_F64_LT(a, b);
#   undef  PARALLILOS_F64_LT
  }

  // [a] <= [b]
  PARALLILOS_INLINE
  SIMD<double>::Mask simd_lte(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return PARALLILOS_F64_LTE(a, b);
#   undef  PARALLILOS_F64_LTE
  }

  // abs([a])
  PARALLILOS_INLINE
  SIMD<double>::Type simd_abs(const SIMD<double>::Type& a) noexcept
  {
    return PARALLILOS_F64_ABS(a);
#   undef  PARALLILOS_F64_ABS
  }

  inline
  std::ostream& operator<<(std::ostream& ostream, const SIMD<double>::Type& vector) noexcept
  {
    for (unsigned k = 0; k < SIMD<double>::size; ++k)
    {
      if (k != 0) PARALLILOS_HOT
      {
        ostream << ' ';
      }

      ostream << reinterpret_cast<const double*>(&vector)[k];
    }

    return ostream;
  }
#endif

#if defined(PARALLILOS_AVX512F)
# define PARALLILOS_I32
  static_assert(sizeof(int32_t) == 4, "int32_t must be 32 bit");
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(int32_t, __m512i, __mmask16, 64, "AVX512F");
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(uint32_t, __m512i, __mmask16, 64, "AVX512F");
# define PARALLILOS_I32_LOADU(data)           _mm512_loadu_si512(data)
# define PARALLILOS_I32_LOADA(data)           _mm512_load_si512(data)
# define PARALLILOS_I32_STOREU(addr, data)    _mm512_storeu_si512((Toid*)addr, data)
# define PARALLILOS_I32_STOREA(addr, data)    _mm512_store_si512((Toid*)addr, data)
# define PARALLILOS_I32_SETVAL(Talue)         _mm512_set1_epi32(Talue)
# define PARALLILOS_I32_SETZERO()             _mm512_setzero_si512()
# define PARALLILOS_I32_MUL(a, b)             _mm512_mullo_epi32 (a, b)
# define PARALLILOS_I32_ADD(a, b)             _mm512_add_epi32(a, b)
# define PARALLILOS_I32_SUB(a, b)             _mm512_sub_epi32(a, b)
# define PARALLILOS_I32_DIV(a, b)             _mm512_cvtps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(a), _mm512_cvtepi32_ps(b)))
# define PARALLILOS_I32_SQRT(a)               _mm512_cvtps_epi32(_mm512_sqrt_ps(_mm512_cvtepi32_ps(a)))
# define PARALLILOS_I32_ADDMUL(a, b, c)       _mm512_add_epi32(a, _mm512_mul_epi32(b, c))
# define PARALLILOS_I32_SUBMUL(a, b, c)       _mm512_sub_epi32(a, _mm512_mul_epi32(b, c))
# if defined(PARALLILOS_SVML)
#   undef  PARALLILOS_I32_DIV
#   define PARALLILOS_I32_DIV(a, b)           _mm512_div_epi32(a, b)
# endif

# define PARALLILOS_I32_EQ(a, b)              _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_EQ)
# define PARALLILOS_I32_NEQ(a, b)             _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NE)
# define PARALLILOS_I32_GT(a, b)              _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLE)
# define PARALLILOS_I32_GTE(a, b)             _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLT)
# define PARALLILOS_I32_LT(a, b)              _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LT)
# define PARALLILOS_I32_LTE(a, b)             _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LE)

# define PARALLILOS_I32_BW_NOT(a)             _mm512_xor_si512(a, _mm512_set1_epi32(-1))
# define PARALLILOS_I32_BW_AND(a, b)          _mm512_and_si512(a, b)
# define PARALLILOS_I32_BW_NAND(a, b)         _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_set1_epi32(-1))
# define PARALLILOS_I32_BW_OR(a, b)           _mm512_or_si512(a, b)
# define PARALLILOS_I32_BW_NOR(a, b)          _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_set1_epi32(-1))
# define PARALLILOS_I32_BW_XOR(a, b)          _mm512_xor_si512(a, b)
# define PARALLILOS_I32_BW_XNOR(a, b)         _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_set1_epi32(-1))

# define PARALLILOS_I32_ABS(a)                _mm512_abs_epi32(a)
#elif defined(PARALLILOS_AVX2)
# define PARALLILOS_I32
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(int32_t, __m256i, __m256i, 32, "AVX2, AVX");
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(uint32_t, __m256i, __m256i, 32, "AVX2, AVX");
# define PARALLILOS_I32_LOADU(data)           _mm256_loadu_si256((const __m256i*)data)
# define PARALLILOS_I32_LOADA(data)           _mm256_load_si256((const __m256i*)data)
# define PARALLILOS_I32_STOREU(addr, data)    _mm256_storeu_si256 ((__m256i*)addr, data)
# define PARALLILOS_I32_STOREA(addr, data)    _mm256_store_si256((__m256i*)addr, data)
# define PARALLILOS_I32_SETVAL(Talue)         _mm256_set1_epi32(Talue)
# define PARALLILOS_I32_SETZERO()             _mm256_setzero_si256()
# define PARALLILOS_I32_MUL(a, b)             _mm256_mullo_epi32(a, b)
# define PARALLILOS_I32_ADD(a, b)             _mm256_add_epi32(a, b)
# define PARALLILOS_I32_SUB(a, b)             _mm256_sub_epi32(a, b)
# define PARALLILOS_I32_DIV(a, b)             _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(a), _mm256_cvtepi32_ps(b)))
# define PARALLILOS_I32_SQRT(a)               _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(a)))
# define PARALLILOS_I32_ADDMUL(a, b, c)       _mm256_add_epi32(a, _mm256_mul_epi32(b, c))
# define PARALLILOS_I32_SUBMUL(a, b, c)       _mm256_sub_epi32(a, _mm256_mul_epi32(b, c))
# if defined(PARALLILOS_SVML)
#   undef  PARALLILOS_I32_DIV
#   define PARALLILOS_I32_DIV(a, b)           _mm256_div_epi32(a, b)
# endif

# define PARALLILOS_I32_EQ(a, b)              _mm256_cmpeq_epi32(a, b)
# define PARALLILOS_I32_NEQ(a, b)             _mm256_xor_si256(_mm256_cmpeq_epi32(a, b), _mm256_cmpeq_epi32(a, a))
# define PARALLILOS_I32_GT(a, b)              _mm256_cmpgt_epi32(a, b)
# define PARALLILOS_I32_GTE(a, b)             _mm256_xor_si256(_mm256_cmpgt_epi32(b, a), _mm256_cmpeq_epi32(a, a))
# define PARALLILOS_I32_LT(a, b)              _mm256_cmpgt_epi32(b, a)
# define PARALLILOS_I32_LTE(a, b)             _mm256_xor_si256(_mm256_cmpgt_epi32(a, b), _mm256_cmpeq_epi32(a, a))

# define PARALLILOS_I32_BW_NOT(a)             _mm256_xor_si256(a, _mm256_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_AND(a, b)          _mm256_and_si256(a, b)
# define PARALLILOS_I32_BW_NAND(a, b)         _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_OR(a, b)           _mm256_or_si256(a, b)
# define PARALLILOS_I32_BW_NOR(a, b)          _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_XOR(a, b)          _mm256_xor_si256(a, b)
# define PARALLILOS_I32_BW_XNOR(a, b)         _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_cmpeq_epi32(a, a))

# define PARALLILOS_I32_ABS(a)                _mm256_abs_epi32(a)
#elif defined(PARALLILOS_SSE4_1)
# define PARALLILOS_I32
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(int32_t, __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(uint32_t, __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
# define PARALLILOS_I32_LOADU(data)           _mm_loadu_si128((const __m128i*)data)
# define PARALLILOS_I32_LOADA(data)           _mm_load_si128((const __m128i*)data)
# define PARALLILOS_I32_STOREU(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
# define PARALLILOS_I32_STOREA(addr, data)    _mm_store_si128((__m128i*)addr, data)
# define PARALLILOS_I32_SETVAL(Talue)         _mm_set1_epi32(Talue)
# define PARALLILOS_I32_SETZERO()             _mm_setzero_si128()
# define PARALLILOS_I32_MUL(a, b)             _mm_mullo_epi32(a, b)
# define PARALLILOS_I32_ADD(a, b)             _mm_add_epi32(a, b)
# define PARALLILOS_I32_SUB(a, b)             _mm_sub_epi32(a, b)
# define PARALLILOS_I32_DIV(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define PARALLILOS_I32_SQRT(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
# define PARALLILOS_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_mul_epi32(b, c))
# define PARALLILOS_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_mul_epi32(b, c))
# if defined(PARALLILOS_SVML)
#   undef  PARALLILOS_I32_DIV
#   define PARALLILOS_I32_DIV(a, b)           _mm_div_epi32(a, b)
# endif

# define PARALLILOS_I32_EQ(a, b)              _mm_cmpeq_epi32(a, b)
# define PARALLILOS_I32_NEQ(a, b)             _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_GT(a, b)              _mm_cmpgt_epi32(a, b)
# define PARALLILOS_I32_GTE(a, b)             _mm_xor_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_LT(a, b)              _mm_cmplt_epi32(a, b)
# define PARALLILOS_I32_LTE(a, b)             _mm_xor_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, a))

# define PARALLILOS_I32_BW_NOT(a)             _mm_xor_si128(a, _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_AND(a, b)          _mm_and_si128(a, b)
# define PARALLILOS_I32_BW_NAND(a, b)         _mm_xor_si128(_mm_and_si128(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_OR(a, b)           _mm_or_si128(a, b)
# define PARALLILOS_I32_BW_NOR(a, b)          _mm_xor_si128(_mm_or_si128(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_XOR(a, b)          _mm_xor_si128(a, b)
# define PARALLILOS_I32_BW_XNOR(a, b)         _mm_xor_si128(_mm_xor_si128(a, b), _mm_cmpeq_epi32(a, a))

# define PARALLILOS_I32_ABS(a)                _mm_abs_epi32(a)
#elif defined(PARALLILOS_SSSE3)
# define PARALLILOS_I32
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(int32_t, __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(uint32_t, __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
# define PARALLILOS_I32_LOADU(data)           _mm_loadu_si128((const __m128i*)data)
# define PARALLILOS_I32_LOADA(data)           _mm_load_si128((const __m128i*)data)
# define PARALLILOS_I32_STOREU(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
# define PARALLILOS_I32_STOREA(addr, data)    _mm_store_si128((__m128i*)addr, data)
# define PARALLILOS_I32_SETVAL(Talue)         _mm_set1_epi32(Talue)
# define PARALLILOS_I32_SETZERO()             _mm_setzero_si128()
# define PARALLILOS_I32_MUL(a, b)             _mm_mullo_epi32(a, b)
# define PARALLILOS_I32_ADD(a, b)             _mm_add_epi32(a, b)
# define PARALLILOS_I32_SUB(a, b)             _mm_sub_epi32(a, b)
# define PARALLILOS_I32_DIV(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define PARALLILOS_I32_SQRT(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
# define PARALLILOS_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_mul_epi32(b, c))
# define PARALLILOS_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_mul_epi32(b, c))
# if defined(PARALLILOS_SVML)
#   undef  PARALLILOS_I32_DIV
#   define PARALLILOS_I32_DIV(a, b)           _mm_div_epi32(a, b)
# endif

# define PARALLILOS_I32_EQ(a, b)              _mm_cmpeq_epi32(a, b)
# define PARALLILOS_I32_NEQ(a, b)             _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_GT(a, b)              _mm_cmpgt_epi32(a, b)
# define PARALLILOS_I32_GTE(a, b)             _mm_xor_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_LT(a, b)              _mm_cmplt_epi32(a, b)
# define PARALLILOS_I32_LTE(a, b)             _mm_xor_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, a))

# define PARALLILOS_I32_BW_NOT(a)             _mm_xor_si128(a, _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_AND(a, b)          _mm_and_si128(a, b)
# define PARALLILOS_I32_BW_NAND(a, b)         _mm_xor_si128(_mm_and_si128(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_OR(a, b)           _mm_or_si128(a, b)
# define PARALLILOS_I32_BW_NOR(a, b)          _mm_xor_si128(_mm_or_si128(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_XOR(a, b)          _mm_xor_si128(a, b)
# define PARALLILOS_I32_BW_XNOR(a, b)         _mm_xor_si128(_mm_xor_si128(a, b), _mm_cmpeq_epi32(a, a))

# define PARALLILOS_I32_ABS(a)                _mm_abs_epi32(a)
#elif defined(PARALLILOS_SSE2)
# define PARALLILOS_I32
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(int32_t, __m128i, __m128i, 16, "SSE2, SSE");
  PARALLILOS_MAKE_SIMD_SPECIALIZATION(uint32_t, __m128i, __m128i, 16, "SSE2, SSE");
# define PARALLILOS_I32_LOADU(data)           _mm_loadu_si128((const __m128i*)data)
# define PARALLILOS_I32_LOADA(data)           _mm_load_si128((const __m128i*)data)
# define PARALLILOS_I32_STOREU(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
# define PARALLILOS_I32_STOREA(addr, data)    _mm_store_si128((__m128i*)addr, data)
# define PARALLILOS_I32_SETVAL(Talue)         _mm_set1_epi32(Talue)
# define PARALLILOS_I32_SETZERO()             _mm_setzero_si128()
# define PARALLILOS_I32_MUL(a, b)             _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define PARALLILOS_I32_ADD(a, b)             _mm_add_epi32(a, b)
# define PARALLILOS_I32_SUB(a, b)             _mm_sub_epi32(a, b)
# define PARALLILOS_I32_DIV(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define PARALLILOS_I32_SQRT(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
# define PARALLILOS_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
# define PARALLILOS_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))

# define PARALLILOS_I32_EQ(a, b)              _mm_cmpeq_epi32(a, b)
# define PARALLILOS_I32_NEQ(a, b)             _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_GT(a, b)              _mm_cmpgt_epi32(a, b)
# define PARALLILOS_I32_GTE(a, b)             _mm_xor_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_LT(a, b)              _mm_cmplt_epi32(a, b)
# define PARALLILOS_I32_LTE(a, b)             _mm_xor_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, a))

# define PARALLILOS_I32_BW_NOT(a)             _mm_xor_si128(a, _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_AND(a, b)          _mm_and_si128(a, b)
# define PARALLILOS_I32_BW_NAND(a, b)         _mm_xor_si128(_mm_and_si128(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_OR(a, b)           _mm_or_si128(a, b)
# define PARALLILOS_I32_BW_NOR(a, b)          _mm_xor_si128(_mm_or_si128(a, b), _mm_cmpeq_epi32(a, a))
# define PARALLILOS_I32_BW_XOR(a, b)          _mm_xor_si128(a, b)
# define PARALLILOS_I32_BW_XNOR(a, b)         _mm_xor_si128(_mm_xor_si128(a, b), _mm_cmpeq_epi32(a, a))

# define PARALLILOS_I32_ABS(a)                                  \
  [&]() -> __m128i {                                            \
    const __m128i signmask = _mm_srai_epi32(a, 31);             \
    return _mm_sub_epi32(_mm_xor_si128(a, signmask), signmask); \
  }()
#endif

#ifdef PARALLILOS_I32
  // load a vector with zeros
  template<> PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_setzero<int32_t>() noexcept
  {
    return PARALLILOS_I32_SETZERO();
  }
  template<> PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_setzero<uint32_t>() noexcept
  {
    return PARALLILOS_I32_SETZERO();
#   undef  PARALLILOS_I32_SETZERO
  }

  // load a vector from unaligned data
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_loadu(const int32_t data[]) noexcept
  {
    return PARALLILOS_I32_LOADU(data);
  }
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_loadu(const uint32_t data[]) noexcept
  {
    return PARALLILOS_I32_LOADU(data);
#   undef  PARALLILOS_I32_LOADU
  }

  // load a vector from aligned data
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_loada(const int32_t data[]) noexcept
  {
    return PARALLILOS_I32_LOADA(data);
  }
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_loada(const uint32_t data[]) noexcept
  {
    return PARALLILOS_I32_LOADA(data);
#   undef  PARALLILOS_I32_LOADA
  }

  // store a vector into unaligned memory
  PARALLILOS_INLINE
  void simd_storeu(int32_t addr[], const SIMD<int32_t>::Type& data)
  {
    PARALLILOS_I32_STOREU(addr, data);
#   undef PARALLILOS_I32_STOREU
  }

  // store a vector into aligned memory
  PARALLILOS_INLINE
  void simd_storea(int32_t addr[], const SIMD<int32_t>::Type& data)
  {
    PARALLILOS_I32_STOREA(addr, data);
  }
  PARALLILOS_INLINE
  void simd_storea(uint32_t addr[], const SIMD<int32_t>::Type& data)
  {
    PARALLILOS_I32_STOREA(addr, data);
#   undef PARALLILOS_I32_STOREA
  }

  // load a vector with a specific value
  template<> PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_setval(const int32_t value) noexcept
  {
    return PARALLILOS_I32_SETVAL(value);
  }
  template<> PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_setval(const uint32_t value) noexcept
  {
    return PARALLILOS_I32_SETVAL(value);
#   undef  PARALLILOS_I32_SETVAL
  }

  // [a] + [b]
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_add(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return PARALLILOS_I32_ADD(a, b);
#   undef  PARALLILOS_I32_ADD
  }

  // [a] * [b]
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_mul(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return PARALLILOS_I32_MUL(a, b);
#   undef  PARALLILOS_I32_MUL
  }

  // [a] - [b]
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_sub(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return PARALLILOS_I32_SUB(a, b);
#   undef  PARALLILOS_I32_SUB
  }

  // [a] / [b]
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_div(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return PARALLILOS_I32_DIV(a, b);
#   undef  PARALLILOS_I32_DIV
  }

  // sqrt([a])
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_sqrt(const SIMD<int32_t>::Type& a) noexcept
  {
    return PARALLILOS_I32_SQRT(a);
#   undef  PARALLILOS_I32_SQRT
  }

  // [a] + ([b] * [c])
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_addmul(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b, const SIMD<int32_t>::Type& c) noexcept
  {
    return PARALLILOS_I32_ADDMUL(a, b, c);
#   undef  PARALLILOS_I32_ADDMUL
  }

  // [a] - ([b] * [c])
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_submul(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b, const SIMD<int32_t>::Type& c) noexcept
  {
    return PARALLILOS_I32_SUBMUL(a, b, c);
#   undef  PARALLILOS_I32_SUBMUL
  }

  // [a] == [b]
  PARALLILOS_INLINE
  auto simd_eq(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return PARALLILOS_I32_EQ(a, b);
#   undef  PARALLILOS_I32_EQ
  }

  // [a] != [b]
  PARALLILOS_INLINE
  auto simd_neq(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return PARALLILOS_I32_NEQ(a, b);
#   undef  PARALLILOS_I32_NEQ
  }

  // [a] > [b]
  PARALLILOS_INLINE
  auto simd_gt(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return PARALLILOS_I32_GT(a, b);
#   undef  PARALLILOS_I32_GT
  }

  // [a] >= [b]
  PARALLILOS_INLINE
  auto simd_gte(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return PARALLILOS_I32_GTE(a, b);
#   undef  PARALLILOS_I32_GTE
  }

  // [a] < [b]
  PARALLILOS_INLINE
  auto simd_lt(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return PARALLILOS_I32_LT(a, b);
#   undef  PARALLILOS_I32_LT
  }

  // [a] <= [b]
  PARALLILOS_INLINE
  auto simd_lte(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return PARALLILOS_I32_LTE(a, b);
#   undef  PARALLILOS_I32_LTE
  }

  inline namespace Bitwise
  {
    // ![a]
    PARALLILOS_INLINE
    SIMD<int32_t>::Type simd_not(const SIMD<int32_t>::Type& a) noexcept
    {
      return PARALLILOS_I32_BW_NOT(a);
#     undef  PARALLILOS_I32_BW_NOT
    }

    // [a] & [b]
    PARALLILOS_INLINE
    SIMD<int32_t>::Type simd_and(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return PARALLILOS_I32_BW_AND(a, b);
#     undef  PARALLILOS_I32_BW_AND
    }

    // !([a] & [b])
    PARALLILOS_INLINE
    SIMD<int32_t>::Type simd_nand(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return PARALLILOS_I32_BW_NAND(a, b);
#     undef  PARALLILOS_I32_BW_NAND
    }

    // [a] | [b]
    PARALLILOS_INLINE
    SIMD<int32_t>::Type simd_or(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return PARALLILOS_I32_BW_OR(a, b);
#     undef  PARALLILOS_I32_BW_OR
    }

    // !([a] | [b])
    PARALLILOS_INLINE
    SIMD<int32_t>::Type simd_nor(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return PARALLILOS_I32_BW_NOR(a, b);
#     undef  PARALLILOS_I32_BW_NOR
    }

    // [a] ^ [b]
    PARALLILOS_INLINE
    SIMD<int32_t>::Type simd_xor(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return PARALLILOS_I32_BW_XOR(a, b);
#     undef  PARALLILOS_I32_BW_XOR
    }

    // !([a] ^ [b])
    PARALLILOS_INLINE
    SIMD<int32_t>::Type simd_xnor(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return PARALLILOS_I32_BW_XNOR(a, b);
#     undef  PARALLILOS_I32_BW_XNOR
    }
  }

  // abs([a])
  PARALLILOS_INLINE
  SIMD<int32_t>::Type simd_abs(const SIMD<int32_t>::Type& a) noexcept
  {
    return PARALLILOS_I32_ABS(a);
#   undef  PARALLILOS_I32_ABS
  }

  inline
  std::ostream& operator<<(std::ostream& ostream, const SIMD<int32_t>::Type& vector) noexcept
  {
    for (unsigned k = 0; k < SIMD<int32_t>::size; ++k)
    {
      if (k != 0) PARALLILOS_HOT
      {
        ostream << ' ';
      }

      ostream << reinterpret_cast<const int32_t*>(&vector)[k];
    }

    return ostream;
  }
#endif
}
#undef PARALLILOS_INLINE
#undef PARALLILOS_COLD
#undef PARALLILOS_HOT
#undef PARALLILOS_THREADSAFE
#undef PARALLILOS_THREADLOCAL
#undef PARALLILOS_MAKE_MUTEX
#undef PARALLILOS_LOCK
#undef PARALLILOS_MAKE_SIMD_SPECIALIZATION
#else
#error "Parallilos: Support for ISO C++11 is required"
#endif
#endif