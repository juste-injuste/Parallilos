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

ppz::SIMD;

-----inclusion guard--------------------------------------------------------------------------------------------------*/
#if not defined(_ppz_HPP)
#if defined(__cplusplus) and (__cplusplus >= 201103L)
#define _ppz_HPP
//---necessary standard libraries---------------------------------------------------------------------------------------
#include <cstddef>      // for size_t
#include <cstdint>      // for fixed-sized integers
#include <cmath>        // for std::sqrt
#include <cstdlib>      // for std::malloc, std::free, std::aligned_alloc
#include <ostream>      // for std::ostream
#include <iostream>     // for std::cerr
#include <type_traits>  // for std::is_arithmetic, std::is_integral, std::enable_if
//---conditionally necessary standard libraries-------------------------------------------------------------------------
#if defined(__STDCPP_THREADS__) and not defined(PARALLILOS_NOT_THREADSAFE)
# define _impl_ppz_THREADSAFE
# include <mutex>       // for std::mutex, std::lock_guard
#endif
#if defined(PARALLILOS_LOGGING)
# include <type_traits> // for std::is_floating_point, std::is_unsigned, std::is_pointer, std::remove_pointer
# include <typeinfo>    // for typeid
# include <cstdio>      // for std::sprintf
#endif
//---Parallilos library-------------------------------------------------------------------------------------------------
#if defined(__AVX512F__)
# define _impl_ppz_AVX512F
# define _impl_ppz_SIMD_HEADER <immintrin.h>
#endif
#if defined(__AVX2__)
# define _impl_ppz_AVX2
# define _impl_ppz_SIMD_HEADER <immintrin.h>
#endif
#if defined(__FMA__)
# define _impl_ppz_FMA
# define _impl_ppz_SIMD_HEADER <immintrin.h>
#endif
#if defined(__AVX__)
# define _impl_ppz_AVX
# define _impl_ppz_SIMD_HEADER <immintrin.h>
#endif

#if defined(__SSE4_2__)
# define _impl_ppz_SSE4_2
# define _impl_ppz_SIMD_HEADER <immintrin.h>
#endif
#if defined(__SSE4_1__)
# define _impl_ppz_SSE4_1
# define _impl_ppz_SIMD_HEADER <immintrin.h>
#endif
#if defined(__SSSE3__)
# define _impl_ppz_SSSE3
# define _impl_ppz_SIMD_HEADER <immintrin.h>
#endif
#if defined(__SSE3__)
# define _impl_ppz_SSE3
# define _impl_ppz_SIMD_HEADER <immintrin.h>
#endif
#if defined(__SSE2__)
# define _impl_ppz_SSE2
# define _impl_ppz_SIMD_HEADER <immintrin.h>
#endif
#if defined(__SSE__)
# define _impl_ppz_SSE
# define _impl_ppz_SIMD_HEADER <immintrin.h>
#endif

// #if defined(__ARM_NEON) or defined(__ARM_NEON__)
// # if defined(__ARM_ARCH_64)
// #   define _impl_ppz_NEON64
// #   define _impl_ppz_SIMD_HEADER <arm64_neon.h>
// # else
// #   define _impl_ppz_NEON
// #   define _impl_ppz_SIMD_HEADER <arm_neon.h>
// # endif
// #endif

#if defined(_impl_ppz_SIMD_HEADER)
# if not defined(__OPTIMIZE__)
#   define __OPTIMIZE__
#   include _impl_ppz_SIMD_HEADER
# else
#   include _impl_ppz_SIMD_HEADER
# endif
#endif

namespace Parallilos
{  
  template<typename T>
  class SIMD;

  template<typename T>
  class Array;

  namespace _io
  {
    static std::ostream log(std::clog.rdbuf()); // logging ostream
  }

  namespace _version
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
#   define _impl_ppz_INLINE __attribute__((always_inline)) inline
#   if (__cplusplus >= 201703L) and defined(_LIBCPP_HAS_C11_FEATURES)
#     define _impl_ppz_HAS_ALIGNED_ALLOC
#   endif
# elif defined(__GNUC__)
#   define _impl_ppz_INLINE __attribute__((always_inline)) inline
#   if (__cplusplus >= 201703L) and defined(_GLIBCXX_HAVE_ALIGNED_ALLOC)
#     define _impl_ppz_HAS_ALIGNED_ALLOC
#   endif
// # elif defined(__MINGW32__) or defined(__MINGW64__)
// #   define _impl_ppz_INLINE __attribute__((always_inline)) inline
// # elif defined(__apple_build_version__)
// #   define _impl_ppz_INLINE __attribute__((always_inline)) inline
// # elif defined(_MSC_VER)
// #   define _impl_ppz_INLINE __forceinline
// # elif defined(__INTEL_COMPILER)
// #   define _impl_ppz_SVML
// #   define _impl_ppz_INLINE __forceinline
// # elif defined(__ARMCC_VERSION)
// #   define _impl_ppz_INLINE __forceinline
# else
#   define _impl_ppz_INLINE inline
# endif

#   define _impl_ppz_PRAGMA(PRAGMA) _Pragma(#PRAGMA)
#   define _impl_ppz_IGNORE(WARNING, ...)                \
      _impl_ppz_PRAGMA(clang diagnostic push)            \
      _impl_ppz_PRAGMA(clang diagnostic ignored WARNING) \
      __VA_ARGS__                                         \
      _impl_ppz_PRAGMA(clang diagnostic pop)

// support from clang 12.0.0 and GCC 10.1 onward
# if defined(__clang__) and (__clang_major__ >= 12)
# if __cplusplus < 202002L
#   define _impl_ppz_HOT  _impl_ppz_IGNORE("-Wc++20-extensions", [[likely]])
#   define _impl_ppz_COLD _impl_ppz_IGNORE("-Wc++20-extensions", [[unlikely]])
# else
#   define _impl_ppz_HOT  [[likely]]
#   define _impl_ppz_COLD [[unlikely]]
# endif
# elif defined(__GNUC__) and (__GNUC__ >= 10)
#   define _impl_ppz_HOT  [[likely]]
#   define _impl_ppz_COLD [[unlikely]]
# else
#   define _impl_ppz_HOT
#   define _impl_ppz_COLD
# endif

# if __cplusplus >= 201402L
#   define _impl_ppz_CONSTEXPR_CPP14 constexpr
# else
#   define _impl_ppz_CONSTEXPR_CPP14
# endif

# if defined(_impl_ppz_THREADSAFE)
#   undef  _impl_ppz_THREADSAFE
#   define _impl_ppz_THREADLOCAL         thread_local
#   define _impl_ppz_DECLARE_MUTEX(...)  static std::mutex __VA_ARGS__
#   define _impl_ppz_DECLARE_LOCK(MUTEX) std::lock_guard<decltype(MUTEX)> _lock{MUTEX}
# else
#   define _impl_ppz_THREADLOCAL
#   define _impl_ppz_DECLARE_MUTEX(...)
#   define _impl_ppz_DECLARE_LOCK(MUTEX) void(0)
# endif

    template<size_t size>
    class _parallel
    {
    public:
      constexpr _parallel(const size_t n_elements) noexcept :
        _current_index(0), _passes_left(size ? (n_elements / size) : 0)
      {}
      constexpr                  size_t     operator*()                  noexcept {return _current_index;}
      _impl_ppz_CONSTEXPR_CPP14 void       operator++()                 noexcept {--_passes_left, _current_index += size;}
      constexpr                  bool       operator!=(const _parallel&) noexcept {return _passes_left;}
      _impl_ppz_CONSTEXPR_CPP14 _parallel& begin()                      noexcept {return *this;}
      constexpr                  _parallel  end()                        noexcept {return _parallel(0);}
    private:
      size_t _current_index;
      size_t _passes_left;
    public:
      const size_t passes = _passes_left;
    };

    template<size_t size>
    class _sequential
    {
    public:
      constexpr _sequential(const size_t n_elements) noexcept :
        _current_index(size ? ((n_elements / size) * size) : 0), _passes_left(n_elements - _current_index)
      {}
      constexpr                  size_t       operator*()                    noexcept {return _current_index;}
      _impl_ppz_CONSTEXPR_CPP14 void         operator++()                   noexcept {--_passes_left, ++_current_index;}
      constexpr                  bool         operator!=(const _sequential&) noexcept {return _passes_left;}
      _impl_ppz_CONSTEXPR_CPP14 _sequential& begin()                        noexcept {return *this;}
      constexpr                  _sequential  end()                          noexcept {return _sequential(0);}
    private:
      size_t _current_index;
      size_t _passes_left;
    public:
      const size_t passes = _passes_left;
    };

# if defined(PARALLILOS_LOGGING)
    static _impl_ppz_THREADLOCAL char _typename_buffer[32];

    template<typename T>
    _impl_ppz_CONSTEXPR_CPP14
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

    static _impl_ppz_THREADLOCAL char _log_buffer[256];
    _impl_ppz_DECLARE_MUTEX(_log_mtx);
    
#   define _impl_ppz_LOG(...)                                             \
      [&](const char* caller){                                            \
        sprintf(_backend::_log_buffer, __VA_ARGS__);                      \
        _impl_ppz_DECLARE_LOCK(_backend::_log_mtx);                       \
        _io::log << caller << ": " << _backend::_log_buffer << std::endl; \
      }(__func__)
# else
#   define _impl_ppz_LOG(...) void(0)
# endif

    template<typename T>
    using _if_integral = typename std::enable_if<std::is_integral<T>::value>::type;
  }
// ---------------------------------------------------------------------------------------------------------------------
  template<typename T>
  class SIMD
  {
  static_assert(std::is_arithmetic<T>::value, "T in SIMD<T> must be an arithmetic type.");
  public:
    static constexpr size_t size      = 0;
    static constexpr size_t alignment = 0;
    using Type = T;
    using Mask = bool;
    static constexpr const char* set = "no SIMD instruction set used for this type";

    static constexpr
    _backend::_parallel<size> parallel(const size_t n_elements) noexcept
    {
      return _backend::_parallel<size>(n_elements);
    }

    static constexpr
    _backend::_sequential<size> sequential(const size_t n_elements) noexcept
    {
      return _backend::_sequential<size>(n_elements);
    }
    
    SIMD() = delete;
  };

  // load a vector from unaligned data
  template<typename T>
  constexpr
  typename SIMD<T>::Type simd_loadu(const T data[]) noexcept
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T*>());
    return *data;
  }

  // load a vector from aligned data
  template<typename T>
  constexpr
  typename SIMD<T>::Type simd_loada(const T data[]) noexcept
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T*>());
    return *data;
  }

  // store a vector into unaligned memory
  template<typename T>
  constexpr
  void simd_storeu(T addr[], const typename SIMD<T>::Type& data)
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T*>());
    *addr = data;
  }

  // store a vector into aligned memory
  template<typename T>
  constexpr
  void simd_storea(T addr[], const typename SIMD<T>::Type& data)
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T*>());
    *addr = data;
  }

  // load a vector with zeros
  template<typename T>
  constexpr
  typename SIMD<T>::Type simd_setzero() noexcept
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return 0;
  }

  // load a vector with a specific value
  template<typename T>
  constexpr
  typename SIMD<T>::Type simd_setval(const T value) noexcept
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return value;
  }

  // [a] + [b]
  template<typename T>
  constexpr
  T simd_add(T a, T b)
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a + b;
  }

  // [a] * [b]
  template<typename T>
  constexpr
  T simd_mul(T a, T b)
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a * b;
  }

  // [a] - [b]
  template<typename T>
  constexpr
  T simd_sub(T a, T b)
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a - b;
  }

  // [a] / [b]
  template<typename T>
  constexpr
  T simd_div(T a, T b)
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a / b;
  }

  // sqrt([a])
  template<typename T>
  constexpr
  T simd_sqrt(T a)
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return std::sqrt(a);
  }

  // [a] + ([b] * [c])
  template<typename T>
  constexpr
  T simd_addmul(T a, T b, T c)
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a + b * c;
  }

  // [a] - ([b] * [c])
  template<typename T>
  constexpr
  T simd_submul(T a, T b, T c)
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a - b * c;
  }

  // [a] == [b]
  template<typename T>
  constexpr
  bool simd_eq(T a, T b) noexcept
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a == b;
  }

  // [a] != [b]
  template<typename T>
  constexpr
  bool simd_neq(T a, T b) noexcept
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a != b;
  }

  // [a] > [b]
  template<typename T>
  constexpr
  bool simd_gt(T a, T b) noexcept
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a > b;
  }

  // [a] >= [b]
  template<typename T>
  constexpr
  bool simd_gte(T a, T b) noexcept
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a >= b;
  }

  // [a] < [b]
  template<typename T>
  constexpr
  bool simd_lt(T a, T b) noexcept
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a < b;
  }

  // [a] <= [b]
  template<typename T>
  constexpr
  bool simd_lte(T a, T b) noexcept
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return a <= b;
  }

  inline namespace Bitwise
  {
    // ![a]
    template<typename T, typename = _backend::_if_integral<T>>
    constexpr
    T simd_not(T a) noexcept
    {
      _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return !a;
    }

    // [a] & [b]
    template<typename T, typename = _backend::_if_integral<T>>
    constexpr
    T simd_and(T a, T b) noexcept
    {
      _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return a & b;
    }

    // !([a] & [b])
    template<typename T, typename = _backend::_if_integral<T>>
    constexpr
    T simd_nand(T a, T b) noexcept
    {
      _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return !(a & b);
    }

    // [a] | [b]
    template<typename T, typename = _backend::_if_integral<T>>
    constexpr
    T simd_or(T a, T b) noexcept
    {
      _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return a | b;
    }

    // !([a] | [b])
    template<typename T, typename = _backend::_if_integral<T>>
    constexpr
    T simd_nor(T a, T b) noexcept
    {
      _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return !(a | b);
    }

    // [a] ^ [b]
    template<typename T, typename = _backend::_if_integral<T>>
    constexpr
    T simd_xor(T a, T b) noexcept
    {
      _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return a ^ b;
    }

    // !([a] ^ [b])
    template<typename T, typename = _backend::_if_integral<T>>
    constexpr
    T simd_xnor(T a, T b) noexcept
    {
      _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
      return !(a ^ b);
    }
  }

  // abs([a])
  template<typename T>
  constexpr
  T simd_abs(T a) noexcept
  {
    _impl_ppz_LOG("type \"%s\" is not SIMD-supported", _backend::_type_name<T>());
    return std::abs(a);
  }

  template<typename T>
  class Array final
  {
  static_assert(std::is_arithmetic<T>::value, "T in Array<T> must be an arithmetic type");
  public:
    constexpr const T* data(const size_t k = 0)   const noexcept { return _array + k; }
    _impl_ppz_CONSTEXPR_CPP14 T*       data(const size_t k = 0)         noexcept { return _array + k; }
    constexpr size_t   size()                     const noexcept { return _numel; }
    constexpr T        operator[](const size_t k) const noexcept { return _array[k]; }
    _impl_ppz_CONSTEXPR_CPP14 T&       operator[](const size_t k)       noexcept { return _array[k]; }
    constexpr operator T*()                             noexcept { return _array; }
    constexpr T*       release()                        noexcept
    {
      T* data = _array;
      _array  = nullptr;
      _numel  = 0;
      return data;
    }
    
    _impl_ppz_CONSTEXPR_CPP14
    Array(const size_t number_of_elements) noexcept :
      _array([number_of_elements]() -> T* {
        if ((number_of_elements == 0)) _impl_ppz_COLD
        {
          return nullptr;
        }

        if (SIMD<T>::alignment == 0) _impl_ppz_COLD
        {
          return reinterpret_cast<T*>(std::malloc(number_of_elements * sizeof(T)));
        }

#     if defined(_impl_ppz_HAS_ALIGNED_ALLOC)
        return reinterpret_cast<T*>(std::aligned_alloc(SIMD<T>::alignment, number_of_elements * sizeof(T)));
#     else
        void* memory_block = std::malloc(number_of_elements * sizeof(T) + SIMD<T>::alignment);

        if (memory_block == nullptr) _impl_ppz_COLD
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
      _numel(_array == nullptr ? 0 : number_of_elements)
    {
      _impl_ppz_LOG("created array of size %u", unsigned(_numel));
    }

    _impl_ppz_CONSTEXPR_CPP14
    Array(const std::initializer_list<T> initializer_list) noexcept :
      Array(initializer_list.size())
    {
      if (_numel != 0) _impl_ppz_HOT
      {
        size_t k = 0;
        for (T value : initializer_list)
        {
          _array[k++] = value;
        }
      }
    }
    
    _impl_ppz_CONSTEXPR_CPP14
    Array(Array&& other) noexcept :
      _array(other._array),
      _numel(other._numel)
    {
      _impl_ppz_LOG("moved array of size %u", unsigned(_numel));
      other._array = nullptr;
      other._numel = 0;
    }

    constexpr
    Array(const Array&) = delete; // copying is disallowed

    ~Array()
    {
      if (_array != nullptr) _impl_ppz_HOT
      {
#   if defined(_impl_ppz_HAS_ALIGNED_ALLOC)
        std::free(_array);
#   else
        if (SIMD<T>::alignment != 0) _impl_ppz_HOT
        {
          std::free(reinterpret_cast<void**>(_array)[-1]);
        }
        else std::free(_array);
#   endif
      }
      _impl_ppz_LOG("freed used memory");
    }

    // interpret aligned array as a vector, k is an offset in elements into the array
    typename SIMD<T>::Type& as_vector(const size_t k) noexcept
    {
      return reinterpret_cast<typename SIMD<T>::Type&>(_array[k]);
    }
  private:
    T*     _array;
    size_t _numel;
  };
// ---------------------------------------------------------------------------------------------------------------------
  namespace _backend
  {
    // T = type, V = vector type, M = mask type, A = alignment, S = sets used
#   define _impl_ppz_MAKE_SIMD_SPECIALIZATION(T, V, M, A, S)                                  \
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
        static constexpr                                                                       \
        _backend::_parallel<size> parallel(const size_t n_elements) noexcept                   \
        { return _backend::_parallel<size>{n_elements}; }                                      \
        static constexpr                                                                       \
        _backend::_sequential<size> sequential(const size_t n_elements) noexcept               \
        { return _backend::_sequential<size>{n_elements}; }                                    \
        SIMD() = delete;                                                                       \
      };
  }

#if defined(_impl_ppz_AVX512F)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define _impl_ppz_F32
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(float, __m512, __mmask16, 64, "AVX512F");
# define _impl_ppz_F32_LOADU(data)           _mm512_loadu_ps(data)
# define _impl_ppz_F32_LOADA(data)           _mm512_load_ps(data)
# define _impl_ppz_F32_STOREU(addr, data)    _mm512_storeu_ps((Toid*)addr, data)
# define _impl_ppz_F32_STOREA(addr, data)    _mm512_store_ps((Toid*)addr, data)
# define _impl_ppz_F32_SETVAL(value)         _mm512_set1_ps(value)
# define _impl_ppz_F32_SETZERO()             _mm512_setzero_ps()
# define _impl_ppz_F32_MUL(a, b)             _mm512_mul_ps(a, b)
# define _impl_ppz_F32_ADD(a, b)             _mm512_add_ps(a, b)
# define _impl_ppz_F32_SUB(a, b)             _mm512_sub_ps(a, b)
# define _impl_ppz_F32_DIV(a, b)             _mm512_div_ps(a, b)
# define _impl_ppz_F32_SQRT(a)               _mm512_sqrt_ps(a)
# define _impl_ppz_F32_ADDMUL(a, b, c)       _mm512_fmadd_ps(b, c, a)
# define _impl_ppz_F32_SUBMUL(a, b, c)       _mm512_fnmadd_ps(a, b, c)
# define _impl_ppz_F32_EQ(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_EQ_UQ)
# define _impl_ppz_F32_NEQ(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_NEQ_UQ)
# define _impl_ppz_F32_GT(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ)
# define _impl_ppz_F32_GTE(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ)
# define _impl_ppz_F32_LT(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ)
# define _impl_ppz_F32_LTE(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ)

# define _impl_ppz_F32_ABS(a)                _mm512_abs_ps(a)
#elif defined(_impl_ppz_FMA)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define _impl_ppz_F32
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(float, __m256, __m256, 32, "AVX, FMA");
# define _impl_ppz_F32_LOADU(data)           _mm256_loadu_ps(data)
# define _impl_ppz_F32_LOADA(data)           _mm256_load_ps(data)
# define _impl_ppz_F32_STOREU(addr, data)    _mm256_storeu_ps(addr, data)
# define _impl_ppz_F32_STOREA(addr, data)    _mm256_store_ps(addr, data)
# define _impl_ppz_F32_SETVAL(value)         _mm256_set1_ps(value)
# define _impl_ppz_F32_SETZERO()             _mm256_setzero_ps()
# define _impl_ppz_F32_MUL(a, b)             _mm256_mul_ps(a, b)
# define _impl_ppz_F32_ADD(a, b)             _mm256_add_ps(a, b)
# define _impl_ppz_F32_SUB(a, b)             _mm256_sub_ps(a, b)
# define _impl_ppz_F32_DIV(a, b)             _mm256_div_ps(a, b)
# define _impl_ppz_F32_SQRT(a)               _mm256_sqrt_ps(a)
# define _impl_ppz_F32_ADDMUL(a, b, c)       _mm256_fmadd_ps(b, c, a)
# define _impl_ppz_F32_SUBMUL(a, b, c)       _mm256_fnmadd_ps(a, b, c)
# define _impl_ppz_F32_EQ(a, b)              _mm256_cmp_ps(a, b, _CMP_EQ_UQ)
# define _impl_ppz_F32_NEQ(a, b)             _mm256_cmp_ps(a, b, _CMP_NEQ_UQ)
# define _impl_ppz_F32_GT(a, b)              _mm256_cmp_ps(a, b, _CMP_GT_OQ)
# define _impl_ppz_F32_GTE(a, b)             _mm256_cmp_ps(a, b, _CMP_GE_OQ)
# define _impl_ppz_F32_LT(a, b)              _mm256_cmp_ps(a, b, _CMP_LT_OQ)
# define _impl_ppz_F32_LTE(a, b)             _mm256_cmp_ps(a, b, _CMP_LE_OQ)
# define _impl_ppz_F32_ABS(a)                _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a)
#elif defined(_impl_ppz_AVX)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define _impl_ppz_F32
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(float, __m256, __m256, 32, "AVX");
# define _impl_ppz_F32_LOADU(data)           _mm256_loadu_ps(data)
# define _impl_ppz_F32_LOADA(data)           _mm256_load_ps(data)
# define _impl_ppz_F32_STOREU(addr, data)    _mm256_storeu_ps(addr, data)
# define _impl_ppz_F32_STOREA(addr, data)    _mm256_store_ps(addr, data)
# define _impl_ppz_F32_SETVAL(value)         _mm256_set1_ps(value)
# define _impl_ppz_F32_SETZERO()             _mm256_setzero_ps()
# define _impl_ppz_F32_MUL(a, b)             _mm256_mul_ps(a, b)
# define _impl_ppz_F32_ADD(a, b)             _mm256_add_ps(a, b)
# define _impl_ppz_F32_SUB(a, b)             _mm256_sub_ps(a, b)
# define _impl_ppz_F32_DIV(a, b)             _mm256_div_ps(a, b)
# define _impl_ppz_F32_SQRT(a)               _mm256_sqrt_ps(a)
# define _impl_ppz_F32_ADDMUL(a, b, c)       _mm256_add_ps(a, _mm256_mul_ps(b, c))
# define _impl_ppz_F32_SUBMUL(a, b, c)       _mm256_sub_ps(a, _mm256_mul_ps(b, c))
# define _impl_ppz_F32_EQ(a, b)              _mm256_cmp_ps(a, b, _CMP_EQ_UQ)
# define _impl_ppz_F32_NEQ(a, b)             _mm256_cmp_ps(a, b, _CMP_NEQ_UQ)
# define _impl_ppz_F32_GT(a, b)              _mm256_cmp_ps(a, b, _CMP_GT_OQ)
# define _impl_ppz_F32_GTE(a, b)             _mm256_cmp_ps(a, b, _CMP_GE_OQ)
# define _impl_ppz_F32_LT(a, b)              _mm256_cmp_ps(a, b, _CMP_LT_OQ)
# define _impl_ppz_F32_LTE(a, b)             _mm256_cmp_ps(a, b, _CMP_LE_OQ)
# define _impl_ppz_F32_ABS(a)                _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a)
#elif defined(_impl_ppz_SSE)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define _impl_ppz_F32
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(float, __m128, __m128, 16, "SSE");
# define _impl_ppz_F32_LOADU(data)           _mm_loadu_ps(data)
# define _impl_ppz_F32_LOADA(data)           _mm_load_ps(data)
# define _impl_ppz_F32_STOREU(addr, data)    _mm_storeu_ps(addr, data)
# define _impl_ppz_F32_STOREA(addr, data)    _mm_store_ps(addr, data)
# define _impl_ppz_F32_SETVAL(value)         _mm_set1_ps(value)
# define _impl_ppz_F32_SETZERO()             _mm_setzero_ps()
# define _impl_ppz_F32_MUL(a, b)             _mm_mul_ps(a, b)
# define _impl_ppz_F32_ADD(a, b)             _mm_add_ps(a, b)
# define _impl_ppz_F32_SUB(a, b)             _mm_sub_ps(a, b)
# define _impl_ppz_F32_DIV(a, b)             _mm_div_ps(a, b)
# define _impl_ppz_F32_SQRT(a)               _mm_sqrt_ps(a)
# define _impl_ppz_F32_ADDMUL(a, b, c)       _mm_add_ps(a, _mm_mul_ps(b, c))
# define _impl_ppz_F32_SUBMUL(a, b, c)       _mm_sub_ps(a, _mm_mul_ps(b, c))
# define _impl_ppz_F32_EQ(a, b)              _mm_cmpeq_ps (a, b)
# define _impl_ppz_F32_NEQ(a, b)             _mm_cmpneq_ps (a, b)
# define _impl_ppz_F32_GT(a, b)              _mm_cmpgt_ps(a, b)
# define _impl_ppz_F32_GTE(a, b)             _mm_cmpge_ps(a, b)
# define _impl_ppz_F32_LT(a, b)              _mm_cmplt_ps(a, b)
# define _impl_ppz_F32_LTE(a, b)             _mm_cmple_ps(a, b)
# define _impl_ppz_F32_ABS(a)                _mm_andnot_ps(_mm_set1_ps(-0.0f), a)
// #elif defined(_impl_ppz_NEON) or defined(_impl_ppz_NEON64)
//   static_assert(sizeof(float) == 4, "float must be 32 bit");
// # define _impl_ppz_F32
//   _impl_ppz_MAKE_SIMD_SPECIALIZATION(float, float32x4_t, uint32x4_t, 0, "NEON");
// # define _impl_ppz_F32_LOADU(data)           vld1q_f32(data)
// # define _impl_ppz_F32_LOADA(data)           vld1q_f32(data)
// # define _impl_ppz_F32_STOREU(addr, data)    vst1q_f32(addr, data)
// # define _impl_ppz_F32_STOREA(addr, data)    vst1q_f32(addr, data)
// # define _impl_ppz_F32_SETVAL(value)         vdupq_n_f32(value)
// # define _impl_ppz_F32_SETZERO()             vdupq_n_f32(0.0f)
// # define _impl_ppz_F32_MUL(a, b)             vmulq_f32(a, b)
// # define _impl_ppz_F32_ADD(a, b)             vaddq_f32(a, b)
// # define _impl_ppz_F32_SUB(a, b)             vsubq_f32(a, b)
// # define _impl_ppz_F32_DIV(a, b)             vdivq_f32(a, b)
// # define _impl_ppz_F32_SQRT(a)               vsqrtq_f32(a)
// # define _impl_ppz_F32_ADDMUL(a, b, c)       vmlaq_f32(a, b, c)
// # define _impl_ppz_F32_SUBMUL(a, b, c)       vmlsq_f32(a, b, c)
// # define _impl_ppz_F32_EQ(a, b)              vceqq_f32(a, b)
// # define _impl_ppz_F32_NEQ(a, b)             vmvnq_u32(Tceqq_f32(a, b))
// # define _impl_ppz_F32_GT(a, b)              vcgtq_f32(a, b)
// # define _impl_ppz_F32_GTE(a, b)             vcgeq_f32(a, b)
// # define _impl_ppz_F32_LT(a, b)              vcltq_f32(a, b)
// # define _impl_ppz_F32_LTE(a, b)             vcleq_f32(a, b)

// # define _impl_ppz_F32_ABS(a)                vabsq_f32(a)
#endif

#ifdef _impl_ppz_F32
  // load a vector with zeros
  template<>
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_setzero<float>() noexcept
  {
    return _impl_ppz_F32_SETZERO();
#   undef  _impl_ppz_F32_SETZERO
  }

  // load a vector from unaligned data
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_loadu(const float data[]) noexcept
  {
    return _impl_ppz_F32_LOADU(data);
#   undef  _impl_ppz_F32_LOADU
  }

  // load a vector from aligned data
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_loada(const float data[]) noexcept
  {
    return _impl_ppz_F32_LOADA(data);
#   undef  _impl_ppz_F32_LOADA
  }

  // store a vector into unaligned memory
  _impl_ppz_INLINE 
  void simd_storeu(float addr[], const SIMD<float>::Type& data)
  {
    _impl_ppz_F32_STOREU(addr, data);
#   undef _impl_ppz_F32_STOREU
  }

  // store a vector into aligned memory
  _impl_ppz_INLINE 
  void simd_storea(float addr[], const SIMD<float>::Type& data)
  {
    _impl_ppz_F32_STOREA(addr, data);
#   undef _impl_ppz_F32_STOREA
  }

  // load a vector with a specific value
  template<>
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_setval(const float value) noexcept
  {
    return _impl_ppz_F32_SETVAL(value);
#   undef  _impl_ppz_F32_SETVAL
  }

  // [a] + [b]
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_add(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _impl_ppz_F32_ADD(a, b);
#   undef  _impl_ppz_F32_ADD
  }

  // [a] * [b]
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_mul(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _impl_ppz_F32_MUL(a, b);
#   undef  _impl_ppz_F32_MUL
  }

  // [a] - [b]
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_sub(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _impl_ppz_F32_SUB(a, b);
#   undef  _impl_ppz_F32_SUB
  }

  // [a] / [b]
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_div(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _impl_ppz_F32_DIV(a, b);
#   undef  _impl_ppz_F32_DIV
  }

  // sqrt([a])
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_sqrt(const SIMD<float>::Type& a) noexcept
  {
    return _impl_ppz_F32_SQRT(a);
#   undef  _impl_ppz_F32_SQRT
  }

  // [a] + ([b] * [c])
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_addmul(const SIMD<float>::Type& a, const SIMD<float>::Type& b, const SIMD<float>::Type& c) noexcept
  {
    return _impl_ppz_F32_ADDMUL(a, b, c);
#   undef  _impl_ppz_F32_ADDMUL
  }

  // [a] - ([b] * [c])
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_submul(const SIMD<float>::Type& a, const SIMD<float>::Type& b, const SIMD<float>::Type& c) noexcept
  {
    return _impl_ppz_F32_SUBMUL(a, b, c);
#   undef  _impl_ppz_F32_SUBMUL
  }

  // [a] == [b]
  _impl_ppz_INLINE 
  SIMD<float>::Mask simd_eq(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _impl_ppz_F32_EQ(a, b);
#   undef  _impl_ppz_F32_EQ
  }

  // [a] != [b]
  _impl_ppz_INLINE 
  SIMD<float>::Mask simd_neq(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _impl_ppz_F32_NEQ(a, b);
#   undef  _impl_ppz_F32_NEQ
  }

  // [a] > [b]
  _impl_ppz_INLINE 
  SIMD<float>::Mask simd_gt(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _impl_ppz_F32_GT(a, b);
#   undef  _impl_ppz_F32_GT
  }

  // [a] >= [b]
  _impl_ppz_INLINE 
  SIMD<float>::Mask simd_gte(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _impl_ppz_F32_GTE(a, b);
#   undef  _impl_ppz_F32_GTE
  }

  // [a] < [b]
  _impl_ppz_INLINE 
  SIMD<float>::Mask simd_lt(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _impl_ppz_F32_LT(a, b);
#   undef  _impl_ppz_F32_LT
  }

  // [a] <= [b]
  _impl_ppz_INLINE 
  SIMD<float>::Mask simd_lte(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _impl_ppz_F32_LTE(a, b);
#   undef  _impl_ppz_F32_LTE
  }

  // abs([a])
  _impl_ppz_INLINE 
  SIMD<float>::Type simd_abs(const SIMD<float>::Type& a) noexcept
  {
    return _impl_ppz_F32_ABS(a);
#   undef  _impl_ppz_F32_ABS
  }

  inline _impl_ppz_CONSTEXPR_CPP14
  std::ostream& operator<<(std::ostream& ostream, const SIMD<float>::Type& vector) noexcept
  {
    for (unsigned k = 0; k < SIMD<float>::size; ++k)
    {
      if (k != 0) _impl_ppz_HOT
      {
        ostream << ' ';
      }

      ostream << reinterpret_cast<const float*>(&vector)[k];
    }

    return ostream;
  }
#endif

#if defined(_impl_ppz_AVX512F)
# define _impl_ppz_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(double, __m512d, __mmask8, 64, "AVX512F");
# define _impl_ppz_F64_LOADU(data)           _mm512_loadu_pd(data)
# define _impl_ppz_F64_LOADA(data)           _mm512_load_pd(data)
# define _impl_ppz_F64_STOREU(addr, data)    _mm512_storeu_pd((Toid*)addr, data)
# define _impl_ppz_F64_STOREA(addr, data)    _mm512_store_pd((Toid*)addr, data)
# define _impl_ppz_F64_SETVAL(value)         _mm512_set1_pd(value)
# define _impl_ppz_F64_SETZERO()             _mm512_setzero_pd()
# define _impl_ppz_F64_MUL(a, b)             _mm512_mul_pd(a, b)
# define _impl_ppz_F64_ADD(a, b)             _mm512_add_pd(a, b)
# define _impl_ppz_F64_SUB(a, b)             _mm512_sub_pd(a, b)
# define _impl_ppz_F64_DIV(a, b)             _mm512_div_pd(a, b)
# define _impl_ppz_F64_SQRT(a)               _mm512_sqrt_pd(a)
# define _impl_ppz_F64_ADDMUL(a, b, c)       _mm512_fmadd_pd(b, c, a)
# define _impl_ppz_F64_SUBMUL(a, b, c)       _mm512_fnmadd_pd(a, b, c)
# define _impl_ppz_F64_EQ(a, b)              _mm512_cmp_pd_mask(a, b, _CMP_EQ_UQ)
# define _impl_ppz_F64_NEQ(a, b)             _mm512_cmp_pd_mask(a, b, _CMP_NEQ_UQ)
# define _impl_ppz_F64_GT(a, b)              _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ)
# define _impl_ppz_F64_GTE(a, b)             _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ)
# define _impl_ppz_F64_LT(a, b)              _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ)
# define _impl_ppz_F64_LTE(a, b)             _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ)

# define _impl_ppz_F64_ABS(a)                _mm512_abs_pd(a)
#elif defined(_impl_ppz_FMA)
# define _impl_ppz_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(double, __m256d, __m256d, 32, "AVX, FMA");
# define _impl_ppz_F64_LOADU(data)           _mm256_loadu_pd(data)
# define _impl_ppz_F64_LOADA(data)           _mm256_load_pd(data)
# define _impl_ppz_F64_STOREU(addr, data)    _mm256_storeu_pd(addr, data)
# define _impl_ppz_F64_STOREA(addr, data)    _mm256_store_pd(addr, data)
# define _impl_ppz_F64_SETVAL(value)         _mm256_set1_pd(value)
# define _impl_ppz_F64_SETZERO()             _mm256_setzero_pd()
# define _impl_ppz_F64_MUL(a, b)             _mm256_mul_pd(a, b)
# define _impl_ppz_F64_ADD(a, b)             _mm256_add_pd(a, b)
# define _impl_ppz_F64_SUB(a, b)             _mm256_sub_pd(a, b)
# define _impl_ppz_F64_DIV(a, b)             _mm256_div_pd(a, b)
# define _impl_ppz_F64_SQRT(a)               _mm256_sqrt_pd(a)
# define _impl_ppz_F64_ADDMUL(a, b, c)       _mm256_fmadd_pd(b, c, a)
# define _impl_ppz_F64_SUBMUL(a, b, c)       _mm256_fnmadd_pd(a, b, c)
# define _impl_ppz_F64_EQ(a, b)              _mm256_cmp_pd(a, b, _CMP_EQ_UQ)
# define _impl_ppz_F64_NEQ(a, b)             _mm256_cmp_pd(a, b, _CMP_NEQ_UQ)
# define _impl_ppz_F64_GT(a, b)              _mm256_cmp_pd(a, b, _CMP_GT_OQ)
# define _impl_ppz_F64_GTE(a, b)             _mm256_cmp_pd(a, b, _CMP_GE_OQ)
# define _impl_ppz_F64_LT(a, b)              _mm256_cmp_pd(a, b, _CMP_LT_OQ)
# define _impl_ppz_F64_LTE(a, b)             _mm256_cmp_pd(a, b, _CMP_LE_OQ)

# define _impl_ppz_F64_ABS(a)                _mm256_andnot_pd(_mm256_set1_pd(-0.0), a)
#elif defined(_impl_ppz_AVX)
# define _impl_ppz_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(double, __m256d, __m256d, 32, "AVX");
# define _impl_ppz_F64_LOADU(data)           _mm256_loadu_pd(data)
# define _impl_ppz_F64_LOADA(data)           _mm256_load_pd(data)
# define _impl_ppz_F64_STOREU(addr, data)    _mm256_storeu_pd(addr, data)
# define _impl_ppz_F64_STOREA(addr, data)    _mm256_store_pd(addr, data)
# define _impl_ppz_F64_SETVAL(value)         _mm256_set1_pd(value)
# define _impl_ppz_F64_SETZERO()             _mm256_setzero_pd()
# define _impl_ppz_F64_MUL(a, b)             _mm256_mul_pd(a, b)
# define _impl_ppz_F64_ADD(a, b)             _mm256_add_pd(a, b)
# define _impl_ppz_F64_SUB(a, b)             _mm256_sub_pd(a, b)
# define _impl_ppz_F64_DIV(a, b)             _mm256_div_pd(a, b)
# define _impl_ppz_F64_SQRT(a)               _mm256_sqrt_pd(a)
# define _impl_ppz_F64_ADDMUL(a, b, c)       _mm256_add_pd(a, _mm256_mul_pd(b, c))
# define _impl_ppz_F64_SUBMUL(a, b, c)       _mm256_sub_pd(a, _mm256_mul_pd(b, c))
# define _impl_ppz_F64_EQ(a, b)              _mm256_cmp_pd(a, b, _CMP_EQ_UQ)
# define _impl_ppz_F64_NEQ(a, b)             _mm256_cmp_pd(a, b, _CMP_NEQ_UQ)
# define _impl_ppz_F64_GT(a, b)              _mm256_cmp_pd(a, b, _CMP_GT_OQ)
# define _impl_ppz_F64_GTE(a, b)             _mm256_cmp_pd(a, b, _CMP_GE_OQ)
# define _impl_ppz_F64_LT(a, b)              _mm256_cmp_pd(a, b, _CMP_LT_OQ)
# define _impl_ppz_F64_LTE(a, b)             _mm256_cmp_pd(a, b, _CMP_LE_OQ)

# define _impl_ppz_F64_ABS(a)                _mm256_andnot_pd(_mm256_set1_pd(-0.0), a)
#elif defined(_impl_ppz_SSE2)
# define _impl_ppz_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(double, __m128d, __m128d, 16, "SSE2");
# define _impl_ppz_F64_LOADU(data)           _mm_loadu_pd(data)
# define _impl_ppz_F64_LOADA(data)           _mm_load_pd(data)
# define _impl_ppz_F64_STOREU(addr, data)    _mm_storeu_pd(addr, data)
# define _impl_ppz_F64_STOREA(addr, data)    _mm_store_pd(addr, data)
# define _impl_ppz_F64_SETVAL(value)         _mm_set1_pd(value)
# define _impl_ppz_F64_SETZERO()             _mm_setzero_pd()
# define _impl_ppz_F64_MUL(a, b)             _mm_mul_pd(a, b)
# define _impl_ppz_F64_ADD(a, b)             _mm_add_pd(a, b)
# define _impl_ppz_F64_SUB(a, b)             _mm_sub_pd(a, b)
# define _impl_ppz_F64_DIV(a, b)             _mm_div_pd(a, b)
# define _impl_ppz_F64_SQRT(a)               _mm_sqrt_pd(a)
# define _impl_ppz_F64_ADDMUL(a, b, c)       _mm_add_pd(a, _mm_mul_pd(b, c))
# define _impl_ppz_F64_SUBMUL(a, b, c)       _mm_sub_pd(a, _mm_mul_pd(b, c))
# define _impl_ppz_F64_EQ(a, b)              _mm_cmpeq_pd(a, b)
# define _impl_ppz_F64_NEQ(a, b)             _mm_cmpneq_pd(a, b)
# define _impl_ppz_F64_GT(a, b)              _mm_cmpgt_pd(a, b)
# define _impl_ppz_F64_GTE(a, b)             _mm_cmpge_pd(a, b)
# define _impl_ppz_F64_LT(a, b)              _mm_cmplt_pd(a, b)
# define _impl_ppz_F64_LTE(a, b)             _mm_cmple_pd(a, b)

# define _impl_ppz_F64_ABS(a)                _mm_andnot_pd(_mm_set1_pd(-0.0), a)
// #elif defined(_impl_ppz_NEON) or defined(_impl_ppz_NEON64)
// # define _impl_ppz_F64
//   static_assert(sizeof(double) == 8, "float must be 64 bit");
//   _impl_ppz_MAKE_SIMD_SPECIALIZATION(double, float64x4_t, float64x4_t, 0, "NEON");
// # define _impl_ppz_F64_LOADU(data)           vld1q_f64(data)
// # define _impl_ppz_F64_LOADA(data)           vld1q_f64(data)
// # define _impl_ppz_F64_STOREU(addr, data)    vst1q_f64(addr, data)
// # define _impl_ppz_F64_STOREA(addr, data)    vst1q_f64(addr, data)
// # define _impl_ppz_F64_SETVAL(value)         vdupq_n_f64(value)
// # define _impl_ppz_F64_SETZERO()             vdupq_n_f64(0.0)
// # define _impl_ppz_F64_MUL(a, b)             vmulq_f64(a, b)
// # define _impl_ppz_F64_ADD(a, b)             vaddq_f64(a, b)
// # define _impl_ppz_F64_SUB(a, b)             vsubq_f64(a, b)
// # define _impl_ppz_F64_DIV(a, b)             vdivq_f64(a, b)
// # define _impl_ppz_F64_SQRT(a)               vsqrtq_f64(a)
// # define _impl_ppz_F64_ADDMUL(a, b, c)       vmlaq_f64(a, b, c)
// # define _impl_ppz_F64_SUBMUL(a, b, c)       vmlsq_f64(a, b, c)
// # define _impl_ppz_F32_EQ(a, b)              vceqq_f64(a, b)
// # define _impl_ppz_F32_NEQ(a, b)             vmvnq_u64(vceqq_f64(a, b))
// # define _impl_ppz_F32_GT(a, b)              vcgtq_f64(a, b)
// # define _impl_ppz_F32_GTE(a, b)             vcgeq_f64(a, b)
// # define _impl_ppz_F32_LT(a, b)              vcltq_f64(a, b)
// # define _impl_ppz_F32_LTE(a, b)             vcleq_f64(a, b)

// # define _impl_ppz_F32_ABS(a)                vabsq_f64(a)
#endif

#ifdef _impl_ppz_F64
  // load a vector with zeros
  template<>
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_setzero<double>() noexcept
  {
    return _impl_ppz_F64_SETZERO();
#   undef  _impl_ppz_F64_SETZERO
  }

  // load a vector from unaligned data
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_loadu(const double data[]) noexcept
  {
    return _impl_ppz_F64_LOADU(data);
#   undef  _impl_ppz_F64_LOADU
  }

  // load a vector from aligned data
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_loada(const double data[]) noexcept
  {
    return _impl_ppz_F64_LOADA(data);
#   undef  _impl_ppz_F64_LOADA
  }

  // store a vector into unaligned memory
  _impl_ppz_INLINE 
  void simd_storeu(double addr[], const SIMD<double>::Type& data)
  {
    _impl_ppz_F64_STOREU(addr, data);
#   undef _impl_ppz_F64_STOREU
  }

  // store a vector into aligned memory
  _impl_ppz_INLINE 
  void simd_storea(double addr[], const SIMD<double>::Type& data)
  {
    _impl_ppz_F64_STOREA(addr, data);
#   undef _impl_ppz_F64_STOREA
  }

  // load a vector with a specific value
  template<>
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_setval(const double value) noexcept
  {
    return _impl_ppz_F64_SETVAL(value);
#   undef  _impl_ppz_F64_SETVAL
  }

  // [a] + [b]
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_add(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _impl_ppz_F64_ADD(a, b);
#   undef  _impl_ppz_F64_ADD
  }

  // [a] * [b]
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_mul(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _impl_ppz_F64_MUL(a, b);
#   undef  _impl_ppz_F64_MUL
  }

  // [a] - [b]
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_sub(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _impl_ppz_F64_SUB(a, b);
#   undef  _impl_ppz_F64_SUB
  }

  // [a] / [b]
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_div(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _impl_ppz_F64_DIV(a, b);
#   undef  _impl_ppz_F64_DIV
  }

  // sqrt([a])
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_sqrt(const SIMD<double>::Type& a) noexcept
  {
    return _impl_ppz_F64_SQRT(a);
#   undef  _impl_ppz_F64_SQRT
  }

  // [a] + ([b] * [c])
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_addmul(const SIMD<double>::Type& a, const SIMD<double>::Type& b, const SIMD<double>::Type& c) noexcept
  {
    return _impl_ppz_F64_ADDMUL(a, b, c);
#   undef  _impl_ppz_F64_ADDMUL
  }

  // [a] - ([b] * [c])
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_submul(const SIMD<double>::Type& a, const SIMD<double>::Type& b, const SIMD<double>::Type& c) noexcept
  {
    return _impl_ppz_F64_SUBMUL(a, b, c);
#   undef  _impl_ppz_F64_SUBMUL
  }

  // [a] == [b]
  _impl_ppz_INLINE 
  SIMD<double>::Mask simd_eq(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _impl_ppz_F64_EQ(a, b);
#   undef  _impl_ppz_F64_EQ
  }

  // [a] != [b]
  _impl_ppz_INLINE 
  SIMD<double>::Mask simd_neq(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _impl_ppz_F64_NEQ(a, b);
#   undef  _impl_ppz_F64_NEQ
  }

  // [a] > [b]
  _impl_ppz_INLINE 
  SIMD<double>::Mask simd_gt(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _impl_ppz_F64_GT(a, b);
#   undef  _impl_ppz_F64_GT
  }

  // [a] >= [b]
  _impl_ppz_INLINE 
  SIMD<double>::Mask simd_gte(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _impl_ppz_F64_GTE(a, b);
#   undef  _impl_ppz_F64_GTE
  }

  // [a] < [b]
  _impl_ppz_INLINE 
  SIMD<double>::Mask simd_lt(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _impl_ppz_F64_LT(a, b);
#   undef  _impl_ppz_F64_LT
  }

  // [a] <= [b]
  _impl_ppz_INLINE 
  SIMD<double>::Mask simd_lte(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _impl_ppz_F64_LTE(a, b);
#   undef  _impl_ppz_F64_LTE
  }

  // abs([a])
  _impl_ppz_INLINE 
  SIMD<double>::Type simd_abs(const SIMD<double>::Type& a) noexcept
  {
    return _impl_ppz_F64_ABS(a);
#   undef  _impl_ppz_F64_ABS
  }

  inline _impl_ppz_CONSTEXPR_CPP14
  std::ostream& operator<<(std::ostream& ostream, const SIMD<double>::Type& vector) noexcept
  {
    for (unsigned k = 0; k < SIMD<double>::size; ++k)
    {
      if (k != 0) _impl_ppz_HOT
      {
        ostream << ' ';
      }

      ostream << reinterpret_cast<const double*>(&vector)[k];
    }

    return ostream;
  }
#endif

#if defined(_impl_ppz_AVX512F)
# define _impl_ppz_I32
  static_assert(sizeof(int32_t) == 4, "int32_t must be 32 bit");
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(int32_t, __m512i, __mmask16, 64, "AVX512F");
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(uint32_t, __m512i, __mmask16, 64, "AVX512F");
# define _impl_ppz_I32_LOADU(data)           _mm512_loadu_si512(data)
# define _impl_ppz_I32_LOADA(data)           _mm512_load_si512(data)
# define _impl_ppz_I32_STOREU(addr, data)    _mm512_storeu_si512((Toid*)addr, data)
# define _impl_ppz_I32_STOREA(addr, data)    _mm512_store_si512((Toid*)addr, data)
# define _impl_ppz_I32_SETVAL(value)         _mm512_set1_epi32(value)
# define _impl_ppz_I32_SETZERO()             _mm512_setzero_si512()
# define _impl_ppz_I32_MUL(a, b)             _mm512_mullo_epi32 (a, b)
# define _impl_ppz_I32_ADD(a, b)             _mm512_add_epi32(a, b)
# define _impl_ppz_I32_SUB(a, b)             _mm512_sub_epi32(a, b)
# define _impl_ppz_I32_DIV(a, b)             _mm512_cvtps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(a), _mm512_cvtepi32_ps(b)))
# define _impl_ppz_I32_SQRT(a)               _mm512_cvtps_epi32(_mm512_sqrt_ps(_mm512_cvtepi32_ps(a)))
# define _impl_ppz_I32_ADDMUL(a, b, c)       _mm512_add_epi32(a, _mm512_mul_epi32(b, c))
# define _impl_ppz_I32_SUBMUL(a, b, c)       _mm512_sub_epi32(a, _mm512_mul_epi32(b, c))
# if defined(_impl_ppz_SVML)
#   undef  _impl_ppz_I32_DIV
#   define _impl_ppz_I32_DIV(a, b)           _mm512_div_epi32(a, b)
# endif

# define _impl_ppz_I32_EQ(a, b)              _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_EQ)
# define _impl_ppz_I32_NEQ(a, b)             _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NE)
# define _impl_ppz_I32_GT(a, b)              _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLE)
# define _impl_ppz_I32_GTE(a, b)             _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLT)
# define _impl_ppz_I32_LT(a, b)              _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LT)
# define _impl_ppz_I32_LTE(a, b)             _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LE)

# define _impl_ppz_I32_BW_NOT(a)             _mm512_xor_si512(a, _mm512_set1_epi32(-1))
# define _impl_ppz_I32_BW_AND(a, b)          _mm512_and_si512(a, b)
# define _impl_ppz_I32_BW_NAND(a, b)         _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_set1_epi32(-1))
# define _impl_ppz_I32_BW_OR(a, b)           _mm512_or_si512(a, b)
# define _impl_ppz_I32_BW_NOR(a, b)          _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_set1_epi32(-1))
# define _impl_ppz_I32_BW_XOR(a, b)          _mm512_xor_si512(a, b)
# define _impl_ppz_I32_BW_XNOR(a, b)         _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_set1_epi32(-1))

# define _impl_ppz_I32_ABS(a)                _mm512_abs_epi32(a)
#elif defined(_impl_ppz_AVX2)
# define _impl_ppz_I32
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(int32_t, __m256i, __m256i, 32, "AVX2, AVX");
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(uint32_t, __m256i, __m256i, 32, "AVX2, AVX");
# define _impl_ppz_I32_LOADU(data)           _mm256_loadu_si256((const __m256i*)data)
# define _impl_ppz_I32_LOADA(data)           _mm256_load_si256((const __m256i*)data)
# define _impl_ppz_I32_STOREU(addr, data)    _mm256_storeu_si256 ((__m256i*)addr, data)
# define _impl_ppz_I32_STOREA(addr, data)    _mm256_store_si256((__m256i*)addr, data)
# define _impl_ppz_I32_SETVAL(value)         _mm256_set1_epi32(value)
# define _impl_ppz_I32_SETZERO()             _mm256_setzero_si256()
# define _impl_ppz_I32_MUL(a, b)             _mm256_mullo_epi32(a, b)
# define _impl_ppz_I32_ADD(a, b)             _mm256_add_epi32(a, b)
# define _impl_ppz_I32_SUB(a, b)             _mm256_sub_epi32(a, b)
# define _impl_ppz_I32_DIV(a, b)             _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(a), _mm256_cvtepi32_ps(b)))
# define _impl_ppz_I32_SQRT(a)               _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(a)))
# define _impl_ppz_I32_ADDMUL(a, b, c)       _mm256_add_epi32(a, _mm256_mul_epi32(b, c))
# define _impl_ppz_I32_SUBMUL(a, b, c)       _mm256_sub_epi32(a, _mm256_mul_epi32(b, c))
# if defined(_impl_ppz_SVML)
#   undef  _impl_ppz_I32_DIV
#   define _impl_ppz_I32_DIV(a, b)           _mm256_div_epi32(a, b)
# endif

# define _impl_ppz_I32_EQ(a, b)              _mm256_cmpeq_epi32(a, b)
# define _impl_ppz_I32_NEQ(a, b)             _mm256_xor_si256(_mm256_cmpeq_epi32(a, b), _mm256_cmpeq_epi32(a, a))
# define _impl_ppz_I32_GT(a, b)              _mm256_cmpgt_epi32(a, b)
# define _impl_ppz_I32_GTE(a, b)             _mm256_xor_si256(_mm256_cmpgt_epi32(b, a), _mm256_cmpeq_epi32(a, a))
# define _impl_ppz_I32_LT(a, b)              _mm256_cmpgt_epi32(b, a)
# define _impl_ppz_I32_LTE(a, b)             _mm256_xor_si256(_mm256_cmpgt_epi32(a, b), _mm256_cmpeq_epi32(a, a))

# define _impl_ppz_I32_BW_NOT(a)             _mm256_xor_si256(a, _mm256_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_AND(a, b)          _mm256_and_si256(a, b)
# define _impl_ppz_I32_BW_NAND(a, b)         _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_OR(a, b)           _mm256_or_si256(a, b)
# define _impl_ppz_I32_BW_NOR(a, b)          _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_XOR(a, b)          _mm256_xor_si256(a, b)
# define _impl_ppz_I32_BW_XNOR(a, b)         _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_cmpeq_epi32(a, a))

# define _impl_ppz_I32_ABS(a)                _mm256_abs_epi32(a)
#elif defined(_impl_ppz_SSE4_1)
# define _impl_ppz_I32
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(int32_t, __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(uint32_t, __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
# define _impl_ppz_I32_LOADU(data)           _mm_loadu_si128((const __m128i*)data)
# define _impl_ppz_I32_LOADA(data)           _mm_load_si128((const __m128i*)data)
# define _impl_ppz_I32_STOREU(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
# define _impl_ppz_I32_STOREA(addr, data)    _mm_store_si128((__m128i*)addr, data)
# define _impl_ppz_I32_SETVAL(value)         _mm_set1_epi32(value)
# define _impl_ppz_I32_SETZERO()             _mm_setzero_si128()
# define _impl_ppz_I32_MUL(a, b)             _mm_mullo_epi32(a, b)
# define _impl_ppz_I32_ADD(a, b)             _mm_add_epi32(a, b)
# define _impl_ppz_I32_SUB(a, b)             _mm_sub_epi32(a, b)
# define _impl_ppz_I32_DIV(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define _impl_ppz_I32_SQRT(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
# define _impl_ppz_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_mul_epi32(b, c))
# define _impl_ppz_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_mul_epi32(b, c))
# if defined(_impl_ppz_SVML)
#   undef  _impl_ppz_I32_DIV
#   define _impl_ppz_I32_DIV(a, b)           _mm_div_epi32(a, b)
# endif

# define _impl_ppz_I32_EQ(a, b)              _mm_cmpeq_epi32(a, b)
# define _impl_ppz_I32_NEQ(a, b)             _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_GT(a, b)              _mm_cmpgt_epi32(a, b)
# define _impl_ppz_I32_GTE(a, b)             _mm_xor_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_LT(a, b)              _mm_cmplt_epi32(a, b)
# define _impl_ppz_I32_LTE(a, b)             _mm_xor_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, a))

# define _impl_ppz_I32_BW_NOT(a)             _mm_xor_si128(a, _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_AND(a, b)          _mm_and_si128(a, b)
# define _impl_ppz_I32_BW_NAND(a, b)         _mm_xor_si128(_mm_and_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_OR(a, b)           _mm_or_si128(a, b)
# define _impl_ppz_I32_BW_NOR(a, b)          _mm_xor_si128(_mm_or_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_XOR(a, b)          _mm_xor_si128(a, b)
# define _impl_ppz_I32_BW_XNOR(a, b)         _mm_xor_si128(_mm_xor_si128(a, b), _mm_cmpeq_epi32(a, a))

# define _impl_ppz_I32_ABS(a)                _mm_abs_epi32(a)
#elif defined(_impl_ppz_SSSE3)
# define _impl_ppz_I32
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(int32_t, __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(uint32_t, __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
# define _impl_ppz_I32_LOADU(data)           _mm_loadu_si128((const __m128i*)data)
# define _impl_ppz_I32_LOADA(data)           _mm_load_si128((const __m128i*)data)
# define _impl_ppz_I32_STOREU(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
# define _impl_ppz_I32_STOREA(addr, data)    _mm_store_si128((__m128i*)addr, data)
# define _impl_ppz_I32_SETVAL(value)         _mm_set1_epi32(value)
# define _impl_ppz_I32_SETZERO()             _mm_setzero_si128()
# define _impl_ppz_I32_MUL(a, b)             _mm_mullo_epi32(a, b)
# define _impl_ppz_I32_ADD(a, b)             _mm_add_epi32(a, b)
# define _impl_ppz_I32_SUB(a, b)             _mm_sub_epi32(a, b)
# define _impl_ppz_I32_DIV(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define _impl_ppz_I32_SQRT(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
# define _impl_ppz_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_mul_epi32(b, c))
# define _impl_ppz_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_mul_epi32(b, c))
# if defined(_impl_ppz_SVML)
#   undef  _impl_ppz_I32_DIV
#   define _impl_ppz_I32_DIV(a, b)           _mm_div_epi32(a, b)
# endif

# define _impl_ppz_I32_EQ(a, b)              _mm_cmpeq_epi32(a, b)
# define _impl_ppz_I32_NEQ(a, b)             _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_GT(a, b)              _mm_cmpgt_epi32(a, b)
# define _impl_ppz_I32_GTE(a, b)             _mm_xor_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_LT(a, b)              _mm_cmplt_epi32(a, b)
# define _impl_ppz_I32_LTE(a, b)             _mm_xor_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, a))

# define _impl_ppz_I32_BW_NOT(a)             _mm_xor_si128(a, _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_AND(a, b)          _mm_and_si128(a, b)
# define _impl_ppz_I32_BW_NAND(a, b)         _mm_xor_si128(_mm_and_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_OR(a, b)           _mm_or_si128(a, b)
# define _impl_ppz_I32_BW_NOR(a, b)          _mm_xor_si128(_mm_or_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_XOR(a, b)          _mm_xor_si128(a, b)
# define _impl_ppz_I32_BW_XNOR(a, b)         _mm_xor_si128(_mm_xor_si128(a, b), _mm_cmpeq_epi32(a, a))

# define _impl_ppz_I32_ABS(a)                _mm_abs_epi32(a)
#elif defined(_impl_ppz_SSE2)
# define _impl_ppz_I32
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(int32_t, __m128i, __m128i, 16, "SSE2, SSE");
  _impl_ppz_MAKE_SIMD_SPECIALIZATION(uint32_t, __m128i, __m128i, 16, "SSE2, SSE");
# define _impl_ppz_I32_LOADU(data)           _mm_loadu_si128((const __m128i*)data)
# define _impl_ppz_I32_LOADA(data)           _mm_load_si128((const __m128i*)data)
# define _impl_ppz_I32_STOREU(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
# define _impl_ppz_I32_STOREA(addr, data)    _mm_store_si128((__m128i*)addr, data)
# define _impl_ppz_I32_SETVAL(value)         _mm_set1_epi32(value)
# define _impl_ppz_I32_SETZERO()             _mm_setzero_si128()
# define _impl_ppz_I32_MUL(a, b)             _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define _impl_ppz_I32_ADD(a, b)             _mm_add_epi32(a, b)
# define _impl_ppz_I32_SUB(a, b)             _mm_sub_epi32(a, b)
# define _impl_ppz_I32_DIV(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define _impl_ppz_I32_SQRT(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
# define _impl_ppz_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
# define _impl_ppz_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))

# define _impl_ppz_I32_EQ(a, b)              _mm_cmpeq_epi32(a, b)
# define _impl_ppz_I32_NEQ(a, b)             _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_GT(a, b)              _mm_cmpgt_epi32(a, b)
# define _impl_ppz_I32_GTE(a, b)             _mm_xor_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_LT(a, b)              _mm_cmplt_epi32(a, b)
# define _impl_ppz_I32_LTE(a, b)             _mm_xor_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, a))

# define _impl_ppz_I32_BW_NOT(a)             _mm_xor_si128(a, _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_AND(a, b)          _mm_and_si128(a, b)
# define _impl_ppz_I32_BW_NAND(a, b)         _mm_xor_si128(_mm_and_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_OR(a, b)           _mm_or_si128(a, b)
# define _impl_ppz_I32_BW_NOR(a, b)          _mm_xor_si128(_mm_or_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _impl_ppz_I32_BW_XOR(a, b)          _mm_xor_si128(a, b)
# define _impl_ppz_I32_BW_XNOR(a, b)         _mm_xor_si128(_mm_xor_si128(a, b), _mm_cmpeq_epi32(a, a))

# define _impl_ppz_I32_ABS(a)                                   \
  [&]() -> __m128i {                                            \
    const __m128i signmask = _mm_srai_epi32(a, 31);             \
    return _mm_sub_epi32(_mm_xor_si128(a, signmask), signmask); \
  }()
#endif

#ifdef _impl_ppz_I32
  // load a vector with zeros
  template<>
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_setzero<int32_t>() noexcept
  {
    return _impl_ppz_I32_SETZERO();
  }
  template<>
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_setzero<uint32_t>() noexcept
  {
    return _impl_ppz_I32_SETZERO();
#   undef  _impl_ppz_I32_SETZERO
  }

  // load a vector from unaligned data
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_loadu(const int32_t data[]) noexcept
  {
    return _impl_ppz_I32_LOADU(data);
  }
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_loadu(const uint32_t data[]) noexcept
  {
    return _impl_ppz_I32_LOADU(data);
#   undef  _impl_ppz_I32_LOADU
  }

  // load a vector from aligned data
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_loada(const int32_t data[]) noexcept
  {
    return _impl_ppz_I32_LOADA(data);
  }
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_loada(const uint32_t data[]) noexcept
  {
    return _impl_ppz_I32_LOADA(data);
#   undef  _impl_ppz_I32_LOADA
  }

  // store a vector into unaligned memory
  _impl_ppz_INLINE 
  void simd_storeu(int32_t addr[], const SIMD<int32_t>::Type& data)
  {
    _impl_ppz_I32_STOREU(addr, data);
#   undef _impl_ppz_I32_STOREU
  }

  // store a vector into aligned memory
  _impl_ppz_INLINE 
  void simd_storea(int32_t addr[], const SIMD<int32_t>::Type& data)
  {
    _impl_ppz_I32_STOREA(addr, data);
  }
  _impl_ppz_INLINE 
  void simd_storea(uint32_t addr[], const SIMD<int32_t>::Type& data)
  {
    _impl_ppz_I32_STOREA(addr, data);
#   undef _impl_ppz_I32_STOREA
  }

  // load a vector with a specific value
  template<>
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_setval(const int32_t value) noexcept
  {
    return _impl_ppz_I32_SETVAL(value);
  }
  template<>
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_setval(const uint32_t value) noexcept
  {
    return _impl_ppz_I32_SETVAL(value);
#   undef  _impl_ppz_I32_SETVAL
  }

  // [a] + [b]
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_add(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _impl_ppz_I32_ADD(a, b);
#   undef  _impl_ppz_I32_ADD
  }

  // [a] * [b]
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_mul(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _impl_ppz_I32_MUL(a, b);
#   undef  _impl_ppz_I32_MUL
  }

  // [a] - [b]
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_sub(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _impl_ppz_I32_SUB(a, b);
#   undef  _impl_ppz_I32_SUB
  }

  // [a] / [b]
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_div(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _impl_ppz_I32_DIV(a, b);
#   undef  _impl_ppz_I32_DIV
  }

  // sqrt([a])
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_sqrt(const SIMD<int32_t>::Type& a) noexcept
  {
    return _impl_ppz_I32_SQRT(a);
#   undef  _impl_ppz_I32_SQRT
  }

  // [a] + ([b] * [c])
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_addmul(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b, const SIMD<int32_t>::Type& c) noexcept
  {
    return _impl_ppz_I32_ADDMUL(a, b, c);
#   undef  _impl_ppz_I32_ADDMUL
  }

  // [a] - ([b] * [c])
  _impl_ppz_INLINE 
  SIMD<int32_t>::Type simd_submul(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b, const SIMD<int32_t>::Type& c) noexcept
  {
    return _impl_ppz_I32_SUBMUL(a, b, c);
#   undef  _impl_ppz_I32_SUBMUL
  }

  // [a] == [b]
  _impl_ppz_INLINE 
  auto simd_eq(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return _impl_ppz_I32_EQ(a, b);
#   undef  _impl_ppz_I32_EQ
  }

  // [a] != [b]
  _impl_ppz_INLINE 
  auto simd_neq(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return _impl_ppz_I32_NEQ(a, b);
#   undef  _impl_ppz_I32_NEQ
  }

  // [a] > [b]
  _impl_ppz_INLINE 
  auto simd_gt(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return _impl_ppz_I32_GT(a, b);
#   undef  _impl_ppz_I32_GT
  }

  // [a] >= [b]
  _impl_ppz_INLINE 
  auto simd_gte(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return _impl_ppz_I32_GTE(a, b);
#   undef  _impl_ppz_I32_GTE
  }

  // [a] < [b]
  _impl_ppz_INLINE 
  auto simd_lt(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return _impl_ppz_I32_LT(a, b);
#   undef  _impl_ppz_I32_LT
  }

  // [a] <= [b]
  _impl_ppz_INLINE 
  auto simd_lte(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept -> SIMD<int32_t>::Mask
  {
    return _impl_ppz_I32_LTE(a, b);
#   undef  _impl_ppz_I32_LTE
  }

  inline namespace Bitwise
  {
    // ![a]
    _impl_ppz_INLINE 
    SIMD<int32_t>::Type simd_not(const SIMD<int32_t>::Type& a) noexcept
    {
      return _impl_ppz_I32_BW_NOT(a);
#     undef  _impl_ppz_I32_BW_NOT
    }

    // [a] & [b]
    _impl_ppz_INLINE 
    SIMD<int32_t>::Type simd_and(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return _impl_ppz_I32_BW_AND(a, b);
#     undef  _impl_ppz_I32_BW_AND
    }

    // !([a] & [b])
    _impl_ppz_INLINE 
    SIMD<int32_t>::Type simd_nand(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return _impl_ppz_I32_BW_NAND(a, b);
#     undef  _impl_ppz_I32_BW_NAND
    }

    // [a] | [b]
    _impl_ppz_INLINE 
    SIMD<int32_t>::Type simd_or(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return _impl_ppz_I32_BW_OR(a, b);
#     undef  _impl_ppz_I32_BW_OR
    }

    // !([a] | [b])
    _impl_ppz_INLINE 
    SIMD<int32_t>::Type simd_nor(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return _impl_ppz_I32_BW_NOR(a, b);
#     undef  _impl_ppz_I32_BW_NOR
    }

    // [a] ^ [b]
    _impl_ppz_INLINE 
    SIMD<int32_t>::Type simd_xor(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return _impl_ppz_I32_BW_XOR(a, b);
#     undef  _impl_ppz_I32_BW_XOR
    }

    // !([a] ^ [b])
    _impl_ppz_INLINE 
    SIMD<int32_t>::Type simd_xnor(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
    {
      return _impl_ppz_I32_BW_XNOR(a, b);
#     undef  _impl_ppz_I32_BW_XNOR
    }
  }

  // abs([a])
  _impl_ppz_INLINE
  SIMD<int32_t>::Type simd_abs(const SIMD<int32_t>::Type& a) noexcept
  {
    return _impl_ppz_I32_ABS(a);
#   undef  _impl_ppz_I32_ABS
  }

  inline _impl_ppz_CONSTEXPR_CPP14
  std::ostream& operator<<(std::ostream& ostream, const SIMD<int32_t>::Type& vector) noexcept
  {
    for (unsigned k = 0; k < SIMD<int32_t>::size; ++k)
    {
      if (k != 0) _impl_ppz_HOT
      {
        ostream << ' ';
      }

      ostream << reinterpret_cast<const int32_t*>(&vector)[k];
    }

    return ostream;
  }
#endif
}
#undef _impl_ppz_INLINE
#undef _impl_ppz_PRAGMA
#undef _impl_ppz_IGNORE
#undef _impl_ppz_COLD
#undef _impl_ppz_HOT
#undef _impl_ppz_THREADLOCAL
#undef _impl_ppz_DECLARE_MUTEX
#undef _impl_ppz_DECLARE_LOCK
#undef _impl_ppz_MAKE_SIMD_SPECIALIZATION
#else
#error "Parallilos: Support for ISO C++11 is required"
#endif
#endif