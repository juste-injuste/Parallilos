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
#include <cstddef>     // for size_t
#include <cstdint>     // for fixed-size integers
#include <cmath>       // for std::sqrt
#include <cstdlib>     // for std::malloc, std::free, std::aligned_alloc
#include <ostream>     // for std::ostream
#include <iostream>    // for std::clog
#include <type_traits> // for std::is_arithmetic, std::is_integral
//---conditionally necessary standard libraries-------------------------------------------------------------------------
#if defined(__STDCPP_THREADS__) and not defined(PPZ_NOT_THREADSAFE)
# define _ppz_impl_THREADSAFE
# include <mutex> // for std::mutex, std::lock_guard
#endif
#if defined(PPZ_DEBUGGING)
# include <type_traits> // for std::is_floating_point, std::is_unsigned, std::is_pointer, std::remove_pointer
# include <typeinfo>    // for typeid
# include <cstdio>      // for std::sprintf
#endif
//---SIMD intrinsic functions libraries---------------------------------------------------------------------------------
#if not defined(__OPTIMIZE__)
# define __OPTIMIZE__
# define _ppz_impl_OPTIMIZE
#endif
#if defined(__AVX512F__)
# define  _ppz_impl_AVX512F
# include <immintrin.h>
#endif
#if defined(__AVX2__)
# define  _ppz_impl_AVX2
# include <immintrin.h>
#endif
#if defined(__FMA__)
# define  _ppz_impl_FMA
# include <immintrin.h>
#endif
#if defined(__AVX__)
# define  _ppz_impl_AVX
# include <immintrin.h>
#endif
#if defined(__SSE4_2__)
# define  _ppz_impl_SSE4_2
# include <nmmintrin.h>
#endif
#if defined(__SSE4_1__)
# define  _ppz_impl_SSE4_1
# include <smmintrin.h>
#endif
#if defined(__SSSE3__)
# define  _ppz_impl_SSSE3
# include <tmmintrin.h>
#endif
#if defined(__SSE3__)
# define  _ppz_impl_SSE3
# include <pmmintrin.h>
#endif
#if defined(__SSE2__)
# define  _ppz_impl_SSE2
# include <emmintrin.h>
#endif
#if defined(__SSE__)
# define  _ppz_impl_SSE
# include <xmmintrin.h>
#endif
// #if defined(__ARM_NEON) or defined(__ARM_NEON__)
// # if defined(__ARM_ARCH_64)
// #   define _ppz_impl_NEON64
// #   include <arm64_neon.h>
// # else
// #   define _ppz_impl_NEON
// #   include <arm_neon.h>
// # endif
// #endif
#if defined(_ppz_impl_OPTIMIZE)
# undef __OPTIMIZE__
# undef _ppz_impl_OPTIMIZE
#endif
//---Parallilos library-------------------------------------------------------------------------------------------------
namespace ppz
{
  template<typename type>
  struct SIMD;

  template<typename type>
  class Array;

  // for exposition purposes only, this is not the implementaiton.
  struct type_t {}; struct simd_t {}; struct mask_t {};

  // loading/storing functions
  simd_t simd_loadu (const type_t* addr);        // load  unaligned
  simd_t simd_loada (const type_t* addr);        // load  aligned
  void   simd_storeu(type_t* addr, simd_t data); // store unaligned
  void   simd_storea(type_t* addr, simd_t data); // store aligned

  // vector creation functions
  simd_t simd_setzero(            ); // [result] = 0
  simd_t simd_setval (type_t value); // [result] = value

  // vector arithmetic functions
  simd_t simd_add    (simd_t a, simd_t b          ); // [result] = [a] + [b]
  simd_t simd_mul    (simd_t a, simd_t b          ); // [result] = [a] * [b]
  simd_t simd_sub    (simd_t a, simd_t b          ); // [result] = [a] - [b]
  simd_t simd_div    (simd_t a, simd_t b          ); // [result] = [a] / [b]
  simd_t simd_addmul (simd_t a, simd_t b, simd_t c); // [result] = [a] + ([b] * [c])
  simd_t simd_submul (simd_t a, simd_t b, simd_t c); // [result] = [a] - ([b] * [c])

  // vector math functions
  simd_t simd_sqrt(simd_t a); // [result] = sqrt([a])
  simd_t simd_abs (simd_t a); // [result] = abs([a])

  // vector comparison functions
  mask_t simd_eq (simd_t a, simd_t b); // [result] = [a] == [b]
  mask_t simd_neq(simd_t a, simd_t b); // [result] = [a] != [b]
  mask_t simd_gt (simd_t a, simd_t b); // [result] = [a] >  [b]
  mask_t simd_gte(simd_t a, simd_t b); // [result] = [a] >= [b]
  mask_t simd_lt (simd_t a, simd_t b); // [result] = [a] <  [b]
  mask_t simd_lte(simd_t a, simd_t b); // [result] = [a] <= [b]

  // integral vector bitwise operation functions
  simd_t simd_compl(simd_t a          ); // [result] =  ~[a]
  simd_t simd_and  (simd_t a, simd_t b); // [result] =   [a] & [b]
  simd_t simd_nand (simd_t a, simd_t b); // [result] = ~([a] & [b])
  simd_t simd_or   (simd_t a, simd_t b); // [result] =   [a] | [b]
  simd_t simd_nor  (simd_t a, simd_t b); // [result] = ~([a] | [b])
  simd_t simd_xor  (simd_t a, simd_t b); // [result] =   [a] ^ [b]
  simd_t simd_xnor (simd_t a, simd_t b); // [result] = ~([a] ^ [b])

  namespace _io
  {
    static std::ostream dbg(std::clog.rdbuf()); // debugging
  }

  namespace _version
  {
    constexpr unsigned long MAJOR  = 000;
    constexpr unsigned long MINOR  = 001;
    constexpr unsigned long PATCH  = 000;
    constexpr unsigned long NUMBER = (MAJOR * 1000 + MINOR) * 1000 + PATCH;
  }
// ---------------------------------------------------------------------------------------------------------------------
  namespace _impl
  {
# if defined(__clang__)
#   define _ppz_impl_INLINE   __attribute__((always_inline, artificial)) inline
#   define _ppz_impl_RESTRICT __restrict__
# elif defined(__GNUC__)
#   define _ppz_impl_INLINE __attribute__((always_inline, artificial)) inline
#   define _ppz_impl_RESTRICT __restrict__
// # elif defined(__MINGW32__) or defined(__MINGW64__)
// #   define _ppz_impl_INLINE __attribute__((always_inline)) inline
// #   define _ppz_impl_RESTRICT
// # elif defined(__apple_build_version__)
// #   define _ppz_impl_INLINE __attribute__((always_inline)) inline
// #   define _ppz_impl_RESTRICT
// # elif defined(_MSC_VER)
// #   define _ppz_impl_INLINE __forceinline
// #   define _ppz_impl_RESTRICT
// # elif defined(__INTEL_COMPILER)
// #   define _ppz_impl_SVML
// #   define _ppz_impl_INLINE __forceinline
// #   define _ppz_impl_RESTRICT
// # elif defined(__ARMCC_VERSION)
// #   define _ppz_impl_INLINE __forceinline
// #   define _ppz_impl_RESTRICT
# else
#   define _ppz_impl_INLINE   inline
#   define _ppz_impl_RESTRICT
# endif

#   define _ppz_impl_PRAGMA(PRAGMA) _Pragma(#PRAGMA)
#   define _ppz_impl_CLANG_IGNORE(WARNING, ...)          \
      _ppz_impl_PRAGMA(clang diagnostic push)            \
      _ppz_impl_PRAGMA(clang diagnostic ignored WARNING) \
      __VA_ARGS__                                        \
      _ppz_impl_PRAGMA(clang diagnostic pop)

// support from clang 12.0.0 and GCC 10.1 onward
# if defined(__clang__) and (__clang_major__ >= 12)
# if __cplusplus < 202002L
#   define _ppz_impl_LIKELY   _ppz_impl_CLANG_IGNORE("-Wc++20-extensions", [[likely]])
#   define _ppz_impl_UNLIKELY _ppz_impl_CLANG_IGNORE("-Wc++20-extensions", [[unlikely]])
# else
#   define _ppz_impl_LIKELY   [[likely]]
#   define _ppz_impl_UNLIKELY [[unlikely]]
# endif
# elif defined(__GNUC__) and (__GNUC__ >= 10)
#   define _ppz_impl_LIKELY   [[likely]]
#   define _ppz_impl_UNLIKELY [[unlikely]]
# else
#   define _ppz_impl_LIKELY
#   define _ppz_impl_UNLIKELY
# endif

// support from clang 3.9.0 and GCC 4.7.3 onward
# if defined(__clang__)
#   define _ppz_impl_EXPECTED(CONDITION) (__builtin_expect(static_cast<bool>(CONDITION), 1)) _ppz_impl_LIKELY
#   define _ppz_impl_ABNORMAL(CONDITION) (__builtin_expect(static_cast<bool>(CONDITION), 0)) _ppz_impl_UNLIKELY
# elif defined(__GNUC__)
#   define _ppz_impl_EXPECTED(CONDITION) (__builtin_expect(static_cast<bool>(CONDITION), 1)) _ppz_impl_LIKELY
#   define _ppz_impl_ABNORMAL(CONDITION) (__builtin_expect(static_cast<bool>(CONDITION), 0)) _ppz_impl_UNLIKELY
# else
#   define _ppz_impl_EXPECTED(CONDITION) (CONDITION) _ppz_impl_LIKELY
#   define _ppz_impl_ABNORMAL(CONDITION) (CONDITION) _ppz_impl_UNLIKELY
# endif

# if __cplusplus >= 201402L
#   define _ppz_impl_CONSTEXPR_CPP14 constexpr
# else
#   define _ppz_impl_CONSTEXPR_CPP14
# endif

# if defined(_ppz_impl_THREADSAFE)
#   undef  _ppz_impl_THREADSAFE
#   define _ppz_impl_THREADLOCAL         thread_local
#   define _ppz_impl_DECLARE_MUTEX(...)  static std::mutex __VA_ARGS__
#   define _ppz_impl_DECLARE_LOCK(MUTEX) std::lock_guard<std::mutex> _lock{MUTEX}
# else
#   define _ppz_impl_THREADLOCAL
#   define _ppz_impl_DECLARE_MUTEX(...)
#   define _ppz_impl_DECLARE_LOCK(MUTEX)
# endif

    template<size_t size>
    class _par_iterator
    {
    public:
      constexpr _par_iterator() noexcept = default;
      constexpr _par_iterator(const size_t n_elements) noexcept :
        _passes_left(size ? (n_elements / size) : 0)
      {}

      _ppz_impl_CONSTEXPR_CPP14
      _par_iterator& begin() noexcept
      {
        return *this;
      }

      constexpr _par_iterator end() noexcept
      {
        return _par_iterator(0);
      }

      constexpr
      size_t operator*() noexcept
      {
        return _current;
      }

      _ppz_impl_CONSTEXPR_CPP14 void operator++() noexcept
      {
        --_passes_left, _current += size;
      }

      constexpr
      bool operator!=(const _par_iterator&) noexcept
      {
        return _passes_left;
      }

    private:
      size_t _current     = 0;
      size_t _passes_left = 0;
    public:
      const size_t passes = _passes_left;
    };

    template<size_t size>
    class _seq_iterator
    {
    public:
      constexpr _seq_iterator() noexcept = default;
      constexpr _seq_iterator(const size_t n_elements) noexcept :
        _current(size ? ((n_elements / size) * size) : 0),
        _passes_left(n_elements - _current)
      {}

      _ppz_impl_CONSTEXPR_CPP14
      _seq_iterator& begin() noexcept
      {
        return *this;
      }
      
      constexpr
      _seq_iterator end() const noexcept
      {
        return _seq_iterator();
      }

      constexpr
      size_t operator*() const noexcept
      {
        return _current;
      }

      _ppz_impl_CONSTEXPR_CPP14
      void operator++() noexcept
      {
        --_passes_left, ++_current;
      }

      constexpr
      bool operator!=(const _seq_iterator&) const noexcept
      {
        return _passes_left;
      }

    private:
      size_t _current     = 0;
      size_t _passes_left = 0;
    public:
      const size_t passes = _passes_left;
    };

# if defined(PPZ_DEBUGGING)
    static _ppz_impl_THREADLOCAL char _typename_buffer[32];

    template<typename type>
    _ppz_impl_CONSTEXPR_CPP14
    auto _type_name() -> const char*
    {
      if (std::is_floating_point<type>::value)
      {
        std::sprintf(_typename_buffer, "float%zu", sizeof(type) * 8);
      }
      else if (std::is_unsigned<type>::value)
      {
        std::sprintf(_typename_buffer, "uint%zu", sizeof(type) * 8);
      }
      else if (std::is_pointer<type>::value)
      {
        _type_name<typename std::remove_pointer<type>::type>();
        char* char_ptr = _typename_buffer;
        while (*char_ptr != '\0') { ++char_ptr; }
        char_ptr[0] = '*';
        char_ptr[1] = '\0';
      }
      else
      {
        std::sprintf(_typename_buffer, "int%zu", sizeof(type) * 8);
      }

      return _typename_buffer;
    }

    static _ppz_impl_THREADLOCAL char _dbg_buf[256];
    _ppz_impl_DECLARE_MUTEX(_dbg_mtx);

#   define _ppz_impl_DEBUG_MESSAGE(...)                              \
      [&](const char* const caller_){                                \
        std::sprintf(_impl::_dbg_buf, __VA_ARGS__);                  \
        _ppz_impl_DECLARE_LOCK(_impl::_dbg_mtx);                     \
        _io::dbg << caller_ << ": " << _impl::_dbg_buf << std::endl; \
      }(__func__)
# else
#   define _ppz_impl_DEBUG_MESSAGE(...) void(0)
# endif
  }
// ---------------------------------------------------------------------------------------------------------------------
  template<typename type>
  struct SIMD
  {
    SIMD() = delete;
    static_assert(std::is_arithmetic<type>::value, "SIMD: 'type' must be arithmetic.");

    static constexpr size_t size      = 0; // amount of elements in SIMD vector
    static constexpr size_t alignment = 0; // memory alignment required for SIMD vector

    using Type = type; // SIMD vector type
    using Mask = bool; // SIMD mask type

    static constexpr const char* set = "no SIMD instruction set used for this type.";

    static constexpr
    auto parallel(size_t n_elements) noexcept -> _impl::_par_iterator<size>;

    static constexpr
    auto sequential(size_t n_elements) noexcept -> _impl::_seq_iterator<size>;
  };
//----------------------------------------------------------------------------------------------------------------------
  template<typename type>
  class Array final
  {
  static_assert(std::is_arithmetic<type>::value, "Array: 'type' must be arithmetic.");
  public:
    inline constexpr
    auto data(size_t k = 0) const noexcept -> const type*;

    inline _ppz_impl_CONSTEXPR_CPP14
    auto data(size_t k = 0) noexcept -> type*;

    inline constexpr
    auto size() const noexcept -> size_t;
    
    inline constexpr 
    auto operator[](size_t k) const noexcept -> type;

    inline _ppz_impl_CONSTEXPR_CPP14
    auto operator[](size_t k) noexcept -> type&;

    inline constexpr
    operator type*() noexcept;

    inline constexpr
    auto release() noexcept -> type*;

    inline // interpret aligned array as a vector, k is an offset in elements into the array
    auto as_vector(size_t k) noexcept -> typename SIMD<type>::Type&;

    inline _ppz_impl_CONSTEXPR_CPP14
    Array(size_t number_of_elements) noexcept;

    inline _ppz_impl_CONSTEXPR_CPP14
    Array(std::initializer_list<type> initializer_list) noexcept;

    inline _ppz_impl_CONSTEXPR_CPP14
    Array(Array&& other_) noexcept;

    inline ~Array();
    Array(const Array&) = delete; // copying is disallowed

  private:
    type*  _array;
    size_t _numel;
  };
// ---------------------------------------------------------------------------------------------------------------------
  template<typename type>
  constexpr
  auto SIMD<type>::parallel(const size_t n_elements_) noexcept -> _impl::_par_iterator<size>
  {
    return _impl::_par_iterator<size>(n_elements_);
  }

  template<typename type>
  constexpr
  auto SIMD<type>::sequential(const size_t n_elements_) noexcept -> _impl::_seq_iterator<size>
  {
    return _impl::_seq_iterator<size>(n_elements_);
  }
//----------------------------------------------------------------------------------------------------------------------
  template<typename type>
  constexpr
  auto Array<type>::data(const size_t k) const noexcept -> const type*
  {
    return _array + k;
  }

  template<typename type>
  _ppz_impl_CONSTEXPR_CPP14
  auto Array<type>::data(const size_t k) noexcept -> type*
  {
    return _array + k;
  }

  template<typename type>
  constexpr
  auto Array<type>::size() const noexcept -> size_t
  {
    return _numel;
  }

  template<typename type>
  constexpr
  auto Array<type>::operator[](const size_t k) const noexcept -> type
  {
    return _array[k];
  }

  template<typename type>
  _ppz_impl_CONSTEXPR_CPP14
  auto Array<type>::operator[](const size_t k) noexcept -> type&
  {
    return _array[k];
  }

  template<typename type>
  constexpr
  Array<type>::operator type*() noexcept
  {
    return _array;
  }

  template<typename type>
  constexpr
  auto Array<type>::release() noexcept -> type*
  {
    type* data = _array;
    _array  = nullptr;
    _numel  = 0;
    return data;
  }

  template<typename type>
  _ppz_impl_CONSTEXPR_CPP14
  Array<type>::Array(const size_t number_of_elements_) noexcept :
    _array([number_of_elements_]() -> type* {
      if _ppz_impl_ABNORMAL((number_of_elements_ == 0))
      {
        return nullptr;
      }

      if _ppz_impl_ABNORMAL(SIMD<type>::alignment == 0)
      {
        return reinterpret_cast<type*>(std::malloc(number_of_elements_ * sizeof(type)));
      }

      void* memory_block = std::malloc(number_of_elements_ * sizeof(type) + SIMD<type>::alignment);

      if _ppz_impl_ABNORMAL(memory_block == nullptr)
      {
        return nullptr;
      }

      // align on alignement boundary
      auto  aligned_address      = (reinterpret_cast<uintptr_t>(memory_block) + SIMD<type>::alignment) & ~(SIMD<type>::alignment - 1);
      void* aligned_memory_block = reinterpret_cast<void*>(aligned_address);

      // bookkeeping of original memory block
      reinterpret_cast<void**>(aligned_memory_block)[-1] = memory_block;

      return reinterpret_cast<type*>(aligned_memory_block);
    }()),
    _numel(_array == nullptr ? 0 : number_of_elements_)
  {
    _ppz_impl_DEBUG_MESSAGE("created array of size %zu", _numel);
  }

  template<typename type>
  _ppz_impl_CONSTEXPR_CPP14
  Array<type>::Array(const std::initializer_list<type> initializer_list_) noexcept :
    Array(initializer_list_.size())
  {
    if _ppz_impl_EXPECTED(_numel != 0)
    {
      size_t k = 0;
      for (type value : initializer_list_)
      {
        _array[k++] = value;
      }
    }
  }

  template<typename type>
  _ppz_impl_CONSTEXPR_CPP14
  Array<type>::Array(Array&& other_) noexcept :
    _array(other_._array), _numel(other_._numel)
  {
    _ppz_impl_DEBUG_MESSAGE("moved array of size %zu.", _numel);
    other_._array = nullptr;
    other_._numel = 0;
  }

  template<typename type>
  Array<type>::~Array()
  {
    if _ppz_impl_EXPECTED(_array != nullptr)
    {
      if _ppz_impl_EXPECTED(SIMD<type>::alignment != 0)
      {
        std::free(reinterpret_cast<void**>(_array)[-1]);
      }
      else std::free(_array);
    }

    _ppz_impl_DEBUG_MESSAGE("freed used memory.");
  }
    
  template<typename type>
  auto Array<type>::as_vector(const size_t k) noexcept -> typename SIMD<type>::Type&
  {
    return reinterpret_cast<typename SIMD<type>::Type&>(_array[k]);
  }
// ---------------------------------------------------------------------------------------------------------------------
  template<typename type>
  constexpr
  typename SIMD<type>::Type simd_loadu(const type* const _ppz_impl_RESTRICT data) noexcept
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type*>());
    return *data;
  }

  template<typename type>
  constexpr
  typename SIMD<type>::Type simd_loada(const type* const _ppz_impl_RESTRICT data) noexcept
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type*>());
    return *data;
  }

  template<typename type>
  constexpr
  void simd_storeu(type* const _ppz_impl_RESTRICT addr, const typename SIMD<type>::Type& data)
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type*>());
    *addr = data;
  }

  template<typename type>
  constexpr
  void simd_storea(type* const _ppz_impl_RESTRICT addr, const typename SIMD<type>::Type& data)
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type*>());
    *addr = data;
  }

  template<typename type>
  constexpr
  typename SIMD<type>::Type simd_setzero() noexcept
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return 0;
  }

  template<typename type>
  constexpr
  auto simd_setval(const type value) noexcept -> typename SIMD<type>::Type
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return value;
  }

  template<typename type>
  constexpr
  type simd_add(const type a, const type b)
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a + b;
  }

  template<typename type>
  constexpr
  type simd_mul(const type a, const type b)
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a * b;
  }

  template<typename type>
  constexpr
  type simd_sub(const type a, const type b)
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a - b;
  }

  template<typename type>
  constexpr
  type simd_div(const type a, const type b)
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a / b;
  }

  template<typename type>
  constexpr
  type simd_addmul(const type a, const type b, const type c)
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a + b * c;
  }

  template<typename type>
  constexpr
  type simd_submul(const type a, const type b, const type c)
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a - b * c;
  }

  template<typename type>
  constexpr
  type simd_sqrt(const type a)
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return std::sqrt(a);
  }

  template<typename type>
  constexpr
  type simd_abs(const type a) noexcept
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return std::abs(a);
  }

  template<typename type>
  constexpr
  bool simd_eq(const type a, const type b) noexcept
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a == b;
  }

  template<typename type>
  constexpr
  bool simd_neq(const type a, const type b) noexcept
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a != b;
  }

  template<typename type>
  constexpr
  bool simd_gt(const type a, const type b) noexcept
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a > b;
  }

  template<typename type>
  constexpr
  bool simd_gte(const type a, const type b) noexcept
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a >= b;
  }

  template<typename type>
  constexpr
  bool simd_lt(const type a, const type b) noexcept
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a < b;
  }

  template<typename type>
  constexpr
  bool simd_lte(const type a, const type b) noexcept
  {
    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a <= b;
  }

  template<typename type>
  constexpr
  type simd_compl(const type a) noexcept
  {
    static_assert(std::is_integral<type>::value, "simd_compl: 'type' must be integral.");

    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return ~a;
  }

  template<typename type>
  constexpr
  type simd_and(const type a, const type b) noexcept
  {
    static_assert(std::is_integral<type>::value, "simd_and: 'type' must be integral.");

    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a & b;
  }

  template<typename type>
  constexpr
  type simd_nand(const type a, const type b) noexcept
  {
    static_assert(std::is_integral<type>::value, "simd_nand: 'type' must be integral.");

    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return ~(a & b);
  }

  template<typename type>
  constexpr
  type simd_or(const type a, const type b) noexcept
  {
    static_assert(std::is_integral<type>::value, "simd_or: 'type' must be integral.");

    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a | b;
  }

  template<typename type>
  constexpr
  type simd_nor(const type a, const type b) noexcept
  {
    static_assert(std::is_integral<type>::value, "simd_nor: 'type' must be integral.");

    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return ~(a | b);
  }

  template<typename type>
  constexpr
  type simd_xor(const type a, const type b) noexcept
  {
    static_assert(std::is_integral<type>::value, "simd_xor: 'type' must be integral.");

    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return a ^ b;
  }

  template<typename type>
  constexpr
  type simd_xnor(const type a, const type b) noexcept
  {
    static_assert(std::is_integral<type>::value, "simd_xnor: 'type' must be integral.");

    _ppz_impl_DEBUG_MESSAGE("\"%s\" is not SIMD-supported", _impl::_type_name<type>());
    return ~(a ^ b);
  }
//----------------------------------------------------------------------------------------------------------------------
  namespace _impl
  {
#   define _ppz_impl_MAKE_SIMD_SPECIALIZATION(TYPE, VECTOR_TYPE, MASK_TYPE, ALIGNMENT, SET) \
      template<>                                                                            \
      struct SIMD<TYPE>                                                                     \
      {                                                                                     \
        SIMD() = delete;                                                                    \
        static_assert(std::is_arithmetic<TYPE>::value, "SIMD: 'TYPE' must be arithmetic."); \
        static constexpr size_t size      = sizeof(VECTOR_TYPE) / sizeof(TYPE);             \
        static constexpr size_t alignment = ALIGNMENT;                                      \
        using Type = VECTOR_TYPE;                                                           \
        using Mask = MASK_TYPE;                                                             \
        static constexpr const char* set = SET;                                             \
        static constexpr                                                                    \
        auto parallel(const size_t n_elements_) noexcept -> _impl::_par_iterator<size>      \
        {                                                                                   \
          return _impl::_par_iterator<size>(n_elements_);                                   \
        }                                                                                   \
        static constexpr                                                                    \
        auto sequential(const size_t n_elements_) noexcept -> _impl::_seq_iterator<size>    \
        {                                                                                   \
          return _impl::_seq_iterator<size>(n_elements_);                                   \
        }                                                                                   \
      };
  }

#if defined(_ppz_impl_AVX512F)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define _ppz_impl_F32
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(float, __m512, __mmask16, 64, "AVX512F");
# define _ppz_impl_F32_LOADU(data)           _mm512_loadu_ps(data)
# define _ppz_impl_F32_LOADA(data)           _mm512_load_ps(data)
# define _ppz_impl_F32_STOREU(addr, data)    _mm512_storeu_ps(reinterpret_cast<void*>(addr), data)
# define _ppz_impl_F32_STOREA(addr, data)    _mm512_store_ps(reinterpret_cast<void*>(addr), data)
# define _ppz_impl_F32_SETVAL(value)         _mm512_set1_ps(value)
# define _ppz_impl_F32_SETZERO()             _mm512_setzero_ps()
# define _ppz_impl_F32_MUL(a, b)             _mm512_mul_ps(a, b)
# define _ppz_impl_F32_ADD(a, b)             _mm512_add_ps(a, b)
# define _ppz_impl_F32_SUB(a, b)             _mm512_sub_ps(a, b)
# define _ppz_impl_F32_DIV(a, b)             _mm512_div_ps(a, b)
# define _ppz_impl_F32_ADDMUL(a, b, c)       _mm512_fmadd_ps(b, c, a)
# define _ppz_impl_F32_SUBMUL(a, b, c)       _mm512_fnmadd_ps(a, b, c)
# define _ppz_impl_F32_SQRT(a)               _mm512_sqrt_ps(a)
# define _ppz_impl_F32_ABS(a)                _mm512_abs_ps(a)
# define _ppz_impl_F32_EQ(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_EQ_UQ)
# define _ppz_impl_F32_NEQ(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_NEQ_UQ)
# define _ppz_impl_F32_GT(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ)
# define _ppz_impl_F32_GTE(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ)
# define _ppz_impl_F32_LT(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ)
# define _ppz_impl_F32_LTE(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ)

#elif defined(_ppz_impl_FMA)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define _ppz_impl_F32
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(float, __m256, __m256, 32, "AVX, FMA");
# define _ppz_impl_F32_LOADU(data)           _mm256_loadu_ps(data)
# define _ppz_impl_F32_LOADA(data)           _mm256_load_ps(data)
# define _ppz_impl_F32_STOREU(addr, data)    _mm256_storeu_ps(addr, data)
# define _ppz_impl_F32_STOREA(addr, data)    _mm256_store_ps(addr, data)
# define _ppz_impl_F32_SETVAL(value)         _mm256_set1_ps(value)
# define _ppz_impl_F32_SETZERO()             _mm256_setzero_ps()
# define _ppz_impl_F32_MUL(a, b)             _mm256_mul_ps(a, b)
# define _ppz_impl_F32_ADD(a, b)             _mm256_add_ps(a, b)
# define _ppz_impl_F32_SUB(a, b)             _mm256_sub_ps(a, b)
# define _ppz_impl_F32_DIV(a, b)             _mm256_div_ps(a, b)
# define _ppz_impl_F32_ADDMUL(a, b, c)       _mm256_fmadd_ps(b, c, a)
# define _ppz_impl_F32_SUBMUL(a, b, c)       _mm256_fnmadd_ps(a, b, c)
# define _ppz_impl_F32_SQRT(a)               _mm256_sqrt_ps(a)
# define _ppz_impl_F32_ABS(a)                _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a)
# define _ppz_impl_F32_EQ(a, b)              _mm256_cmp_ps(a, b, _CMP_EQ_UQ)
# define _ppz_impl_F32_NEQ(a, b)             _mm256_cmp_ps(a, b, _CMP_NEQ_UQ)
# define _ppz_impl_F32_GT(a, b)              _mm256_cmp_ps(a, b, _CMP_GT_OQ)
# define _ppz_impl_F32_GTE(a, b)             _mm256_cmp_ps(a, b, _CMP_GE_OQ)
# define _ppz_impl_F32_LT(a, b)              _mm256_cmp_ps(a, b, _CMP_LT_OQ)
# define _ppz_impl_F32_LTE(a, b)             _mm256_cmp_ps(a, b, _CMP_LE_OQ)

#elif defined(_ppz_impl_AVX)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define _ppz_impl_F32
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(float, __m256, __m256, 32, "AVX");
# define _ppz_impl_F32_LOADU(data)           _mm256_loadu_ps(data)
# define _ppz_impl_F32_LOADA(data)           _mm256_load_ps(data)
# define _ppz_impl_F32_STOREU(addr, data)    _mm256_storeu_ps(addr, data)
# define _ppz_impl_F32_STOREA(addr, data)    _mm256_store_ps(addr, data)
# define _ppz_impl_F32_SETVAL(value)         _mm256_set1_ps(value)
# define _ppz_impl_F32_SETZERO()             _mm256_setzero_ps()
# define _ppz_impl_F32_MUL(a, b)             _mm256_mul_ps(a, b)
# define _ppz_impl_F32_ADD(a, b)             _mm256_add_ps(a, b)
# define _ppz_impl_F32_SUB(a, b)             _mm256_sub_ps(a, b)
# define _ppz_impl_F32_DIV(a, b)             _mm256_div_ps(a, b)
# define _ppz_impl_F32_ADDMUL(a, b, c)       _mm256_add_ps(a, _mm256_mul_ps(b, c))
# define _ppz_impl_F32_SUBMUL(a, b, c)       _mm256_sub_ps(a, _mm256_mul_ps(b, c))
# define _ppz_impl_F32_SQRT(a)               _mm256_sqrt_ps(a)
# define _ppz_impl_F32_ABS(a)                _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a)
# define _ppz_impl_F32_EQ(a, b)              _mm256_cmp_ps(a, b, _CMP_EQ_UQ)
# define _ppz_impl_F32_NEQ(a, b)             _mm256_cmp_ps(a, b, _CMP_NEQ_UQ)
# define _ppz_impl_F32_GT(a, b)              _mm256_cmp_ps(a, b, _CMP_GT_OQ)
# define _ppz_impl_F32_GTE(a, b)             _mm256_cmp_ps(a, b, _CMP_GE_OQ)
# define _ppz_impl_F32_LT(a, b)              _mm256_cmp_ps(a, b, _CMP_LT_OQ)
# define _ppz_impl_F32_LTE(a, b)             _mm256_cmp_ps(a, b, _CMP_LE_OQ)

#elif defined(_ppz_impl_SSE)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define _ppz_impl_F32
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(float, __m128, __m128, 16, "SSE");
# define _ppz_impl_F32_LOADU(data)           _mm_loadu_ps(data)
# define _ppz_impl_F32_LOADA(data)           _mm_load_ps(data)
# define _ppz_impl_F32_STOREU(addr, data)    _mm_storeu_ps(addr, data)
# define _ppz_impl_F32_STOREA(addr, data)    _mm_store_ps(addr, data)
# define _ppz_impl_F32_SETVAL(value)         _mm_set1_ps(value)
# define _ppz_impl_F32_SETZERO()             _mm_setzero_ps()
# define _ppz_impl_F32_MUL(a, b)             _mm_mul_ps(a, b)
# define _ppz_impl_F32_ADD(a, b)             _mm_add_ps(a, b)
# define _ppz_impl_F32_SUB(a, b)             _mm_sub_ps(a, b)
# define _ppz_impl_F32_DIV(a, b)             _mm_div_ps(a, b)
# define _ppz_impl_F32_ADDMUL(a, b, c)       _mm_add_ps(a, _mm_mul_ps(b, c))
# define _ppz_impl_F32_SUBMUL(a, b, c)       _mm_sub_ps(a, _mm_mul_ps(b, c))
# define _ppz_impl_F32_SQRT(a)               _mm_sqrt_ps(a)
# define _ppz_impl_F32_ABS(a)                _mm_andnot_ps(_mm_set1_ps(-0.0f), a)
# define _ppz_impl_F32_EQ(a, b)              _mm_cmpeq_ps (a, b)
# define _ppz_impl_F32_NEQ(a, b)             _mm_cmpneq_ps (a, b)
# define _ppz_impl_F32_GT(a, b)              _mm_cmpgt_ps(a, b)
# define _ppz_impl_F32_GTE(a, b)             _mm_cmpge_ps(a, b)
# define _ppz_impl_F32_LT(a, b)              _mm_cmplt_ps(a, b)
# define _ppz_impl_F32_LTE(a, b)             _mm_cmple_ps(a, b)

// #elif defined(_ppz_impl_NEON) or defined(_ppz_impl_NEON64)
//   static_assert(sizeof(float) == 4, "float must be 32 bit");
// # define _ppz_impl_F32
//   _ppz_impl_MAKE_SIMD_SPECIALIZATION(float, float32x4_t, uint32x4_t, 0, "NEON");
// # define _ppz_impl_F32_LOADU(data)           vld1q_f32(data)
// # define _ppz_impl_F32_LOADA(data)           vld1q_f32(data)
// # define _ppz_impl_F32_STOREU(addr, data)    vst1q_f32(addr, data)
// # define _ppz_impl_F32_STOREA(addr, data)    vst1q_f32(addr, data)
// # define _ppz_impl_F32_SETVAL(value)         vdupq_n_f32(value)
// # define _ppz_impl_F32_SETZERO()             vdupq_n_f32(0.0f)
// # define _ppz_impl_F32_MUL(a, b)             vmulq_f32(a, b)
// # define _ppz_impl_F32_ADD(a, b)             vaddq_f32(a, b)
// # define _ppz_impl_F32_SUB(a, b)             vsubq_f32(a, b)
// # define _ppz_impl_F32_DIV(a, b)             vdivq_f32(a, b)
// # define _ppz_impl_F32_ADDMUL(a, b, c)       vmlaq_f32(a, b, c)
// # define _ppz_impl_F32_SUBMUL(a, b, c)       vmlsq_f32(a, b, c)
// # define _ppz_impl_F32_SQRT(a)               vsqrtq_f32(a)
// # define _ppz_impl_F32_ABS(a)                vabsq_f32(a)
// # define _ppz_impl_F32_EQ(a, b)              vceqq_f32(a, b)
// # define _ppz_impl_F32_NEQ(a, b)             vmvnq_u32(Tceqq_f32(a, b))
// # define _ppz_impl_F32_GT(a, b)              vcgtq_f32(a, b)
// # define _ppz_impl_F32_GTE(a, b)             vcgeq_f32(a, b)
// # define _ppz_impl_F32_LT(a, b)              vcltq_f32(a, b)
// # define _ppz_impl_F32_LTE(a, b)             vcleq_f32(a, b)

#endif

#ifdef _ppz_impl_F32
  template<>
  _ppz_impl_INLINE
  SIMD<float>::Type simd_setzero<float>() noexcept
  {
    return _ppz_impl_F32_SETZERO();
#   undef  _ppz_impl_F32_SETZERO
  }

  _ppz_impl_INLINE
  SIMD<float>::Type simd_loadu(const float* const _ppz_impl_RESTRICT data) noexcept
  {
    return _ppz_impl_F32_LOADU(data);
#   undef  _ppz_impl_F32_LOADU
  }

  _ppz_impl_INLINE
  SIMD<float>::Type simd_loada(const float* const _ppz_impl_RESTRICT data) noexcept
  {
    return _ppz_impl_F32_LOADA(data);
#   undef  _ppz_impl_F32_LOADA
  }

  _ppz_impl_INLINE
  void simd_storeu(float* const _ppz_impl_RESTRICT addr, const SIMD<float>::Type& data)
  {
    _ppz_impl_F32_STOREU(addr, data);
#   undef _ppz_impl_F32_STOREU
  }

  _ppz_impl_INLINE
  void simd_storea(float* const _ppz_impl_RESTRICT addr, const SIMD<float>::Type& data)
  {
    _ppz_impl_F32_STOREA(addr, data);
#   undef _ppz_impl_F32_STOREA
  }

  template<>
  _ppz_impl_INLINE
  SIMD<float>::Type simd_setval(const float value) noexcept
  {
    return _ppz_impl_F32_SETVAL(value);
#   undef  _ppz_impl_F32_SETVAL
  }

  _ppz_impl_INLINE
  SIMD<float>::Type simd_add(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _ppz_impl_F32_ADD(a, b);
#   undef  _ppz_impl_F32_ADD
  }

  _ppz_impl_INLINE
  SIMD<float>::Type simd_mul(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _ppz_impl_F32_MUL(a, b);
#   undef  _ppz_impl_F32_MUL
  }

  _ppz_impl_INLINE
  SIMD<float>::Type simd_sub(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _ppz_impl_F32_SUB(a, b);
#   undef  _ppz_impl_F32_SUB
  }

  _ppz_impl_INLINE
  SIMD<float>::Type simd_div(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _ppz_impl_F32_DIV(a, b);
#   undef  _ppz_impl_F32_DIV
  }

  _ppz_impl_INLINE
  SIMD<float>::Type simd_addmul(const SIMD<float>::Type& a, const SIMD<float>::Type& b, const SIMD<float>::Type& c) noexcept
  {
    return _ppz_impl_F32_ADDMUL(a, b, c);
#   undef  _ppz_impl_F32_ADDMUL
  }

  _ppz_impl_INLINE
  SIMD<float>::Type simd_submul(const SIMD<float>::Type& a, const SIMD<float>::Type& b, const SIMD<float>::Type& c) noexcept
  {
    return _ppz_impl_F32_SUBMUL(a, b, c);
#   undef  _ppz_impl_F32_SUBMUL
  }

  _ppz_impl_INLINE
  SIMD<float>::Type simd_sqrt(const SIMD<float>::Type& a) noexcept
  {
    return _ppz_impl_F32_SQRT(a);
#   undef  _ppz_impl_F32_SQRT
  }

  _ppz_impl_INLINE
  SIMD<float>::Type simd_abs(const SIMD<float>::Type& a) noexcept
  {
    return _ppz_impl_F32_ABS(a);
#   undef  _ppz_impl_F32_ABS
  }

  _ppz_impl_INLINE
  SIMD<float>::Mask simd_eq(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _ppz_impl_F32_EQ(a, b);
#   undef  _ppz_impl_F32_EQ
  }

  _ppz_impl_INLINE
  SIMD<float>::Mask simd_neq(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _ppz_impl_F32_NEQ(a, b);
#   undef  _ppz_impl_F32_NEQ
  }

  _ppz_impl_INLINE
  SIMD<float>::Mask simd_gt(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _ppz_impl_F32_GT(a, b);
#   undef  _ppz_impl_F32_GT
  }

  _ppz_impl_INLINE
  SIMD<float>::Mask simd_gte(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _ppz_impl_F32_GTE(a, b);
#   undef  _ppz_impl_F32_GTE
  }

  _ppz_impl_INLINE
  SIMD<float>::Mask simd_lt(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _ppz_impl_F32_LT(a, b);
#   undef  _ppz_impl_F32_LT
  }

  _ppz_impl_INLINE
  SIMD<float>::Mask simd_lte(const SIMD<float>::Type& a, const SIMD<float>::Type& b) noexcept
  {
    return _ppz_impl_F32_LTE(a, b);
#   undef  _ppz_impl_F32_LTE
  }
#endif

#if defined(_ppz_impl_AVX512F)
# define _ppz_impl_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(double, __m512d, __mmask8, 64, "AVX512F");
# define _ppz_impl_F64_LOADU(data)           _mm512_loadu_pd(data)
# define _ppz_impl_F64_LOADA(data)           _mm512_load_pd(data)
# define _ppz_impl_F64_STOREU(addr, data)    _mm512_storeu_pd(reinterpret_cast<void*>(addr), data)
# define _ppz_impl_F64_STOREA(addr, data)    _mm512_store_pd(reinterpret_cast<void*>(addr), data)
# define _ppz_impl_F64_SETVAL(value)         _mm512_set1_pd(value)
# define _ppz_impl_F64_SETZERO()             _mm512_setzero_pd()
# define _ppz_impl_F64_MUL(a, b)             _mm512_mul_pd(a, b)
# define _ppz_impl_F64_ADD(a, b)             _mm512_add_pd(a, b)
# define _ppz_impl_F64_SUB(a, b)             _mm512_sub_pd(a, b)
# define _ppz_impl_F64_DIV(a, b)             _mm512_div_pd(a, b)
# define _ppz_impl_F64_ADDMUL(a, b, c)       _mm512_fmadd_pd(b, c, a)
# define _ppz_impl_F64_SUBMUL(a, b, c)       _mm512_fnmadd_pd(a, b, c)
# define _ppz_impl_F64_SQRT(a)               _mm512_sqrt_pd(a)
# define _ppz_impl_F64_ABS(a)                _mm512_abs_pd(a)
# define _ppz_impl_F64_EQ(a, b)              _mm512_cmp_pd_mask(a, b, _CMP_EQ_UQ)
# define _ppz_impl_F64_NEQ(a, b)             _mm512_cmp_pd_mask(a, b, _CMP_NEQ_UQ)
# define _ppz_impl_F64_GT(a, b)              _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ)
# define _ppz_impl_F64_GTE(a, b)             _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ)
# define _ppz_impl_F64_LT(a, b)              _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ)
# define _ppz_impl_F64_LTE(a, b)             _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ)

#elif defined(_ppz_impl_FMA)
# define _ppz_impl_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(double, __m256d, __m256d, 32, "AVX, FMA");
# define _ppz_impl_F64_LOADU(data)           _mm256_loadu_pd(data)
# define _ppz_impl_F64_LOADA(data)           _mm256_load_pd(data)
# define _ppz_impl_F64_STOREU(addr, data)    _mm256_storeu_pd(addr, data)
# define _ppz_impl_F64_STOREA(addr, data)    _mm256_store_pd(addr, data)
# define _ppz_impl_F64_SETVAL(value)         _mm256_set1_pd(value)
# define _ppz_impl_F64_SETZERO()             _mm256_setzero_pd()
# define _ppz_impl_F64_MUL(a, b)             _mm256_mul_pd(a, b)
# define _ppz_impl_F64_ADD(a, b)             _mm256_add_pd(a, b)
# define _ppz_impl_F64_SUB(a, b)             _mm256_sub_pd(a, b)
# define _ppz_impl_F64_DIV(a, b)             _mm256_div_pd(a, b)
# define _ppz_impl_F64_ADDMUL(a, b, c)       _mm256_fmadd_pd(b, c, a)
# define _ppz_impl_F64_SUBMUL(a, b, c)       _mm256_fnmadd_pd(a, b, c)
# define _ppz_impl_F64_SQRT(a)               _mm256_sqrt_pd(a)
# define _ppz_impl_F64_ABS(a)                _mm256_andnot_pd(_mm256_set1_pd(-0.0), a)
# define _ppz_impl_F64_EQ(a, b)              _mm256_cmp_pd(a, b, _CMP_EQ_UQ)
# define _ppz_impl_F64_NEQ(a, b)             _mm256_cmp_pd(a, b, _CMP_NEQ_UQ)
# define _ppz_impl_F64_GT(a, b)              _mm256_cmp_pd(a, b, _CMP_GT_OQ)
# define _ppz_impl_F64_GTE(a, b)             _mm256_cmp_pd(a, b, _CMP_GE_OQ)
# define _ppz_impl_F64_LT(a, b)              _mm256_cmp_pd(a, b, _CMP_LT_OQ)
# define _ppz_impl_F64_LTE(a, b)             _mm256_cmp_pd(a, b, _CMP_LE_OQ)

#elif defined(_ppz_impl_AVX)
# define _ppz_impl_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(double, __m256d, __m256d, 32, "AVX");
# define _ppz_impl_F64_LOADU(data)           _mm256_loadu_pd(data)
# define _ppz_impl_F64_LOADA(data)           _mm256_load_pd(data)
# define _ppz_impl_F64_STOREU(addr, data)    _mm256_storeu_pd(addr, data)
# define _ppz_impl_F64_STOREA(addr, data)    _mm256_store_pd(addr, data)
# define _ppz_impl_F64_SETVAL(value)         _mm256_set1_pd(value)
# define _ppz_impl_F64_SETZERO()             _mm256_setzero_pd()
# define _ppz_impl_F64_MUL(a, b)             _mm256_mul_pd(a, b)
# define _ppz_impl_F64_ADD(a, b)             _mm256_add_pd(a, b)
# define _ppz_impl_F64_SUB(a, b)             _mm256_sub_pd(a, b)
# define _ppz_impl_F64_DIV(a, b)             _mm256_div_pd(a, b)
# define _ppz_impl_F64_ADDMUL(a, b, c)       _mm256_add_pd(a, _mm256_mul_pd(b, c))
# define _ppz_impl_F64_SUBMUL(a, b, c)       _mm256_sub_pd(a, _mm256_mul_pd(b, c))
# define _ppz_impl_F64_SQRT(a)               _mm256_sqrt_pd(a)
# define _ppz_impl_F64_ABS(a)                _mm256_andnot_pd(_mm256_set1_pd(-0.0), a)
# define _ppz_impl_F64_EQ(a, b)              _mm256_cmp_pd(a, b, _CMP_EQ_UQ)
# define _ppz_impl_F64_NEQ(a, b)             _mm256_cmp_pd(a, b, _CMP_NEQ_UQ)
# define _ppz_impl_F64_GT(a, b)              _mm256_cmp_pd(a, b, _CMP_GT_OQ)
# define _ppz_impl_F64_GTE(a, b)             _mm256_cmp_pd(a, b, _CMP_GE_OQ)
# define _ppz_impl_F64_LT(a, b)              _mm256_cmp_pd(a, b, _CMP_LT_OQ)
# define _ppz_impl_F64_LTE(a, b)             _mm256_cmp_pd(a, b, _CMP_LE_OQ)

#elif defined(_ppz_impl_SSE2)
# define _ppz_impl_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(double, __m128d, __m128d, 16, "SSE2");
# define _ppz_impl_F64_LOADU(data)           _mm_loadu_pd(data)
# define _ppz_impl_F64_LOADA(data)           _mm_load_pd(data)
# define _ppz_impl_F64_STOREU(addr, data)    _mm_storeu_pd(addr, data)
# define _ppz_impl_F64_STOREA(addr, data)    _mm_store_pd(addr, data)
# define _ppz_impl_F64_SETVAL(value)         _mm_set1_pd(value)
# define _ppz_impl_F64_SETZERO()             _mm_setzero_pd()
# define _ppz_impl_F64_MUL(a, b)             _mm_mul_pd(a, b)
# define _ppz_impl_F64_ADD(a, b)             _mm_add_pd(a, b)
# define _ppz_impl_F64_SUB(a, b)             _mm_sub_pd(a, b)
# define _ppz_impl_F64_DIV(a, b)             _mm_div_pd(a, b)
# define _ppz_impl_F64_ADDMUL(a, b, c)       _mm_add_pd(a, _mm_mul_pd(b, c))
# define _ppz_impl_F64_SUBMUL(a, b, c)       _mm_sub_pd(a, _mm_mul_pd(b, c))
# define _ppz_impl_F64_SQRT(a)               _mm_sqrt_pd(a)
# define _ppz_impl_F64_ABS(a)                _mm_andnot_pd(_mm_set1_pd(-0.0), a)
# define _ppz_impl_F64_EQ(a, b)              _mm_cmpeq_pd(a, b)
# define _ppz_impl_F64_NEQ(a, b)             _mm_cmpneq_pd(a, b)
# define _ppz_impl_F64_GT(a, b)              _mm_cmpgt_pd(a, b)
# define _ppz_impl_F64_GTE(a, b)             _mm_cmpge_pd(a, b)
# define _ppz_impl_F64_LT(a, b)              _mm_cmplt_pd(a, b)
# define _ppz_impl_F64_LTE(a, b)             _mm_cmple_pd(a, b)

// #elif defined(_ppz_impl_NEON) or defined(_ppz_impl_NEON64)
// # define _ppz_impl_F64
//   static_assert(sizeof(double) == 8, "float must be 64 bit");
//   _ppz_impl_MAKE_SIMD_SPECIALIZATION(double, float64x4_t, float64x4_t, 0, "NEON");
// # define _ppz_impl_F64_LOADU(data)           vld1q_f64(data)
// # define _ppz_impl_F64_LOADA(data)           vld1q_f64(data)
// # define _ppz_impl_F64_STOREU(addr, data)    vst1q_f64(addr, data)
// # define _ppz_impl_F64_STOREA(addr, data)    vst1q_f64(addr, data)
// # define _ppz_impl_F64_SETVAL(value)         vdupq_n_f64(value)
// # define _ppz_impl_F64_SETZERO()             vdupq_n_f64(0.0)
// # define _ppz_impl_F64_MUL(a, b)             vmulq_f64(a, b)
// # define _ppz_impl_F64_ADD(a, b)             vaddq_f64(a, b)
// # define _ppz_impl_F64_SUB(a, b)             vsubq_f64(a, b)
// # define _ppz_impl_F64_DIV(a, b)             vdivq_f64(a, b)
// # define _ppz_impl_F64_ADDMUL(a, b, c)       vmlaq_f64(a, b, c)
// # define _ppz_impl_F64_SUBMUL(a, b, c)       vmlsq_f64(a, b, c)
// # define _ppz_impl_F64_SQRT(a)               vsqrtq_f64(a)
// # define _ppz_impl_F32_ABS(a)                vabsq_f64(a)
// # define _ppz_impl_F32_EQ(a, b)              vceqq_f64(a, b)
// # define _ppz_impl_F32_NEQ(a, b)             vmvnq_u64(vceqq_f64(a, b))
// # define _ppz_impl_F32_GT(a, b)              vcgtq_f64(a, b)
// # define _ppz_impl_F32_GTE(a, b)             vcgeq_f64(a, b)
// # define _ppz_impl_F32_LT(a, b)              vcltq_f64(a, b)
// # define _ppz_impl_F32_LTE(a, b)             vcleq_f64(a, b)
#endif

#ifdef _ppz_impl_F64
  template<>
  _ppz_impl_INLINE
  SIMD<double>::Type simd_setzero<double>() noexcept
  {
    return _ppz_impl_F64_SETZERO();
#   undef  _ppz_impl_F64_SETZERO
  }

  _ppz_impl_INLINE
  SIMD<double>::Type simd_loadu(const double* const _ppz_impl_RESTRICT data) noexcept
  {
    return _ppz_impl_F64_LOADU(data);
#   undef  _ppz_impl_F64_LOADU
  }

  _ppz_impl_INLINE
  SIMD<double>::Type simd_loada(const double* const _ppz_impl_RESTRICT data) noexcept
  {
    return _ppz_impl_F64_LOADA(data);
#   undef  _ppz_impl_F64_LOADA
  }

  _ppz_impl_INLINE
  void simd_storeu(double* const _ppz_impl_RESTRICT addr, const SIMD<double>::Type& data)
  {
    _ppz_impl_F64_STOREU(addr, data);
#   undef _ppz_impl_F64_STOREU
  }

  _ppz_impl_INLINE
  void simd_storea(double* const _ppz_impl_RESTRICT addr, const SIMD<double>::Type& data)
  {
    _ppz_impl_F64_STOREA(addr, data);
#   undef _ppz_impl_F64_STOREA
  }

  template<>
  _ppz_impl_INLINE
  SIMD<double>::Type simd_setval(const double value) noexcept
  {
    return _ppz_impl_F64_SETVAL(value);
#   undef  _ppz_impl_F64_SETVAL
  }

  _ppz_impl_INLINE
  SIMD<double>::Type simd_add(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _ppz_impl_F64_ADD(a, b);
#   undef  _ppz_impl_F64_ADD
  }

  _ppz_impl_INLINE
  SIMD<double>::Type simd_mul(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _ppz_impl_F64_MUL(a, b);
#   undef  _ppz_impl_F64_MUL
  }

  _ppz_impl_INLINE
  SIMD<double>::Type simd_sub(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _ppz_impl_F64_SUB(a, b);
#   undef  _ppz_impl_F64_SUB
  }

  _ppz_impl_INLINE
  SIMD<double>::Type simd_div(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _ppz_impl_F64_DIV(a, b);
#   undef  _ppz_impl_F64_DIV
  }

  _ppz_impl_INLINE
  SIMD<double>::Type simd_addmul(const SIMD<double>::Type& a, const SIMD<double>::Type& b, const SIMD<double>::Type& c) noexcept
  {
    return _ppz_impl_F64_ADDMUL(a, b, c);
#   undef  _ppz_impl_F64_ADDMUL
  }

  _ppz_impl_INLINE
  SIMD<double>::Type simd_submul(const SIMD<double>::Type& a, const SIMD<double>::Type& b, const SIMD<double>::Type& c) noexcept
  {
    return _ppz_impl_F64_SUBMUL(a, b, c);
#   undef  _ppz_impl_F64_SUBMUL
  }

  _ppz_impl_INLINE
  SIMD<double>::Type simd_sqrt(const SIMD<double>::Type& a) noexcept
  {
    return _ppz_impl_F64_SQRT(a);
#   undef  _ppz_impl_F64_SQRT
  }

  _ppz_impl_INLINE
  SIMD<double>::Type simd_abs(const SIMD<double>::Type& a) noexcept
  {
    return _ppz_impl_F64_ABS(a);
#   undef  _ppz_impl_F64_ABS
  }

  _ppz_impl_INLINE
  SIMD<double>::Mask simd_eq(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _ppz_impl_F64_EQ(a, b);
#   undef  _ppz_impl_F64_EQ
  }

  _ppz_impl_INLINE
  SIMD<double>::Mask simd_neq(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _ppz_impl_F64_NEQ(a, b);
#   undef  _ppz_impl_F64_NEQ
  }

  _ppz_impl_INLINE
  SIMD<double>::Mask simd_gt(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _ppz_impl_F64_GT(a, b);
#   undef  _ppz_impl_F64_GT
  }

  _ppz_impl_INLINE
  SIMD<double>::Mask simd_gte(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _ppz_impl_F64_GTE(a, b);
#   undef  _ppz_impl_F64_GTE
  }

  _ppz_impl_INLINE
  SIMD<double>::Mask simd_lt(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _ppz_impl_F64_LT(a, b);
#   undef  _ppz_impl_F64_LT
  }

  _ppz_impl_INLINE
  SIMD<double>::Mask simd_lte(const SIMD<double>::Type& a, const SIMD<double>::Type& b) noexcept
  {
    return _ppz_impl_F64_LTE(a, b);
#   undef  _ppz_impl_F64_LTE
  }
#endif

#if defined(_ppz_impl_AVX512F)
# define _ppz_impl_I32
  static_assert(sizeof(int32_t) == 4, "int32_t must be 32 bit");
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(int32_t,  __m512i, __mmask16, 64, "AVX512F");
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(uint32_t, __m512i, __mmask16, 64, "AVX512F");
# define _ppz_impl_I32_LOADU(data)           _mm512_loadu_si512(data)
# define _ppz_impl_I32_LOADA(data)           _mm512_load_si512(data)
# define _ppz_impl_I32_STOREU(addr, data)    _mm512_storeu_si512(reinterpret_cast<void*>(addr), data)
# define _ppz_impl_I32_STOREA(addr, data)    _mm512_store_si512(reinterpret_cast<void*>(addr), data)
# define _ppz_impl_I32_SETVAL(value)         _mm512_set1_epi32(value)
# define _ppz_impl_I32_SETZERO()             _mm512_setzero_si512()
# define _ppz_impl_I32_MUL(a, b)             _mm512_mullo_epi32 (a, b)
# define _ppz_impl_I32_ADD(a, b)             _mm512_add_epi32(a, b)
# define _ppz_impl_I32_SUB(a, b)             _mm512_sub_epi32(a, b)
# define _ppz_impl_I32_DIV(a, b)             _mm512_cvtps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(a), _mm512_cvtepi32_ps(b)))
# define _ppz_impl_I32_ADDMUL(a, b, c)       _mm512_add_epi32(a, _mm512_mul_epi32(b, c))
# define _ppz_impl_I32_SUBMUL(a, b, c)       _mm512_sub_epi32(a, _mm512_mul_epi32(b, c))
# define _ppz_impl_I32_SQRT(a)               _mm512_cvtps_epi32(_mm512_sqrt_ps(_mm512_cvtepi32_ps(a)))
# define _ppz_impl_I32_ABS(a)                _mm512_abs_epi32(a)
# if defined(_ppz_impl_SVML)
#   undef  _ppz_impl_I32_DIV
#   define _ppz_impl_I32_DIV(a, b)           _mm512_div_epi32(a, b)
# endif

# define _ppz_impl_I32_EQ(a, b)              _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_EQ)
# define _ppz_impl_I32_NEQ(a, b)             _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NE)
# define _ppz_impl_I32_GT(a, b)              _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLE)
# define _ppz_impl_I32_GTE(a, b)             _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLT)
# define _ppz_impl_I32_LT(a, b)              _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LT)
# define _ppz_impl_I32_LTE(a, b)             _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LE)

# define _ppz_impl_I32_BW_NOT(a)             _mm512_xor_si512(a, _mm512_set1_epi32(-1))
# define _ppz_impl_I32_BW_AND(a, b)          _mm512_and_si512(a, b)
# define _ppz_impl_I32_BW_NAND(a, b)         _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_set1_epi32(-1))
# define _ppz_impl_I32_BW_OR(a, b)           _mm512_or_si512(a, b)
# define _ppz_impl_I32_BW_NOR(a, b)          _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_set1_epi32(-1))
# define _ppz_impl_I32_BW_XOR(a, b)          _mm512_xor_si512(a, b)
# define _ppz_impl_I32_BW_XNOR(a, b)         _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_set1_epi32(-1))

#elif defined(_ppz_impl_AVX2)
# define _ppz_impl_I32
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(int32_t,  __m256i, __m256i, 32, "AVX2, AVX");
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(uint32_t, __m256i, __m256i, 32, "AVX2, AVX");
# define _ppz_impl_I32_LOADU(data)           _mm256_loadu_si256((const __m256i*)data)
# define _ppz_impl_I32_LOADA(data)           _mm256_load_si256((const __m256i*)data)
# define _ppz_impl_I32_STOREU(addr, data)    _mm256_storeu_si256 ((__m256i*)addr, data)
# define _ppz_impl_I32_STOREA(addr, data)    _mm256_store_si256((__m256i*)addr, data)
# define _ppz_impl_I32_SETVAL(value)         _mm256_set1_epi32(value)
# define _ppz_impl_I32_SETZERO()             _mm256_setzero_si256()
# define _ppz_impl_I32_MUL(a, b)             _mm256_mullo_epi32(a, b)
# define _ppz_impl_I32_ADD(a, b)             _mm256_add_epi32(a, b)
# define _ppz_impl_I32_SUB(a, b)             _mm256_sub_epi32(a, b)
# define _ppz_impl_I32_DIV(a, b)             _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(a), _mm256_cvtepi32_ps(b)))
# define _ppz_impl_I32_ADDMUL(a, b, c)       _mm256_add_epi32(a, _mm256_mul_epi32(b, c))
# define _ppz_impl_I32_SUBMUL(a, b, c)       _mm256_sub_epi32(a, _mm256_mul_epi32(b, c))
# define _ppz_impl_I32_SQRT(a)               _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(a)))
# define _ppz_impl_I32_ABS(a)                _mm256_abs_epi32(a)
# if defined(_ppz_impl_SVML)
#   undef  _ppz_impl_I32_DIV
#   define _ppz_impl_I32_DIV(a, b)           _mm256_div_epi32(a, b)
# endif

# define _ppz_impl_I32_EQ(a, b)              _mm256_cmpeq_epi32(a, b)
# define _ppz_impl_I32_NEQ(a, b)             _mm256_xor_si256(_mm256_cmpeq_epi32(a, b), _mm256_cmpeq_epi32(a, a))
# define _ppz_impl_I32_GT(a, b)              _mm256_cmpgt_epi32(a, b)
# define _ppz_impl_I32_GTE(a, b)             _mm256_xor_si256(_mm256_cmpgt_epi32(b, a), _mm256_cmpeq_epi32(a, a))
# define _ppz_impl_I32_LT(a, b)              _mm256_cmpgt_epi32(b, a)
# define _ppz_impl_I32_LTE(a, b)             _mm256_xor_si256(_mm256_cmpgt_epi32(a, b), _mm256_cmpeq_epi32(a, a))

# define _ppz_impl_I32_BW_NOT(a)             _mm256_xor_si256(a, _mm256_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_AND(a, b)          _mm256_and_si256(a, b)
# define _ppz_impl_I32_BW_NAND(a, b)         _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_OR(a, b)           _mm256_or_si256(a, b)
# define _ppz_impl_I32_BW_NOR(a, b)          _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_XOR(a, b)          _mm256_xor_si256(a, b)
# define _ppz_impl_I32_BW_XNOR(a, b)         _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_cmpeq_epi32(a, a))

#elif defined(_ppz_impl_SSE4_1)
# define _ppz_impl_I32
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(int32_t,  __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(uint32_t, __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
# define _ppz_impl_I32_LOADU(data)           _mm_loadu_si128((const __m128i*)data)
# define _ppz_impl_I32_LOADA(data)           _mm_load_si128((const __m128i*)data)
# define _ppz_impl_I32_STOREU(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
# define _ppz_impl_I32_STOREA(addr, data)    _mm_store_si128((__m128i*)addr, data)
# define _ppz_impl_I32_SETVAL(value)         _mm_set1_epi32(value)
# define _ppz_impl_I32_SETZERO()             _mm_setzero_si128()
# define _ppz_impl_I32_MUL(a, b)             _mm_mullo_epi32(a, b)
# define _ppz_impl_I32_ADD(a, b)             _mm_add_epi32(a, b)
# define _ppz_impl_I32_SUB(a, b)             _mm_sub_epi32(a, b)
# define _ppz_impl_I32_DIV(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define _ppz_impl_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_mul_epi32(b, c))
# define _ppz_impl_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_mul_epi32(b, c))
# define _ppz_impl_I32_SQRT(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
# define _ppz_impl_I32_ABS(a)                _mm_abs_epi32(a)
# if defined(_ppz_impl_SVML)
#   undef  _ppz_impl_I32_DIV
#   define _ppz_impl_I32_DIV(a, b)           _mm_div_epi32(a, b)
# endif

# define _ppz_impl_I32_EQ(a, b)              _mm_cmpeq_epi32(a, b)
# define _ppz_impl_I32_NEQ(a, b)             _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_GT(a, b)              _mm_cmpgt_epi32(a, b)
# define _ppz_impl_I32_GTE(a, b)             _mm_xor_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_LT(a, b)              _mm_cmplt_epi32(a, b)
# define _ppz_impl_I32_LTE(a, b)             _mm_xor_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, a))

# define _ppz_impl_I32_BW_NOT(a)             _mm_xor_si128(a, _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_AND(a, b)          _mm_and_si128(a, b)
# define _ppz_impl_I32_BW_NAND(a, b)         _mm_xor_si128(_mm_and_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_OR(a, b)           _mm_or_si128(a, b)
# define _ppz_impl_I32_BW_NOR(a, b)          _mm_xor_si128(_mm_or_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_XOR(a, b)          _mm_xor_si128(a, b)
# define _ppz_impl_I32_BW_XNOR(a, b)         _mm_xor_si128(_mm_xor_si128(a, b), _mm_cmpeq_epi32(a, a))

#elif defined(_ppz_impl_SSSE3)
# define _ppz_impl_I32
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(int32_t,  __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(uint32_t, __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
# define _ppz_impl_I32_LOADU(data)           _mm_loadu_si128((const __m128i*)data)
# define _ppz_impl_I32_LOADA(data)           _mm_load_si128((const __m128i*)data)
# define _ppz_impl_I32_STOREU(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
# define _ppz_impl_I32_STOREA(addr, data)    _mm_store_si128((__m128i*)addr, data)
# define _ppz_impl_I32_SETVAL(value)         _mm_set1_epi32(value)
# define _ppz_impl_I32_SETZERO()             _mm_setzero_si128()
# define _ppz_impl_I32_MUL(a, b)             _mm_mul_epu32(a, b)
# define _ppz_impl_I32_ADD(a, b)             _mm_add_epi32(a, b)
# define _ppz_impl_I32_SUB(a, b)             _mm_sub_epi32(a, b)
# define _ppz_impl_I32_DIV(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define _ppz_impl_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_mul_epu32(b, c))
# define _ppz_impl_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_mul_epu32(b, c))
# define _ppz_impl_I32_SQRT(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
# define _ppz_impl_I32_ABS(a)                _mm_abs_epi32(a)
# if defined(_ppz_impl_SVML)
#   undef  _ppz_impl_I32_DIV
#   define _ppz_impl_I32_DIV(a, b)           _mm_div_epi32(a, b)
# endif

# define _ppz_impl_I32_EQ(a, b)              _mm_cmpeq_epi32(a, b)
# define _ppz_impl_I32_NEQ(a, b)             _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_GT(a, b)              _mm_cmpgt_epi32(a, b)
# define _ppz_impl_I32_GTE(a, b)             _mm_xor_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_LT(a, b)              _mm_cmplt_epi32(a, b)
# define _ppz_impl_I32_LTE(a, b)             _mm_xor_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, a))

# define _ppz_impl_I32_BW_NOT(a)             _mm_xor_si128(a, _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_AND(a, b)          _mm_and_si128(a, b)
# define _ppz_impl_I32_BW_NAND(a, b)         _mm_xor_si128(_mm_and_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_OR(a, b)           _mm_or_si128(a, b)
# define _ppz_impl_I32_BW_NOR(a, b)          _mm_xor_si128(_mm_or_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_XOR(a, b)          _mm_xor_si128(a, b)
# define _ppz_impl_I32_BW_XNOR(a, b)         _mm_xor_si128(_mm_xor_si128(a, b), _mm_cmpeq_epi32(a, a))

#elif defined(_ppz_impl_SSE2)
# define _ppz_impl_I32
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(int32_t,  __m128i, __m128i, 16, "SSE2, SSE");
  _ppz_impl_MAKE_SIMD_SPECIALIZATION(uint32_t, __m128i, __m128i, 16, "SSE2, SSE");
# define _ppz_impl_I32_LOADU(data)           _mm_loadu_si128((const __m128i*)data)
# define _ppz_impl_I32_LOADA(data)           _mm_load_si128((const __m128i*)data)
# define _ppz_impl_I32_STOREU(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
# define _ppz_impl_I32_STOREA(addr, data)    _mm_store_si128((__m128i*)addr, data)
# define _ppz_impl_I32_SETVAL(value)         _mm_set1_epi32(value)
# define _ppz_impl_I32_SETZERO()             _mm_setzero_si128()
// # define _ppz_impl_I32_MUL(a, b)             _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define _ppz_impl_I32_MUL(a, b)             _mm_mul_epu32(a, b)
# define _ppz_impl_I32_ADD(a, b)             _mm_add_epi32(a, b)
# define _ppz_impl_I32_SUB(a, b)             _mm_sub_epi32(a, b)
# define _ppz_impl_I32_DIV(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
// # define _ppz_impl_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
// # define _ppz_impl_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
# define _ppz_impl_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_mul_epu32(b, c))
# define _ppz_impl_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_mul_epu32(b, c))
# define _ppz_impl_I32_SQRT(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
# define _ppz_impl_I32_ABS(a)                                   \
  [&]() -> __m128i {                                            \
    const __m128i signmask = _mm_srai_epi32(a, 31);             \
    return _mm_sub_epi32(_mm_xor_si128(a, signmask), signmask); \
  }()

# define _ppz_impl_I32_EQ(a, b)              _mm_cmpeq_epi32(a, b)
# define _ppz_impl_I32_NEQ(a, b)             _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_GT(a, b)              _mm_cmpgt_epi32(a, b)
# define _ppz_impl_I32_GTE(a, b)             _mm_xor_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_LT(a, b)              _mm_cmplt_epi32(a, b)
# define _ppz_impl_I32_LTE(a, b)             _mm_xor_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, a))

# define _ppz_impl_I32_BW_NOT(a)             _mm_xor_si128(a, _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_AND(a, b)          _mm_and_si128(a, b)
# define _ppz_impl_I32_BW_NAND(a, b)         _mm_xor_si128(_mm_and_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_OR(a, b)           _mm_or_si128(a, b)
# define _ppz_impl_I32_BW_NOR(a, b)          _mm_xor_si128(_mm_or_si128(a, b), _mm_cmpeq_epi32(a, a))
# define _ppz_impl_I32_BW_XOR(a, b)          _mm_xor_si128(a, b)
# define _ppz_impl_I32_BW_XNOR(a, b)         _mm_xor_si128(_mm_xor_si128(a, b), _mm_cmpeq_epi32(a, a))
#endif

#ifdef _ppz_impl_I32
  template<>
  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_setzero<int32_t>() noexcept
  {
    return _ppz_impl_I32_SETZERO();
  }

  template<>
  _ppz_impl_INLINE
  SIMD<uint32_t>::Type simd_setzero<uint32_t>() noexcept
  {
    return _ppz_impl_I32_SETZERO();
#   undef  _ppz_impl_I32_SETZERO
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_loadu(const int32_t* const _ppz_impl_RESTRICT data) noexcept
  {
    return _ppz_impl_I32_LOADU(data);
  }

  _ppz_impl_INLINE
  SIMD<uint32_t>::Type simd_loadu(const uint32_t* const _ppz_impl_RESTRICT data) noexcept
  {
    return _ppz_impl_I32_LOADU(data);
#   undef  _ppz_impl_I32_LOADU
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_loada(const int32_t* const _ppz_impl_RESTRICT data) noexcept
  {
    return _ppz_impl_I32_LOADA(data);
  }

  _ppz_impl_INLINE
  SIMD<uint32_t>::Type simd_loada(const uint32_t* const _ppz_impl_RESTRICT data) noexcept
  {
    return _ppz_impl_I32_LOADA(data);
#   undef  _ppz_impl_I32_LOADA
  }

  _ppz_impl_INLINE
  void simd_storeu(int32_t addr[], const SIMD<int32_t>::Type& data)
  {
    _ppz_impl_I32_STOREU(addr, data);
  }

  _ppz_impl_INLINE
  void simd_storeu(uint32_t addr[], const SIMD<uint32_t>::Type& data)
  {
    _ppz_impl_I32_STOREU(addr, data);
#   undef _ppz_impl_I32_STOREU
  }

  _ppz_impl_INLINE
  void simd_storea(int32_t addr[], const SIMD<int32_t>::Type& data)
  {
    _ppz_impl_I32_STOREA(addr, data);
  }

  _ppz_impl_INLINE
  void simd_storea(uint32_t addr[], const SIMD<uint32_t>::Type& data)
  {
    _ppz_impl_I32_STOREA(addr, data);
#   undef _ppz_impl_I32_STOREA
  }

  template<>
  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_setval(const int32_t value) noexcept
  {
    return _ppz_impl_I32_SETVAL(value);
  }

  template<>
  _ppz_impl_INLINE
  SIMD<uint32_t>::Type simd_setval(const uint32_t value) noexcept
  {
    return _ppz_impl_I32_SETVAL(value);
#   undef  _ppz_impl_I32_SETVAL
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_add(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_ADD(a, b);
#   undef  _ppz_impl_I32_ADD
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_mul(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_MUL(a, b);
#   undef  _ppz_impl_I32_MUL
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_sub(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_SUB(a, b);
#   undef  _ppz_impl_I32_SUB
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_div(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_DIV(a, b);
#   undef  _ppz_impl_I32_DIV
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_addmul(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b, const SIMD<int32_t>::Type& c) noexcept
  {
    return _ppz_impl_I32_ADDMUL(a, b, c);
#   undef  _ppz_impl_I32_ADDMUL
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_submul(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b, const SIMD<int32_t>::Type& c) noexcept
  {
    return _ppz_impl_I32_SUBMUL(a, b, c);
#   undef  _ppz_impl_I32_SUBMUL
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_sqrt(const SIMD<int32_t>::Type& a) noexcept
  {
    return _ppz_impl_I32_SQRT(a);
#   undef  _ppz_impl_I32_SQRT
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_abs(const SIMD<int32_t>::Type& a) noexcept
  {
    return _ppz_impl_I32_ABS(a);
#   undef  _ppz_impl_I32_ABS
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Mask simd_eq(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_EQ(a, b);
#   undef  _ppz_impl_I32_EQ
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Mask simd_neq(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_NEQ(a, b);
#   undef  _ppz_impl_I32_NEQ
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Mask simd_gt(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_GT(a, b);
#   undef  _ppz_impl_I32_GT
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Mask simd_gte(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_GTE(a, b);
#   undef  _ppz_impl_I32_GTE
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Mask simd_lt(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_LT(a, b);
#   undef  _ppz_impl_I32_LT
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Mask simd_lte(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_LTE(a, b);
#   undef  _ppz_impl_I32_LTE
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_compl(const SIMD<int32_t>::Type& a) noexcept
  {
    return _ppz_impl_I32_BW_NOT(a);
#   undef  _ppz_impl_I32_BW_NOT
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_and(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_BW_AND(a, b);
#   undef  _ppz_impl_I32_BW_AND
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_nand(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_BW_NAND(a, b);
#   undef  _ppz_impl_I32_BW_NAND
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_or(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_BW_OR(a, b);
#   undef  _ppz_impl_I32_BW_OR
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_nor(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_BW_NOR(a, b);
#   undef  _ppz_impl_I32_BW_NOR
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_xor(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_BW_XOR(a, b);
#   undef  _ppz_impl_I32_BW_XOR
  }

  _ppz_impl_INLINE
  SIMD<int32_t>::Type simd_xnor(const SIMD<int32_t>::Type& a, const SIMD<int32_t>::Type& b) noexcept
  {
    return _ppz_impl_I32_BW_XNOR(a, b);
#   undef  _ppz_impl_I32_BW_XNOR
  }
#endif
}

  inline _ppz_impl_CONSTEXPR_CPP14
  std::ostream& operator<<(std::ostream& ostream, const ppz::SIMD<double>::Type& vector) noexcept
  {
    for (size_t k = 0; k < ppz::SIMD<double>::size; ++k)
    {
      if _ppz_impl_EXPECTED(k != 0)
      {
        ostream << ' ';
      }

      ostream << reinterpret_cast<const double*>(&vector)[k];
    }

    return ostream;
  }

  inline _ppz_impl_CONSTEXPR_CPP14
  std::ostream& operator<<(std::ostream& ostream, const ppz::SIMD<float>::Type& vector) noexcept
  {
    for (size_t k = 0; k < ppz::SIMD<float>::size; ++k)
    {
      if _ppz_impl_EXPECTED(k != 0)
      {
        ostream << ' ';
      }

      ostream << reinterpret_cast<const float*>(&vector)[k];
    }

    return ostream;
  }

  inline _ppz_impl_CONSTEXPR_CPP14
  std::ostream& operator<<(std::ostream& ostream, const ppz::SIMD<int32_t>::Type& vector) noexcept
  {
    for (size_t k = 0; k < ppz::SIMD<int32_t>::size; ++k)
    {
      if _ppz_impl_EXPECTED(k != 0)
      {
        ostream << ' ';
      }

      ostream << reinterpret_cast<const int32_t*>(&vector)[k];
    }

    return ostream;
  }
#undef _ppz_impl_INLINE
#undef _ppz_impl_PRAGMA
#undef _ppz_impl_CLANG_IGNORE
#undef _ppz_impl_UNLIKELY
#undef _ppz_impl_LIKELY
#undef _ppz_impl_THREADLOCAL
#undef _ppz_impl_DECLARE_MUTEX
#undef _ppz_impl_DECLARE_LOCK
#undef _ppz_impl_MAKE_SIMD_SPECIALIZATION
#else
#error "ppz: Support for ISO C++11 is required"
#endif
#endif