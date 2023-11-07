// include this file to support add built-in array manipulation capabilities
#if not defined(PARALLILOS_HPP)
# error "you must #include Parallilos.hpp before Parallilos_array.hpp"
#else
#ifndef PARALLILOS_ARRAY_HPP
#define PARALLILOS_ARRAY_HPP
namespace Parallilos
{
  inline namespace Frontend
  {
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

    template<typename T>
    void add_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      // for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
      // {
      //   simd_storea(r+k, simd_add(simd_loada(a+k), simd_loada(b+k)));
      // }

      for (size_t k : simd<T>(n))
      {
        simd_storea(r+k, simd_add(simd_loada(a+k), simd_loada(b+k)));
      }

      // sequential fallback
      for (; k < n; ++k)
      {
        r[k] = a[k] + b[k];
      }
    }

    template<typename T>
    void add_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      const typename simd<T>::type a_vector = simd_setval(a);
      if (is_aligned(a) && is_aligned(b) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_add(a_vector, simd_loada(b+k)));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_add(a_vector, simd_loadu(b+k)));

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a + b[k];
    }

    template<typename T>
    void add_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      const typename simd<T>::type b_vector = simd_setval(b);
      if (is_aligned(a) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_add(simd_loada(a+k), b_vector));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_add(simd_loadu(a+k), b_vector));

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a[k] + b;
    }

    template<typename T>
    void mul_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (is_aligned(a) && is_aligned(b) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_mul(simd_loada(a+k), simd_loada(b+k)));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_mul(simd_loadu(a+k), simd_loadu(b+k)));

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a[k] * b[k];
    }

    template<typename T>
    void mul_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      const typename simd<T>::type a_vector = simd_setval(a);
      if (is_aligned(b) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_mul(a_vector, simd_loada(b+k)));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_mul(a_vector, simd_loadu(b+k)));

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a * b[k];
    }

    template<typename T>
    void mul_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      const typename simd<T>::type b_vector = simd_setval(b);
      if (is_aligned(a) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_mul(simd_loada(a+k), b_vector));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_mul(simd_loadu(a+k), b_vector));

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a[k] * b;
    }

    template<typename T>
    void sub_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (is_aligned(a) && is_aligned(b) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_sub(simd_loada(a+k), simd_loada(b+k)));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_sub(simd_loadu(a+k), simd_loadu(b+k)));

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a[k] - b[k];
    }

    template<typename T>
    void sub_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      const typename simd<T>::type a_vector = simd_setval(a);
      if (is_aligned(b) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_sub(a_vector, simd_loada(b+k)));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_sub(a_vector, simd_loadu(b+k)));

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a - b[k];
    }

    template<typename T>
    void sub_arrays(const T* a, const T b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      const typename simd<T>::type b_vector = simd_setval(b);
      if (is_aligned(a) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_sub(simd_loada(a+k), b_vector));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_sub(simd_loadu(a+k), b_vector));

      // sequential fallback
      for (; k < n; ++k)
        r[k] = a[k] - b;
    }

    template<typename T>
    void div_arrays(const T* a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      if (is_aligned(a) && is_aligned(b) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_div(simd_loada(a+k), simd_loada(b+k)));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_div(simd_loadu(a+k), simd_loadu(b+k)));

      // sequential fallback
      for (; k < n; ++k)
        r[k] = (b[k] != 0) ? a[k] / b[k] : std::numeric_limits<T>::min();
    }

    template<typename T>
    void div_arrays(const T a, const T* b, T* r, const size_t n)
    {
      size_t k = 0;

      // parallel addition using SIMD
      const typename simd<T>::type a_vector = simd_setval(a);
      if (is_aligned(b) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_div(a_vector, simd_loada(b+k)));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_div(a_vector, simd_loadu(b+k)));

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
      const typename simd<T>::type b_vector = simd_setval(b);
      if (is_aligned(a) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_div(simd_loada(a+k), b_vector));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_div(simd_loadu(a+k), b_vector));

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
      if (is_aligned(a) && is_aligned(r))
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storea(r+k, simd_sqrt(simd_loada(a+k)));
      else
        for (size_t passes_left = simd<T>::passes(n); passes_left; --passes_left, k += simd<T>::size)
          simd_storeu(r+k, simd_sqrt(simd_loadu(a+k)));

      // sequential fallback
      for (; k < n; ++k)
        r[k] = std::sqrt(a[k]);
    }
  }
}
#endif
#endif