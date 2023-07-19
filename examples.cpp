#include <cstddef>
#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include "include/Parallilos.hpp"
//#include "experimental/int32_support.hpp"

template<typename T>
void print_array(T* array, size_t n)
{
  for (size_t k = 0; k < n; ++k)
    std::cout << array[k] << ' ';
  std::cout << '\n';
  
  std::cout << "SIMD instruction set: " << Parallilos::simd_properties<T>::set << '\n';
  std::cout << "SIMD passes: " << Parallilos::simd_properties<T>::iterations(n) << '\n';
  std::cout << "Sequential passes: " << Parallilos::simd_properties<T>::sequential(n) << '\n';
}

template<typename type>
void stack_array()
{
  // stack allocated array
  const size_t n = 20;
  type a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  type b[] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  type c[n];

  // SIMD-enhanced
  Parallilos::add_arrays(a, b, c, n);

  // display result array
  print_array(c, n);
}

template<typename type>
void standard_vector()
{
  // dynamic array
  const size_t n = 20;
  std::vector<type> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<type> b = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<type> c(n);

  // SIMD-enhanced
  Parallilos::mul_arrays(a.data(), b.data(), c.data(), n);

  // display result array
  print_array(c.data(), n);
}

template<typename type>
void heap_array()
{
  // heap allocated array
  const size_t n = 20;
  std::unique_ptr<type[]> a = std::unique_ptr<type[]>(new type[n]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::unique_ptr<type[]> b = std::unique_ptr<type[]>(new type[n]{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::unique_ptr<type[]> c = std::unique_ptr<type[]>(new type[n]);

  // SIMD-enhanced
  Parallilos::sub_arrays(a.get(), b.get(), c.get(), n);

  // display result array
  print_array(c.get(), n);
}

template<typename type>
void standard_array()
{
  // fixed-size array
  const size_t n = 20;
  std::array<type, n> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<type, n> b = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<type, n> c;

  // SIMD-enhanced
  Parallilos::div_arrays(a.data(), b.data(), c.data(), n);

  // display result array
  print_array(c.data(), n);
}

template<typename type>
void custom_implementation()
{
  using namespace Parallilos;

  // aligned memory allocation
  const size_t n = 16;
  unique_array<type> a = get_array<type>(n);
  unique_array<type> b = get_array<type>(n);
  unique_array<type> c = get_array<type>(n);

  // initialize arrays
  for (size_t k = 0; k < n; ++k) {
    a[k] = k;
    b[k] = k + n;
  }
  
  // SIMD-enhanced
  const size_t iterations = simd_properties<type>::iterations(n);
  size_t k = 0;
  for (size_t i = 0; i < iterations; ++i, k+=simd_properties<type>::size)
    simd_storea(c.get()+k, simd_add(simd_loada(a.get()+k), simd_loada(b.get()+k)));
  for (; k < n; ++k)
    c[k] = a[k] + b[k];

  // display result array
  print_array(c.get(), n);
}

int main()
{
  std::cout << "\nstack allocated array example:\n";
  stack_array<float>();
  
  std::cout << "\nheap allocated array example:\n";
  heap_array<double>();

  std::cout << "\nvector example:\n";
  standard_vector<double>();

  std::cout << "\nstandard array example:\n";
  standard_array<float>();

  std::cout << "\ncustom implementation example:\n";
  custom_implementation<float>();
}