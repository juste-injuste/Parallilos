#include <cstddef>
#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include "include/Parallilos.hpp"
//#include "experimental/int32_support.hpp"

template<typename T>
void print_info(T* array, size_t n)
{
  for (size_t k = 0; k < n; ++k)
    std::cout << array[k] << ' ';
  std::cout << '\n';
  
  std::cout << "SIMD instruction set: " << Parallilos::simd_properties<T>::set << '\n';
  std::cout << "SIMD passes: " << Parallilos::simd_properties<T>::iterations(n) << '\n';
  std::cout << "Sequential passes: " << Parallilos::simd_properties<T>::sequential(n) << '\n';
}

template<typename T>
void stack_array()
{
  // stack array
  const size_t n = 20;
  T a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  T b[] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  T c[n];

  // SIMD-enhanced operation
  Parallilos::add_arrays(a, b, c, n);

  // display result array
  print_info(c, n);
}

template<typename T>
void standard_vector()
{
  // dynamic array
  const size_t n = 20;
  std::vector<T> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<T> b = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<T> c(n);

  // SIMD-enhanced operation
  Parallilos::mul_arrays(a.data(), b.data(), c.data(), n);

  // display result array
  print_info(c.data(), n);
}

template<typename T>
void heap_array()
{
  // heap array
  const size_t n = 20;
  std::unique_ptr<T[]> a = std::unique_ptr<T[]>(new T[n]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::unique_ptr<T[]> b = std::unique_ptr<T[]>(new T[n]{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::unique_ptr<T[]> c = std::unique_ptr<T[]>(new T[n]);

  // SIMD-enhanced operation
  Parallilos::sub_arrays(a.get(), b.get(), c.get(), n);

  // display result array
  print_info(c.get(), n);
}

template<typename T>
void standard_array()
{
  // standard array
  const size_t n = 20;
  std::array<T, n> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<T, n> b = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<T, n> c;

  // SIMD-enhanced operation
  Parallilos::div_arrays(a.data(), b.data(), c.data(), n);

  // display result array
  print_info(c.data(), n);
}

template<typename T>
void custom_implementation()
{
  using namespace Parallilos;

  // aligned memory allocation
  const size_t n = 16;
  unique_array<T> a = get_array<T>(n);
  unique_array<T> b = get_array<T>(n);
  unique_array<T> c = get_array<T>(n);

  // initialize arrays
  for (size_t k = 0; k < n; ++k) {
    a[k] = k;
    b[k] = k + n;
  }

  // SIMD-enhanced operation
  T* a_data = a.get();
  T* b_data = b.get();
  T* c_data = c.get();
  const size_t iterations = simd_properties<T>::iterations(n);
  size_t k = 0;
  for (size_t i = 0; i < iterations; ++i, k+=simd_properties<T>::size)
    simd_storea(c_data+k, simd_add(simd_loada(a_data+k), simd_loada(b_data+k)));
  for (; k < n; ++k)
    c_data[k] = a_data[k] + b_data[k];

  // display result array
  print_info(c.get(), n);
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