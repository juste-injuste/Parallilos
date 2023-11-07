#include <cstddef>
#include <iostream>
#include <vector>
#include <array>
#include <memory>
#define PARALLILOS_WARNINGS
#include "include/Parallilos.hpp"
#include "ignore/Chronometro.hpp"

template<typename T>
void print_info(T* array, size_t n)
{
  using namespace Parallilos;

  for (size_t k = 0; k < n; ++k)
    std::cout << array[k] << ' ';
  std::cout << '\n';
  
  std::cout << "Instruction set:   " << SIMD<T>::set << '\n';
  std::cout << "Parallel passes:   " << SIMD<T>::parallel(n).passes << '\n';
  std::cout << "Sequential passes: " << SIMD<T>::sequential(n).passes << '\n';
}

template<typename T>
void stack_array()
{
  using namespace Parallilos;

  std::cout << "\nstack allocated array example:\n";

  // stack array
  const size_t n = 20;
  T a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  T b[] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  T c[n];

  // parallel processing
  for (const size_t k : SIMD<T>::parallel(n))
  {
    simd_storeu(c+k, simd_add(simd_loadu(a+k), simd_loadu(b+k)));
  }

  // sequential fallback
  for (const size_t k : SIMD<T>::sequential(n))
  {
    c[k] = a[k] + b[k];
  }

  // display result array
  print_info(c, n);
}

template<typename T>
void heap_array()
{
  using namespace Parallilos;

  std::cout << "\nheap allocated array example:\n";

  // heap array
  const size_t n = 20;
  Array<T> a = make_array<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Array<T> b = make_array<T>({9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Array<T> c = make_array<T>(n);

  const T* a_data = a.get();
  const T* b_data = b.get();
  T* c_data = c.get();

  // parallel processing
  for (const size_t k : SIMD<T>::parallel(n))
  {
    simd_storea(c_data+k, simd_mul(simd_loada(a_data+k), simd_loada(b_data+k)));
  }

  // sequential fallback
  for (const size_t k : SIMD<T>::sequential(n))
  {
    c_data[k] = a_data[k] * b_data[k];
  }

  // display result array
  print_info(c_data, n);
}

template<typename T>
void standard_vector()
{
  using namespace Parallilos;

  std::cout << "\nvector example:\n";

  // dynamic array
  const size_t n = 20;
  std::vector<T> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<T> b = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<T> c(n);

  const T* a_data = a.data();
  const T* b_data = b.data();
  T* c_data = c.data();

  // parallel processing
  for (const size_t k : SIMD<T>::parallel(n))
  {
    simd_storeu(c_data+k, simd_sub(simd_loadu(a_data+k), simd_loadu(b_data+k)));
  }

  // sequential fallback
  for (const size_t k : SIMD<T>::sequential(n))
  {
    c_data[k] = a_data[k] - b_data[k];
  }

  // display result array
  print_info(c_data, n);
}

template<typename T>
void standard_array()
{
  using namespace Parallilos;

  std::cout << "\nstandard array example:\n";

  // standard array
  const size_t n = 20;
  std::array<T, n> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<T, n> b = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<T, n> c;

  const T* a_data = a.data();
  const T* b_data = b.data();
  T* c_data = c.data();

  // parallel processing
  for (const size_t k : SIMD<T>::parallel(n))
  {
    simd_storeu(c_data+k, simd_div(simd_loadu(a_data+k), simd_loadu(b_data+k)));
  }

  // sequential fallback
  for (const size_t k : SIMD<T>::sequential(n))
  {
    c_data[k] = a_data[k] / b_data[k];
  }

  // display result array
  print_info(c.data(), n);
}

int main()
{
  stack_array<int>();
  heap_array<int>();
  standard_vector<int>();
  standard_array<int>();

  using namespace Parallilos;
  auto t = make_array<float>({1, 2, 4, 5, 6});
}