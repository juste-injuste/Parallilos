#include <cstddef>
#include <iostream>
#include <vector>
#include <array>
#include <memory>
#define PARALLILOS_WARNINGS
#include "include/Parallilos.hpp"
#include "experimental/Parallilos_array.hpp"
#include "ignore/Chronometro.hpp"

template<typename T>
void print_info(T* array, size_t n)
{
  for (size_t k = 0; k < n; ++k)
    std::cout << array[k] << ' ';
  std::cout << '\n';
  
  std::cout << "Instruction set:   " << Parallilos::SIMD<T>::set << '\n';
  std::cout << "Parallel passes:   " << Parallilos::SIMD<T>::parallel(n).passes << '\n';
  std::cout << "Sequential passes: " << Parallilos::SIMD<T>::sequential(n).passes << '\n';
}

template<typename T>
void stack_array()
{
  std::cout << "\nstack allocated array example:\n";

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
  std::cout << "\nvector example:\n";

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
  std::cout << "\nheap allocated array example:\n";

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
  std::cout << "\nstandard array example:\n";

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
  std::cout << "\ncustom implementation example:\n";

  using namespace Parallilos;

  // aligned memory allocation
  const size_t n = 40;
  Array<T> a = make_array<T>(n);
  Array<T> b = make_array<T>(n);
  Array<T> c = make_array<T>(n);

  // initialize arrays
  for (size_t k = 0; k < n; ++k)
  {
    a[k] = k;
    b[k] = k + n;
  }

  // SIMD-enhanced operation
  T* a_data = a.get();
  T* b_data = b.get();
  T* c_data = c.get();

  for (const size_t k : SIMD<T>::parallel(n))
  {
    simd_storea(c_data+k, simd_add(simd_loada(a_data+k), simd_loada(b_data+k)));
  }

  for (const size_t k : SIMD<T>::sequential(n))
  {
    c_data[k] = a_data[k] + b_data[k];
  }

  // display result array
  print_info(c.get(), n);
}

int main()
{
  stack_array<float>();
  heap_array<double>();
  standard_vector<double>();
  standard_array<float>();
  custom_implementation<int>();

  using namespace Parallilos;
  using T = float;
  const size_t n = 40;
  Array<T> a = make_array<T>(n);
  Array<T> b = make_array<T>(n);
  Array<T> c = make_array<T>(n);

  using namespace Chronometro;
  
loop:
  CHRONOMETRO_MEASURE(100000) add_arrays(a, b, &c, n);
  CHRONOMETRO_MEASURE(100000) add_arrays(a.get(), b.get(), c.get(), n);
  std::cin.get();
  goto loop;

  simd_loadu<float>(nullptr);
}