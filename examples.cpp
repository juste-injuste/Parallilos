#include <cstddef>
#include <iostream>
#include <vector>
#include <array>
#include <memory>
#define PARALLILOS_WARNINGS
#define __AVX512F__
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

int main()
{
  using namespace Parallilos;
  using type = float;

  // heap array
  const size_t n = 20;
  SIMD<type>::Array a = SIMD<type>::make_array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  SIMD<type>::Array b = SIMD<type>::make_array({9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  SIMD<type>::Array c = SIMD<type>::make_array(n);

  // parallel processing
  for (const size_t k : SIMD<type>::parallel(n))
  {
    simd_storea(c.get()+k, simd_mul(simd_loada(a.get()+k), simd_loada(b.get()+k)));
  }

  // sequential fallback
  for (const size_t k : SIMD<type>::sequential(n))
  {
    c.get()[k] = a.get()[k] * b.get()[k];
  }

  // display result array
  print_info(c.get(), n);

  // simd_eq(SIMD<long long>::Type{}, SIMD<long long>::Type{});


  using T = typename SIMD<long long>::Type;
  using M = Mask<T>;

  M m;
}