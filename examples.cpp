#include <cstddef>
#include <iostream>
#include <vector>
#include <array>
#include <memory>

#define PARALLILOS_WARNINGS
#include "include/Parallilos.hpp"

template<typename T>
void print_info(T* array, size_t n)
{
  using namespace Parallilos;

  for (size_t k = 0; k < n; ++k)
    std::cout << array[k] << ' ';
  std::cout << '\n';
  
  std::cout << "Instruction set:          " << SIMD<T>::set  << '\n';
  std::cout << "Data per parallel passes: " << SIMD<T>::size << '\n';
  std::cout << "Parallel passes:          " << SIMD<T>::parallel(n).passes   << '\n';
  std::cout << "Sequential passes:        " << SIMD<T>::sequential(n).passes << '\n';
}

int main()
{
  using namespace Parallilos;
  using type = float;

  // heap array
  const size_t n = 20;
  Array<type> a = Array<type>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Array<type> b = Array<type>({9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Array<type> c = Array<type>(n);

  // parallel processing
  for (auto k : SIMD<type>::parallel(n))
  {
    simd_storea(c+k, simd_mul(a.as_vector(k), b.as_vector(k)));
    // simd_storea(c.get()+k, simd_mul(simd_loada(a.get()+k), simd_loada(b.get()+k)));
  }

  // sequential fallback
  for (auto k : SIMD<type>::sequential(n))
  {
    c[k] = a[k] * b[k];
  }

  // display result array
  print_info(c.data(), n);

  SIMD<double>::Type v1{};
  SIMD<char>::Type v2{};
  SIMD<int32_t>::Type v3{};
  SIMD<float>::Type v4{};

  auto m1 = simd_gt(v1, v1);
  auto m2 = simd_gt(v2, v2);
  auto m3 = simd_gt(v3, v3);
  auto m4 = simd_gt(v4, v4);

  std::cout << typeid(decltype(m1)).name() << '\n';
  std::cout << typeid(decltype(m2)).name() << '\n';
  std::cout << typeid(decltype(m3)).name() << '\n';
  std::cout << typeid(decltype(m4)).name() << '\n';
}