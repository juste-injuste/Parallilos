#include <iostream>
#include <cstdint>

// #define PARALLILOS_LOGGING
#include "include/Parallilos.hpp"

template<typename T> inline
void print_array(const Parallilos::Array<T>& array)
{
  using namespace Parallilos;

  for (unsigned k = 0; k < array.size(); ++k)
    std::cout << array[k] << ' ';
  std::cout << '\n';
  
  std::cout << "Instruction set:          " << SIMD<T>::set  << '\n';
  std::cout << "Data per parallel passes: " << SIMD<T>::size << '\n';
  std::cout << "Parallel passes:          " << SIMD<T>::parallel(array.size()).passes   << '\n';
  std::cout << "Sequential passes:        " << SIMD<T>::sequential(array.size()).passes << '\n';
}

int main()
{
  using namespace Parallilos;
  using type = float;

  // heap array
  const unsigned n = 20;
  Array<type> a = Array<type>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Array<type> b = Array<type>({9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Array<type> c = Array<type>(n);

  // parallel processing
  for (auto k : SIMD<type>::parallel(n))
  {
    simd_storea(c+k, simd_mul(a.as_vector(k), b.as_vector(k)));
  }

  // sequential fallback
  for (auto k : SIMD<type>::sequential(n))
  {
    c[k] = a[k] * b[k];
  }
  
  // display result array
  print_array(c);

  auto v1 = simd_setval<double>(1);
  auto v2 = simd_setval<char>('2');
  auto v3 = simd_setval<int32_t>(3);
  auto v4 = simd_setval<float>(4);

  auto m1 = simd_gt(v1, v1);
  auto m2 = simd_gt(v2, v2);
  auto m3 = simd_gt(v3, v3);
  auto m4 = simd_gt(v4, v4);

  std::cout << typeid(decltype(m1)).name() << '\n';
  std::cout << typeid(decltype(m2)).name() << '\n';
  std::cout << typeid(decltype(m3)).name() << '\n';
  std::cout << typeid(decltype(m4)).name() << '\n';

  std::cout << "double: " << v1 << '\n';
  std::cout << "char:   " << v2 << '\n';
  std::cout << "int32:  " << v3 << '\n';
  std::cout << "float:  " << v4 << '\n';

  auto a1 = simd_setval(1.0f);
  auto b1 = simd_setval(2.0f);
  std::cout << simd_add(a1, b1) << '\n'; // prints " 3   3   ...  3"
  std::cout << simd_sub(a1, b1) << '\n'; // prints "-1  -1   ... -1"
  std::cout << simd_mul(a1, b1) << '\n'; // prints " 2   2   ...  2"
  std::cout << simd_div(a1, b1) << '\n'; // prints " 0.5 0.5 ...  0.5"

  auto a2 = simd_setval(2);
  auto b2 = simd_setval(-2);
  std::cout << a2 << " -> " << simd_abs(a2) << '\n';
  std::cout << b2 << " -> " << simd_abs(b2) << '\n';
}