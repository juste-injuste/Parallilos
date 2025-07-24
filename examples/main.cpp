#include <iostream> // for std::cout
#include <cstdint>  // for fixed-size integers
#include <typeinfo> // for typeid

// #define PPZ_DEBUGGING
#include "parallilos.hpp"
#include "Chronometro.hpp"

template<typename T> inline
void print_array(const stz::Array<T>& array)
{
  for (unsigned k = 0; k < array.size(); ++k)
  {
    std::cout << array[k] << ' ';
  }
  std::cout << '\n';
  
  std::cout << "Instruction set:          " << stz::SIMD<T>::set  << '\n';
  std::cout << "Data per parallel passes: " << stz::SIMD<T>::size << '\n';
  std::cout << "Parallel passes:          " << stz::SIMD<T>::parallel(array.size()).passes   << '\n';
  std::cout << "Sequential passes:        " << stz::SIMD<T>::sequential(array.size()).passes << '\n';
}

int main()
{
  using type = float;

  // heap array
  const unsigned n = 20;
  auto a = stz::Array<type>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto b = stz::Array<type>({9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto c = stz::Array<type>(n);
  
  CHZ_MEASURE(100)
  CHZ_LOOP_FOR(1000000)
  {
    for (const size_t k : stz::SIMD<type>::parallel(n))
    {
      stz::simd_storea(c.data(k), stz::simd_mul(a.as_vector(k), b.as_vector(k)));
    }

    for (const size_t k : stz::SIMD<type>::sequential(n))
    {
      c[k] = a[k] * b[k];
    }
  }
  
  CHZ_MEASURE(100)
  CHZ_LOOP_FOR(1000000)
  {
    for (size_t k = 0; k < n; ++k)
    {
      c[k] = a[k] * b[k];
    }
  }

  stz::simd_lt(a.as_vector(0), b.as_vector(0));



  // std::cout << "Instruction set: " << stz::SIMD<type>::set  << '\n';
  // // display result array
  // print_array(c);


  // auto v1 = stz::simd_setval<double>(1);
  // auto v2 = stz::simd_setval<char>('2');
  // auto v3 = stz::simd_setval<int32_t>(3);
  // auto v4 = stz::simd_setval<float>(4);

  // auto m1 = stz::simd_gt(v1, v1);
  // auto m2 = stz::simd_gt(v2, v2);
  // auto m3 = stz::simd_gt(v3, v3);
  // auto m4 = stz::simd_gt(v4, v4);

  // std::cout << typeid(decltype(m1)).name() << '\n';
  // std::cout << typeid(decltype(m2)).name() << '\n';
  // std::cout << typeid(decltype(m3)).name() << '\n';
  // std::cout << typeid(decltype(m4)).name() << '\n';

  // std::cout << "double: " << v1 << '\n';
  // std::cout << "char:   " << v2 << '\n';
  // std::cout << "int32:  " << v3 << '\n';
  // std::cout << "float:  " << v4 << '\n';

  // auto a1 = stz::simd_setval(1.0f);
  // auto b1 = stz::simd_setval(2.0f);
  // std::cout << stz::simd_add(a1, b1) << '\n'; // prints " 3   3   ...  3"
  // std::cout << stz::simd_sub(a1, b1) << '\n'; // prints "-1  -1   ... -1"
  // std::cout << stz::simd_mul(a1, b1) << '\n'; // prints " 2   2   ...  2"
  // std::cout << stz::simd_div(a1, b1) << '\n'; // prints " 0.5 0.5 ...  0.5"

  // auto a2 = stz::simd_setval(2);
  // auto b2 = stz::simd_setval(-2);
  // std::cout << a2 << " -> " << stz::simd_abs(a2) << '\n';
  // std::cout << b2 << " -> " << stz::simd_abs(b2) << '\n';
}