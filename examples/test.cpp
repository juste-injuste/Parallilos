#include <cstddef>
#include <iostream>
#include <vector>
#include <array>
#include "../include/Parallilos.hpp"

int main()
{
  using type = int;
  
  // stack allocated array
  const size_t n1 = 10;
  type a1[n1] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  type b1[n1] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  type c1[n1];

  // heap allocated array
  const size_t n2 = 10;
  type* a2 = new type[n2]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  type* b2 = new type[n2]{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  type* c2 = new type[n2];

  // dynamic array
  std::vector<type> a3 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<type> b3 = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  const size_t n3 = a3.size();
  std::vector<type> c3(n3);

  // fixed-size array
  const size_t n4 = 20;
  std::array<type, n4> a4 = {0, 1, -2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<type, n4> b4 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  std::array<type, n4> c4;

  Parallilos::add_arrays(a1, b1, c1, n1);
  Parallilos::sub_arrays(a2, b2, c2, n2);
  Parallilos::mul_arrays(a3.data(), b3.data(), c3.data(), n3);
  Parallilos::div_arrays(a4.data(), b4.data(), c4.data(), n4);

  for (size_t k = 0; k < n1; ++k)
    std::cout << c1[k] << ' ';
  std::cout << std::endl;

  for (size_t k = 0; k < n2; ++k)
    std::cout << c2[k] << ' ';
  std::cout << std::endl;

  for (size_t k = 0; k < n3; ++k)
    std::cout << c3[k] << ' ';
  std::cout << std::endl;

  for (size_t k = 0; k < n4; ++k)
    std::cout << c4[k] << ' ';
  std::cout << std::endl;
  
  std::cout << "SIMD instruction set: " << PARALLILOS_EXTENDED_INSTRUCTION_SET << '\n';
  #ifdef PARALLILOS_USE_PARALLELISM
  const int n = 10;
  std::cout << "SIMD passes: " << Parallilos::simd_properties<type>::iterations(n) << '\n';
  #endif
}
// -mno-avx512f -mno-avx2 -mno-avx -mno-sse4.2 -mno-sse4.1 -mno-ssse3 -mno-sse3 -mno-sse2