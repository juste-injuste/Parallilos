#include <cstddef>
#include <iostream>
#include <vector>
#include <array>
#include "../include/Parallilos.hpp"

int main()
{
  using namespace Parallilos;
  using type = float;

  //*
  // stack allocated array
  const size_t n1 = 20;
  type a1[n1] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  type b1[n1] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  type c1[n1];
   add_arrays(a1, b1, c1, n1);
  for (size_t k = 0; k < n1; ++k)
    std::cout << c1[k] << ' ';
  std::cout << std::endl;
  std::cout << "SIMD instruction set: " <<  simd_properties<type>::set << '\n';
  std::cout << "SIMD passes: " <<  simd_properties<type>::iterations(n1) << '\n';
  //*/

  /*
  // dynamic array
  std::vector<type> a3 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<type> b3 = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  const size_t n2 = a3.size();
  std::vector<type> c3(n2);
   mul_arrays(a3.data(), b3.data(), c3.data(), n2);
  for (size_t k = 0; k < n2; ++k)
    std::cout << c3[k] << ' ';
  std::cout << std::endl;
  std::cout << "SIMD instruction set: " <<  simd_properties<type>::set << '\n';
  std::cout << "SIMD passes: " <<  simd_properties<type>::iterations(n2) << '\n';
  //*/

  /*
  // heap allocated array
  const size_t n3 = 10;
  type* a2 = new type[n3]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  type* b2 = new type[n3]{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  type* c2 = new type[n3];
   sub_arrays(a2, b2, c2, n3);
  for (size_t k = 0; k < n3; ++k)
    std::cout << c2[k] << ' ';
  std::cout << std::endl;
  std::cout << "SIMD instruction set: " <<  simd_properties<type>::set << '\n';
  std::cout << "SIMD passes: " <<  simd_properties<type>::iterations(n3) << '\n';
  //*/

  /*
  // fixed-size array
  const size_t n4 = 10;
  std::array<type, n4> a4 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<type, n4> b4 = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  std::array<type, n4> c4;
   div_arrays(a4.data(), b4.data(), c4.data(), n4);
  for (size_t k = 0; k < n4; ++k)
    std::cout << c4[k] << ' ';
  std::cout << std::endl;
  std::cout << "SIMD instruction set: " <<  simd_properties<type>::set << '\n';
  std::cout << "SIMD passes: " <<  simd_properties<type>::iterations(n4) << '\n';
  //*/


  using test_type = float;
  const size_t n5 = 16;
  test_type* a = allocate<test_type>(n5);
  test_type* b = allocate<test_type>(n5);
  test_type* c = allocate<test_type>(n5);
  for (size_t k = 0; k < n5; ++k) {
    a[k] = k;
    b[k] = k + n5;
  }
  size_t k = 0;
  const size_t size =  simd_properties<test_type>::size;
  const size_t iterations =  simd_properties<test_type>::iterations(n5);
  for (size_t i = 0; i < iterations; ++i, k+=size) {
    simd_storeu(c+k, simd_add( simd_loadu(a+k), simd_loadu(b+k)));
  }
  for (; k < n5; ++k) {
    c[k] = a[k] + b[k];
  }
  for (size_t k = 0; k < n5; ++k)
    std::cout << c[k] << ' ';
  std::cout << std::endl;
  std::cout << "SIMD instruction set: " <<  simd_properties<test_type>::set << '\n';
  std::cout << "SIMD passes: " << simd_properties<test_type>::iterations(n5) << '\n';
}
// -mno-avx512f -mno-avx2 -mno-avx -mno-sse4.2 -mno-sse4.1 -mno-ssse3 -mno-sse3 -mno-sse2