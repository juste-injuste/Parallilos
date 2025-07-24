#include "../include/parallilos.hpp"
#include <cstdint>

namespace stz
{
  inline namespace parallilos
  {
    namespace para
    {

    }

    namespace sequ
    {

    }
  }
}

#define impl_parallelize(...)                \
  { using namespace stz::para; __VA_ARGS__ } \
  { using namespace stz::sequ; __VA_ARGS__ }
#define parallelize(size)     impl_parallelize

int main()
{
  stz::Array<float> a(250), b(250), c(250);
  stz::SIMD<float>::Type av, bv, cv;


  parallelize(1)
  (
    cv = av + bv;
  )



}