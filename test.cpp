#include <iostream>
#include <cstdint>
#include <immintrin.h>

std::ostream& operator<<(std::ostream& ostream, const __m128i vector) noexcept
{
    for (unsigned k = 0; k < 4; ++k)
    {
        if (k != 0) ostream << ' ';
        ostream << reinterpret_cast<const int32_t*>(&vector)[k];
    }
    return ostream;
}


int32_t abs(int32_t a)
{
    auto s = (a >> 31);
    std::cout << "mask: " << s << '\n';
    return (a ^ s) + (s & 1);
}

__m128i abs(__m128i a)
{
    auto s   = _mm_srai_epi32 (a, 31);
    std::cout << "mask: " << s << '\n';
    auto lhs = _mm_xor_si128(a, s);
    auto rhs = _mm_and_si128(s, _mm_set1_epi32(1));
    return _mm_add_epi32(lhs, rhs);
}

int32_t main()
{
    int32_t a = 2, b = -2;
    __m128i c = _mm_set1_epi32(2), d = _mm_set1_epi32(-2);

    std::cout << a << ' ' << abs(a) << '\n';
    std::cout << b << ' ' << abs(b) << '\n';

    std::cout << c << ' ' << abs(c) << '\n';
    std::cout << d << ' ' << abs(d) << '\n';
}
