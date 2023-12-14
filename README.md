# Parallilos
Parallilos is a simple and lightweight C++11 (and newer) library that abstracts away SIMD usage to facilitate generic parallelism.

---

## Supported SIMD instruction sets
### Advanced Vector Extension (AVX)
* `AVX512F`
* `AVX2`
* `FMA`
* `AVX`

### Streaming SIMD Extension (SSE)
* `SSE4.2`
* `SSE4.1`
* `SSSE3`
* `SSE3`
* `SSE2`
* `SSE` note: `SVML` is leveraged to optimize SSE integer division if compiling with ICC.

### Advanced SIMD (Neon)
* `Neon`

---

<!-- load a vector from unaligned data -->
simd_loadu(const T data[]) -> vector

<!-- load a vector from aligned data -->
simd_loada(const T data[]) -> vector

<!-- store a vector into unaligned memory -->
simd_storeu(T addr[], const typename SIMD<T>::Type& data)

<!-- store a vector into aligned memory -->
simd_storea(T addr[], const typename SIMD<T>::Type& data)

<!-- load a vector with zeros -->
simd_setzero() -> vector

<!-- load a vector with a specific value -->
simd_setval(const T value) -> vector

<!-- [a] + [b] -->
simd_add(a, b) -> V

<!-- [a] * [b] -->
simd_mul(a, b) -> V

<!-- [a] - [b] -->
simd_sub(a, b) -> V

<!-- [a] / [b] -->
simd_div(a, b) -> V

<!-- sqrt([a]) -->
simd_sqrt(a) -> V

<!-- [a] + ([b] * [c]) -->
simd_addmul(a, b, c) -> V

<!-- [a] - ([b] * [c]) -->
simd_submul(a, b, c) -> V

<!-- [a] == [b] -->
simd_eq(a, b) noexcept -> bool

<!-- [a] != [b] -->
simd_neq(a, b) noexcept -> bool

<!-- [a] > [b] -->
simd_gt(a, b) noexcept -> bool

<!-- [a] >= [b] -->
simd_gte(a, b) noexcept -> bool

<!-- [a] < [b] -->
  template<typename V> PARALLILOS_INLINE
simd_lt(a, b) noexcept -> bool

<!-- [a] <= [b] -->
  simd_lte(a, b)


---

## Supported types
* `double` 64 bit wide floating point
* `float` 32 bit wide floating point
* `int32` 32 bit wide integer
* `uint32` 32 bit wide unsigned integer

---

## Compiler support
* GCC >= 7.1
* Clang >= ?
* MSVC >= ?
* ICC >= ?
* MinGW >= ?
* Xcode >= ?






---

## SIMD-enhanced functions

* [simd_add](#simd_add)
* [simd_sub](#simd_sub)
* [simd_mul](#simd_mul)
* [simd_div](#simd_div)

---

### simd_add
```text
r = simd_add(a, b)
```

This function returns the result vector of the element-wise addition of `a` and `b`. The function is `noexcept`.

#### Example

```cpp
auto a = simd_setval(1.0f);
auto b = simd_setval(2.0f);
std::cout << simd_add(a, b); // prints "3 3 ... 3"
```

#### Logs
* `simd_add: type "T" is not SIMD-supported`

---

### simd_mul
```text
r = simd_mul(a, b)
```

This function returns the result vector of the element-wise multiplication of `a` and `b`. The function is `noexcept`.

#### Example

```cpp
auto a = simd_setval(1.0f);
auto b = simd_setval(2.0f);
std::cout << simd_mul(a, b); // prints "2 2 ... 2"
```

#### Logs
* `simd_mul: type "T" is not SIMD-supported`

---

### simd_div
```text
r = simd_div(a, b)
```

This function returns the result vector of the element-wise division of `a` by `b`. The function is `noexcept`.

#### Example

```cpp
auto a = simd_setval(1.0f);
auto b = simd_setval(2.0f);
std::cout << simd_div(a, b); // prints "0.5 0.5 ... 0.5"
```

#### Logs
* `simd_div: type "T" is not SIMD-supported`