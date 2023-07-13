#include <iostream>
#include <type_traits>

template <typename T, bool = std::is_same<T, double>::value>
struct MyStruct;

template <typename T>
struct MyStruct<T, true> {
    MyStruct() {
        static_assert(sizeof(T) != sizeof(T), "Unsupported type: double");
    }
};

template <typename T>
struct MyStruct<T, false> {
    MyStruct() {
        std::cout << "Doing something with the supported type" << std::endl;
    }
};

int main() {
    MyStruct<int> s1;  // Compiles and executes
    //MyStruct<double> s2;  // Triggers static_assert and produces the error message

    return 0;
}
