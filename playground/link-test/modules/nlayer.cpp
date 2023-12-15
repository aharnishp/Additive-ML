#include "nlayer.hpp"

int myfunct(int a, int b) {
    std::cout << "myfunct(" << a << ", " << b << ") = " << a + b << std::endl;
    return a + b;
}

class test{
    public:
    int x= 0;

    int test_x(int a, int b){
        return a * b;
    }

};