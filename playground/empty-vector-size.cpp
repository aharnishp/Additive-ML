#include <iostream>
#include <vector>
#include "../modules/nlayer.cpp"


int main() {
    std::vector<nlayer> emptyVector;
    // std::vector<int> emptyVector;

    nlayer newl(5,5,1,1,0.05);
    emptyVector.push_back(newl);
    std::cout << "Size of an empty vector: " << sizeof(emptyVector) << " bytes" << std::endl;
    return 0;
}
