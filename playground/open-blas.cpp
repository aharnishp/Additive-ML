// g++ open-blas.cpp -o open-blas -lopenblas

#include <iostream>
#include <vector>
#include <chrono>
#include <cblas.h>

using namespace std;

// Matrix multiplication using BLAS
void multiplyBLAS(vector<double>& matrixA, vector<double>& matrixB, vector<double>& result, int size) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1.0, matrixA.data(), size, matrixB.data(), size, 0.0, result.data(), size);
}

// Function to generate a random matrix
void generateMatrix(vector<double>& matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix.push_back(rand() % 10);
    }
}

int main() {
    int size = 1000; // Change the matrix size as per your requirement

    // Create std::vector matrices
    vector<double> vectorA;
    vector<double> vectorB;
    vector<double> vectorResult(size * size);

    // Generate random matrices
    generateMatrix(vectorA, size);
    generateMatrix(vectorB, size);

    // Measure time taken using BLAS
    auto startBLAS = chrono::high_resolution_clock::now();
    multiplyBLAS(vectorA, vectorB, vectorResult, size);
    auto endBLAS = chrono::high_resolution_clock::now();
    auto durationBLAS = chrono::duration_cast<chrono::milliseconds>(endBLAS - startBLAS);

    cout << "Matrix multiplication using BLAS took " << durationBLAS.count() << " ms" << endl;

    return 0;
}
