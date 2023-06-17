#include <iostream>
#include <vector>
#include <chrono>
#include <cblas.h>

using namespace std;

// Matrix multiplication using OpenBLAS with 3D vectors
void multiplyOpenBLAS(vector<vector<vector<double>>>& matrixA, vector<vector<vector<double>>>& matrixB, vector<vector<vector<double>>>& result, int size) {
    const int m = size;
    const int n = size;
    const int k = size;
    const double alpha = 1.0;
    const double beta = 0.0;
    const int lda = size;
    const int ldb = size;
    const int ldc = size;
    const CBLAS_LAYOUT layout = CblasRowMajor;
    const CBLAS_TRANSPOSE transA = CblasNoTrans;
    const CBLAS_TRANSPOSE transB = CblasNoTrans;

    // Flatten the 3D vectors to 1D arrays
    vector<double> arrayA(size * size * size);
    vector<double> arrayB(size * size * size);
    vector<double> arrayResult(size * size * size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                arrayA[i * size * size + j * size + k] = matrixA[i][j][k];
                arrayB[i * size * size + j * size + k] = matrixB[i][j][k];
            }
        }
    }

    cblas_dgemm(CblasRowMajor, transA, transB, m, n, k, alpha, arrayA.data(), lda, arrayB.data(), ldb, beta, arrayResult.data(), ldc);

    // Convert the flattened 1D array back to a 3D vector
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                result[i][j][k] = arrayResult[i * size * size + j * size + k];
            }
        }
    }
}

// Function to generate random 3D vectors
void generate3DVectors(vector<vector<vector<double>>>& matrix, int size) {
    for (int i = 0; i < size; i++) {
        vector<vector<double>> subMatrix;
        for (int j = 0; j < size; j++) {
            vector<double> row;
            for (int k = 0; k < size; k++) {
                row.push_back(rand() % 10);
            }
            subMatrix.push_back(row);
        }
        matrix.push_back(subMatrix);
    }
}

int main() {
    int size = 600; // Change the matrix size as per your requirement

    // Create 3D vectors
    vector<vector<vector<double>>> matrixA;
    vector<vector<vector<double>>> matrixB;
    vector<vector<vector<double>>> matrixResult(size, vector<vector<double>>(size, vector<double>(size)));

    // Generate random 3D vectors
    generate3DVectors(matrixA, size);
    generate3DVectors(matrixB, size);

    // Measure time taken using OpenBLAS
    auto startOpenBLAS = chrono::high_resolution_clock::now();
    multiplyOpenBLAS(matrixA, matrixB, matrixResult, size);
    auto endOpenBLAS = chrono::high_resolution_clock::now();
    auto durationOpenBLAS = chrono::duration_cast<chrono::milliseconds>(endOpenBLAS - startOpenBLAS);

    cout << "Matrix multiplication using OpenBLAS took " << durationOpenBLAS.count() << " ms" << endl;

    return 0;
}
