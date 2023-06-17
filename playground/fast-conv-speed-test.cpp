// g++ 

#include <iostream>
#include <vector>
#include <cblas.h>
#include <chrono>

// Function to perform 3D convolution using OpenBLAS
std::vector<std::vector<std::vector<double>>> convolve3D(
    const std::vector<std::vector<std::vector<std::vector<double>>>>& filters,
    const std::vector<std::vector<std::vector<double>>>& matrix)
{
    int numFilters = filters.size();
    int filterDepth = filters[0][0].size();
    int filterSize = filters[0][0][0].size();
    int matrixDepth = matrix[0].size();
    int matrixSize = matrix[0][0].size();

    int outputDepth = matrixDepth - filterDepth + 1;
    int outputSize = matrixSize - filterSize + 1;

    std::vector<std::vector<std::vector<double>>> output(
        numFilters, std::vector<std::vector<double>>(
            outputDepth, std::vector<double>(outputSize)));

    #pragma omp parallel for
    for (int f = 0; f < numFilters; f++) {
        for (int i = 0; i < outputDepth; i++) {
            for (int j = 0; j < outputSize; j++) {
                const double* filterData = filters[f][0][0].data();
                const double* matrixData = matrix[0][i].data() + j;

                double result = cblas_ddot(
                    numFilters * filterDepth * filterSize,
                    filterData, 1, matrixData, matrixSize);

                output[f][i][j] = result;
            }
        }
    }

    return output;
}

int main() {
    // Define the dimensions of the large matrix and filters
    const int matrixDepth = 1000;
    const int matrixSize = 1000;
    const int filterDepth = 3;
    const int filterSize = 3;
    const int numFilters = 10;

    // Generate a large random matrix
    std::vector<std::vector<std::vector<double>>> matrix(
        numFilters, std::vector<std::vector<double>>(
            matrixDepth, std::vector<double>(matrixSize)));

    // Generate random filters
    std::vector<std::vector<std::vector<std::vector<double>>>> filters(
        numFilters, std::vector<std::vector<std::vector<double>>>(
            filterDepth, std::vector<std::vector<double>>(
                filterSize, std::vector<double>(filterSize))));

    // Start the timer
    auto start = std::chrono::steady_clock::now();

    // Perform the 3D convolution
    std::vector<std::vector<std::vector<double>>> result = convolve3D(filters, matrix);

    // Stop the timer
    auto end = std::chrono::steady_clock::now();

    // Calculate the elapsed time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print the elapsed time
    std::cout << "Time taken: " << duration.count() << " ms\n";

    return 0;
}
