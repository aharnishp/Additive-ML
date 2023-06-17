// g++ conv-fast-2.cpp -o conv-fast-2 -lopenblas -fopenmp

#include <iostream>
#include <vector>
#include <cblas.h>
#include <omp.h>

// Function to perform 3D convolution using OpenBLAS and OpenMP
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

    // Transpose the filters for improved memory access
    std::vector<std::vector<std::vector<std::vector<double>>>> transposedFilters(
        numFilters, std::vector<std::vector<std::vector<double>>>(
            filterSize, std::vector<std::vector<double>>(
                filterDepth, std::vector<double>(numFilters))));

    #pragma omp parallel for
    for (int f = 0; f < numFilters; f++) {
        for (int i = 0; i < filterDepth; i++) {
            for (int j = 0; j < filterSize; j++) {
                for (int k = 0; k < numFilters; k++) {
                    transposedFilters[f][j][i][k] = filters[k][f][i][j];
                }
            }
        }
    }

    #pragma omp parallel for
    for (int f = 0; f < numFilters; f++) {
        for (int i = 0; i < outputDepth; i++) {
            for (int j = 0; j < outputSize; j++) {
                const double* filterData = transposedFilters[f][0][0].data();
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
    // Define the filters and matrix
    std::vector<std::vector<std::vector<std::vector<double>>>> filters = {
        { { {1, 0, 1}, {0, 1, 0}, {1, 0, 1} },
          { {0, 1, 0}, {1, 0, 1}, {0, 1, 0} } },

        { { {1, 1, 1}, {0, 0, 0}, {1, 1, 1} },
          { {0, 0, 0}, {1, 1, 1}, {0, 0, 0} } }
    };

    std::vector<std::vector<std::vector<double>>> matrix = {
        { {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3} },
        { {4, 4, 4, 4}, {5, 5, 5, 5}, {6, 6, 6, 6} },
        { {7, 7, 7, 7}, {8, 8, 8, 8}, {9, 9, 9, 9} }
    };

    // Set the number of OpenMP threads
    omp_set_num_threads(omp_get_max_threads());

    // Perform 3D convolution
    std::vector<std::vector<std::vector<double>>> result = convolve3D(filters, matrix);

    // Print the resulting convolutions
    for (int f = 0; f < result.size(); f++) {
        std::cout << "Filter " << f << " convolution:\n";
        for (int i = 0; i < result[f].size(); i++) {
            for (int j = 0; j < result[f][i].size(); j++) {
                std::cout << result[f][i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}
