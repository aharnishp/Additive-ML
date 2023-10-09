#include <iostream>
#include <vector>
#include <cblas.h>

// Function to multiply two matrices and store the result in C
void multiplyMatrices(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int m, int n, int k) {
    // Ensure that the dimensions are compatible for matrix multiplication
    if (A.size() != (m + 3) * (k + 3) || B.size() != (k + 3) * (n + 3)) {
        std::cerr << "Matrix dimensions are not compatible for multiplication." << std::endl;
        return;
    }

    // Set the matrix order and transpose options
    CBLAS_ORDER order = CblasRowMajor;
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    // Perform matrix multiplication using the actual dimensions (m x n)
    cblas_dgemm(order, transA, transB, m, n, k, 1.0, &A[3 * (k + 3) + 3], k + 3, &B[3 * (n + 3) + 3], n + 3, 0.0, &C[0], n);

    // Note: We use the offset (3 * (k + 3) + 3) and (3 * (n + 3) + 3) to skip the extra rows and columns.
}

int main() {
    // Define the dimensions of the matrices
    int m = 3; // Rows of A
    int k = 4; // Columns of A and Rows of B
    int n = 2; // Columns of B

    // Create 1D vectors to store flattened matrices A, B, and C (result)
    int extra_rows = 3;
    int extra_cols = 3;
    std::vector<double> A((m + extra_rows) * (k + extra_cols));
    std::vector<double> B((k + extra_rows) * (n + extra_cols));
    std::vector<double> C(m * n);

    // Populate matrices A and B (you should fill in your data here)

    // Call the multiplication function
    multiplyMatrices(A, B, C, m, n, k);

    // Display the result matrix C
    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
