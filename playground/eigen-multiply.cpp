#include <iostream>
#include <Eigen/Dense>
#include <ctime> // Include for random number seeding

int main() {
    // Seed the random number generator with the current time
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Define the dimensions of the matrices
    int m = 3; // Rows of matrix A
    int k = 4; // Columns of matrix A and Rows of matrix B
    int n = 2; // Columns of matrix B

    // Create dynamic matrices A, B, and C
    Eigen::MatrixXd A(m, k);
    Eigen::MatrixXd B(k, n);
    Eigen::MatrixXd C;

    // Fill matrices A and B with random values between 0 and 1
    A = Eigen::MatrixXd::Random(m, k);
    B = Eigen::MatrixXd::Random(k, n);

    // Perform matrix multiplication
    C = A * B;

    // Display the result matrix C
    std::cout << "Matrix A:" << std::endl << A << std::endl;
    std::cout << "Matrix B:" << std::endl << B << std::endl;
    std::cout << "Result Matrix C:" << std::endl << C << std::endl;

    return 0;
}
