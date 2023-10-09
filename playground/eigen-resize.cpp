#include <iostream>
#include <Eigen/Dense>
#include <ctime>


int main() {
    // Seed the random number generator with the current time
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    int m = 3;
    int n = 4;
    int additionalRows = 2;
    int additionalCols = 3;

    Eigen::MatrixXd A(m,n);

    A = Eigen::MatrixXd::Random(m,n);

    std::cout << "Matrix A:" << std::endl << A << std::endl;


    A.conservativeResize(m + additionalRows, n + additionalCols);


    std::cout << "Matrix A Resized:" << std::endl << A << std::endl;


    A.block(m, 0, additionalRows, A.cols()) = Eigen::MatrixXd::Random(additionalRows, A.cols());
    A.block(0, n, A.rows(), additionalCols) = Eigen::MatrixXd::Random(A.rows(), additionalCols);
    // A.block(m, n, additionalRows, additionalCols) = Eigen::MatrixXd::Random(additionalRows, additionalCols);

    std::cout << "Matrix A Resized & Filled:" << std::endl << A << std::endl;

    return 0;
}
