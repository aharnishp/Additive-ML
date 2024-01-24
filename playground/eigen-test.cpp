#include<iostream>
#include<Eigen/Dense>

#define reserve_column 1
 

int main() {
    for(int iter = 0; iter < 100000; iter++){
        // Define the matrix size
        int rows = 3;
        int cols = 4; // Assuming 4 columns with the last column reserved for expansion

        // Create a flattened 1D row-major vector with reserved space
        Eigen::VectorXd flattenedVector(rows * cols);
        // Fill the vector with your data
        flattenedVector << 1, 2, 3, 0,  // 0 is reserved for expansion
                        4, 5, 6, 0,
                        7, 8, 9, 0;

        // // Create Eigen matrices using Map and partial views
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrixA(flattenedVector.data(), rows, cols - reserve_column);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrixB(flattenedVector.data(), rows, cols - reserve_column);

        // directly map to eigen matrix
        // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrixA(flattenedVector.data(), rows, cols);
        // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrixB(flattenedVector.data(), rows, cols);

        // print the dimension and elements of the matrix
        // printing dimensions of A
        std::cout << "The dimensions of A are " << matrixA.rows() << "x" << matrixA.cols() << std::endl;
        // printing dimensions of B
        std::cout << "The dimensions of B are " << matrixB.rows() << "x" << matrixB.cols() << std::endl;

        // printing elements of A
        std::cout << "The elements of A are:\n" << matrixA << std::endl;
        // printing elements of B
        std::cout << "The elements of B are:\n" << matrixB << std::endl;


        // Perform matrix multiplication
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result = matrixA * matrixB.transpose();

        // Display the result
        std::cout << "Result of matrix multiplication:\n" << result << std::endl;
    }
    return 0;
}