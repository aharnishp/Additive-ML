#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

// Matrix multiplication using C-style arrays
void multiplyArrays(int** matrixA, int** matrixB, int** result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = 0;
            for (int k = 0; k < size; k++) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

// Matrix multiplication using std::vector
void multiplyVectors(vector<vector<int>>& matrixA, vector<vector<int>>& matrixB, vector<vector<int>>& result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = 0;
            for (int k = 0; k < size; k++) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

// Function to generate a random matrix
void generateMatrix(vector<vector<int>>& matrix, int size) {
    for (int i = 0; i < size; i++) {
        vector<int> row;
        for (int j = 0; j < size; j++) {
            row.push_back(rand() % 10);
        }
        matrix.push_back(row);
    }
}

int main() {
    int size = 1000; // Change the matrix size as per your requirement

    // Create C-style arrays
    int** arrayA = new int*[size];
    int** arrayB = new int*[size];
    int** arrayResult = new int*[size];
    for (int i = 0; i < size; i++) {
        arrayA[i] = new int[size];
        arrayB[i] = new int[size];
        arrayResult[i] = new int[size];
    }

    // Create std::vector matrices
    vector<vector<int>> vectorA(size, vector<int>(size));
    vector<vector<int>> vectorB(size, vector<int>(size));
    vector<vector<int>> vectorResult(size, vector<int>(size));

    // Generate random matrices
    generateMatrix(vectorA, size);
    generateMatrix(vectorB, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            arrayA[i][j] = vectorA[i][j];
            arrayB[i][j] = vectorB[i][j];
        }
    }

    // Measure time taken using C-style arrays
    auto startArray = chrono::high_resolution_clock::now();
    multiplyArrays(arrayA, arrayB, arrayResult, size);
    auto endArray = chrono::high_resolution_clock::now();
    auto durationArray = chrono::duration_cast<chrono::milliseconds>(endArray - startArray);

    // Measure time taken using std::vector
    auto startVector = chrono::high_resolution_clock::now();
    multiplyVectors(vectorA, vectorB, vectorResult, size);
    auto endVector = chrono::high_resolution_clock::now();
    auto durationVector = chrono::duration_cast<chrono::milliseconds>(endVector - startVector);

    // Print the results
    cout << "Matrix multiplication using C-style arrays took " << durationArray.count() << " ms" << endl;
    cout << "Matrix multiplication using std::vector took " << durationVector.count() << " ms" << endl;

    // Clean up the memory
    for (int i = 0; i < size; i++) {
        delete[] arrayA[i];
        delete[] arrayB[i];
        delete[] arrayResult[i];
    }
    delete[] arrayA;
    delete[] arrayB;
    delete[] arrayResult;

    return 0;
}
