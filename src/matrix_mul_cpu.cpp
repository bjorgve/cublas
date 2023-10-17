#include "matrix_mul_cpu.h"

void multiply_matrices_cpu(EigenMat& matA, EigenMat& matB, EigenMat& matC){
    // Use Eigen's built-in operator* for matrix multiplication
    matC = matA * matB;
}
