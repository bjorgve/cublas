#include <Eigen/Dense>
#include <cuComplex.h>
#include <cublas_v2.h>

// Create alias for complex Eigen matrix.
typedef Eigen::MatrixXcd EigenMat;

// Function Prototype
void multiply_matrices_gpu(EigenMat& matA, EigenMat& matB, EigenMat& matC);
