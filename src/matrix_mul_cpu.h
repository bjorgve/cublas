#include <Eigen/Dense>

// Create alias for complex Eigen matrix.
typedef Eigen::MatrixXcd EigenMat;

// Function Prototype

void multiply_matrices_cpu(EigenMat& matA, EigenMat& matB, EigenMat& matC);
