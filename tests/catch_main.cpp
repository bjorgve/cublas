#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "matrix_mul_cpu.h"
#include "matrix_mul_gpu.h"

TEST_CASE("Compare CPU and GPU multiplication results", "[matrix_mult]") {
    const int MATRIX_SIZE = 840;
    EigenMat matA = Eigen::MatrixXcd::Random(MATRIX_SIZE, MATRIX_SIZE);
    EigenMat matB = Eigen::MatrixXcd::Random(MATRIX_SIZE, MATRIX_SIZE);
    EigenMat matC_cpu = EigenMat(MATRIX_SIZE, MATRIX_SIZE);
    EigenMat matC_gpu = EigenMat(MATRIX_SIZE, MATRIX_SIZE);

    multiply_matrices_cpu(matA, matB, matC_cpu);
    multiply_matrices_gpu(matA, matB, matC_gpu);

    // Check that result is same for both CPU and GPU version
    REQUIRE(matC_cpu.isApprox(matC_gpu));
}
