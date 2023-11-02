#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "matrix_mul_cpu.h"
#include "matrix_mul_gpu.h"

TEST_CASE("Compare CPU and GPU multiplication results", "[matrix_mult]") {

    std::vector<std::tuple<int, int, int, int, bool>> matrix_dimensions = {
        {4, 2, 2, 4, true},
        {8, 4, 4, 8, true},
        {16, 8, 8, 16, true},
    };

    for (auto const& dimensions: matrix_dimensions) {

        SECTION("Dimensions: A(" + std::to_string(std::get<0>(dimensions)) + ","
                 + std::to_string(std::get<1>(dimensions)) + "), B("
                 + std::to_string(std::get<2>(dimensions)) + ","
                 + std::to_string(std::get<3>(dimensions)) + ")") {

            auto MATRIX_A_ROWS = std::get<0>(dimensions);
            auto MATRIX_A_COLUMNS = std::get<1>(dimensions);
            auto MATRIX_B_ROWS = std::get<2>(dimensions);
            auto MATRIX_B_COLUMNS = std::get<3>(dimensions);
            bool should_approx = std::get<4>(dimensions);

            auto MATRIX_C_ROWS = MATRIX_A_ROWS;
            auto MATRIX_C_COLUMNS = MATRIX_B_COLUMNS;

            EigenMat matA = Eigen::MatrixXcd::Random(MATRIX_A_ROWS, MATRIX_A_COLUMNS);
            EigenMat matB = Eigen::MatrixXcd::Random(MATRIX_B_ROWS, MATRIX_B_COLUMNS);

            EigenMat matC_cpu = EigenMat(MATRIX_C_ROWS, MATRIX_C_COLUMNS);
            EigenMat matC_gpu = EigenMat(MATRIX_C_ROWS, MATRIX_C_COLUMNS);

            try {
                multiply_matrices_cpu(matA, matB, matC_cpu);
                multiply_matrices_gpu(matA, matB, matC_gpu);
            } catch (std::invalid_argument& e) {
                REQUIRE_FALSE(should_approx);
                continue;
            }
            REQUIRE(should_approx == matC_cpu.isApprox(matC_gpu));
        }
    }
}
