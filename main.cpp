#include "matrix_mul_cpu.h"
#include "matrix_mul_gpu.h"
#include <iostream>
#include <chrono>  // for high_resolution_clock

int main() {
    auto MATRIX_SIZE = 840;
    EigenMat matA = Eigen::MatrixXcd::Random(MATRIX_SIZE, MATRIX_SIZE);
    EigenMat matB = Eigen::MatrixXcd::Random(MATRIX_SIZE, MATRIX_SIZE);

    // multiply on CPU
    EigenMat matC_cpu = EigenMat(MATRIX_SIZE, MATRIX_SIZE);

    auto start_cpu = std::chrono::high_resolution_clock::now();  // Start timer
    std::cout << "[INFO] Starting CPU matrix multiplication..." << std::endl;
    multiply_matrices_cpu(matA, matB, matC_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();  // End timer
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;  // Compute duration
    std::cout << "[INFO] CPU matrix multiplication done in " << diff_cpu.count() << " s" << std::endl;

    // multiply on GPU
    EigenMat matC_gpu = EigenMat(MATRIX_SIZE, MATRIX_SIZE);

    auto start_gpu = std::chrono::high_resolution_clock::now();  // Start timer
    std::cout << "[INFO] Starting GPU matrix multiplication..." << std::endl;
    multiply_matrices_gpu(matA, matB, matC_gpu);
    auto end_gpu = std::chrono::high_resolution_clock::now();  // End timer
    std::chrono::duration<double> diff_gpu = end_gpu - start_gpu;  // Compute duration
    std::cout << "[INFO] GPU matrix multiplication done in " << diff_gpu.count() << " s" << std::endl;

    return 0;
}
