#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <Eigen/Dense>
#include <vector>

#define m 6
#define n 6
#define k 6
#define lda m
#define ldb k
#define ldc m

// Function to print a matrix
void print_matrix(Eigen::MatrixXcf matrix) {
    for(int i = 0; i < matrix.rows(); i++) {
        for(int j = 0; j < matrix.cols(); j++) {
            printf("(%3.1f, %3.1f) ", matrix(i, j).real(), matrix(i, j).imag());
        }
        printf("\n");
    }
}

int main(void) {
    cublasHandle_t handle;
    Eigen::MatrixXcf a = Eigen::MatrixXcf::Random(m, k);
    Eigen::MatrixXcf b = Eigen::MatrixXcf::Random(k, n);
    Eigen::MatrixXcf c = Eigen::MatrixXcf::Zero(m, n);

    printf("Matrix A:\n");
    print_matrix(a);

    printf("\nMatrix B:\n");
    print_matrix(b);

    // Convert Eigen matrices to std::vector<cuComplex>
    std::vector<cuComplex> a_data(a.size());
    std::vector<cuComplex> b_data(b.size());
    std::vector<cuComplex> c_data(c.size());
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            a_data[j * a.rows() + i] = make_cuComplex(a(i, j).real(), a(i, j).imag());
            b_data[j * b.rows() + i] = make_cuComplex(b(i, j).real(), b(i, j).imag());
        }
    }

    // Allocate device memory
    cuComplex* d_a;
    cudaMalloc((void**)&d_a, m*k*sizeof(cuComplex));
    cuComplex* d_b;
    cudaMalloc((void**)&d_b, k*n*sizeof(cuComplex));
    cuComplex* d_c;
    cudaMalloc((void**)&d_c, m*n*sizeof(cuComplex));

    // Initialize the CUBLAS library
    cublasCreate(&handle);

    // Copy matrices to the device
    cublasSetMatrix(m, k, sizeof(cuComplex), a_data.data(), lda, d_a, lda);
    cublasSetMatrix(k, n, sizeof(cuComplex), b_data.data(), ldb, d_b, ldb);

    cuComplex al = make_cuComplex(1.0f, 0.0f);
    cuComplex bet = make_cuComplex(0.0f, 0.0f);

    // Perform operation using cublas
    cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, d_a, lda, d_b, ldb, &bet, d_c, ldc);

    // Retrieve the result matrix C from the device
    cublasGetMatrix(m, n, sizeof(cuComplex), d_c, ldc, c_data.data(), ldc);

    // Convert result back to Eigen matrix
    for (int i = 0; i < c.rows(); ++i) {
        for (int j = 0; j < c.cols(); ++j) {
            c(i, j) = std::complex<float>(c_data[j * c.rows() + i].x, c_data[j * c.rows() + i].y);
        }
    }

    printf("\nResult Matrix C:\n");
    print_matrix(c);

    // Destroy the handle
    cublasDestroy(handle);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return EXIT_SUCCESS;
}
