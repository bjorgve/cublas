#include "matrix_mul_gpu.h"

// Function Definition
void multiply_matrices_gpu(EigenMat& matA, EigenMat& matB, EigenMat& matC){
    // Check if matrix dimensions match
    assert(matA.cols() == matB.rows());

    // Extract the matrix dimensions
    int m = matA.rows(); // Number of rows in op(A) and C.
    int n = matB.cols(); // Number of columns in op(B) and C.
    int k = matA.cols(); // Number of columns in op(A) and rows in op(B).

    // Check if the output matrix has the correct dimensions
    assert(matC.rows() == m);
    assert(matC.cols() == n);
    assert(k == matB.rows());


    // Set the leading dimensions
    // Rows for column-major ordering
    int lda = matA.rows();
    int ldb = matB.rows();
    int ldc = matC.rows();

    // Create a cuBlas handle.
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Create device pointers.
    cuDoubleComplex* d_A;
    cuDoubleComplex* d_B;
    cuDoubleComplex* d_C;


    auto matA_size = matA.rows()*matA.cols()*sizeof(cuDoubleComplex);
    cudaMalloc((void**)&d_A, matA_size);
    cudaMemcpy(d_A, matA.data(), matA_size, cudaMemcpyHostToDevice);

    auto matB_size = matB.rows()*matB.cols()*sizeof(cuDoubleComplex);
    cudaMalloc((void**)&d_B, matB_size);
    cudaMemcpy(d_B, matB.data(), matB_size, cudaMemcpyHostToDevice);

    auto matC_size = matC.rows()*matC.cols()*sizeof(cuDoubleComplex);
    cudaMalloc((void**)&d_C, matC_size);

    // Constants for cublasZgemm routine.
    const cuDoubleComplex alf = make_cuDoubleComplex(1,0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0,0);

    // Perform multiplication.
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alf, d_A, lda, d_B, ldb, &bet, d_C, ldc);

    // Copy the result back to the output Eigen matrix.
    cudaMemcpy(matC.data(), d_C, matC_size, cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy the cuBlas handle.
    cublasDestroy(handle);
}
