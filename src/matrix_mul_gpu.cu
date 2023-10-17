#include "matrix_mul_gpu.h"

// Function Definition
void multiply_matrices_gpu(EigenMat& matA, EigenMat& matB, EigenMat& matC){
    // Check if matrix dimensions match
    assert(matA.cols() == matB.rows());

    // Extract the matrix dimensions
    int ar = matA.rows();    // number of rows in the A matrix
    int ac = matA.cols();    // number of columns in the A matrix (rows in B)
    int bc = matB.cols();    // number of columns in the B matrix

    // Create a cuBlas handle.
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Create device pointers.
    cuDoubleComplex* d_A;
    cuDoubleComplex* d_B;
    cuDoubleComplex* d_C;

    cudaMalloc((void**)&d_A, ar*ac*sizeof(cuDoubleComplex));
    cudaMemcpy(d_A, matA.data(), ar*ac*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_B, ac*bc*sizeof(cuDoubleComplex));
    cudaMemcpy(d_B, matB.data(), ac*bc*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_C, ar*bc*sizeof(cuDoubleComplex));

    // Constants for cublasZgemm routine.
    const cuDoubleComplex alf = make_cuDoubleComplex(1,0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0,0);

    // Perform multiplication.
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ar, bc, ac, &alf, d_A, ar, d_B, ac, &bet, d_C, ar);

    // Copy the result back to the output Eigen matrix.
    cudaMemcpy(matC.data(), d_C, ar*bc*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy the cuBlas handle.
    cublasDestroy(handle);
}
