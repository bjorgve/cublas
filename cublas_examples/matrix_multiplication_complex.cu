#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>

#define m 6
#define n 6
#define k 6
#define lda m
#define ldb k
#define ldc m

// Function to print a matrix
void print_matrix(cuComplex *matrix, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("(%3.1f, %3.1f) ", matrix[i + j*lda].x, matrix[i + j*lda].y);
        }
        printf("\n");
    }
}

int main(void) {
    cublasHandle_t handle;
    cuComplex* a;
    cuComplex* b;
    cuComplex* c;
    a = (cuComplex *)malloc(m * k * sizeof(cuComplex));
    b = (cuComplex *)malloc(k * n * sizeof(cuComplex));
    c = (cuComplex *)malloc(m * n * sizeof(cuComplex));

    // Initialize the matrices here...
    for(int i = 0; i < m*k; i++){
        a[i].x = i + 1;
        a[i].y = i + 1;
    }

    for(int i = 0; i < k*n; i++){
        b[i].x = i + 1;
        b[i].y = i + 1;
    }

    printf("Matrix A:\n");
    print_matrix(a, m, k);

    printf("\nMatrix B:\n");
    print_matrix(b, k, n);

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
    cublasSetMatrix(m, k, sizeof(cuComplex), a, lda, d_a, lda);
    cublasSetMatrix(k, n, sizeof(cuComplex), b, ldb, d_b, ldb);

    cuComplex al = make_cuComplex(1.0f, 0.0f);
    cuComplex bet = make_cuComplex(0.0f, 0.0f);

    // Perform operation using cublas
    cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, d_a, lda, d_b, ldb, &bet, d_c, ldc);

    // Retrieve the result matrix C from the device
    cublasGetMatrix(m, n, sizeof(cuComplex), d_c, ldc, c, ldc);

    printf("\nResult Matrix C:\n");
    print_matrix(c, m, n);

    // Destroy the handle
    cublasDestroy(handle);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return EXIT_SUCCESS;
}
