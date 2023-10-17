#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define m 6
#define n 6
#define k 6
#define lda m
#define ldb k
#define ldc m

// Function to print a matrix
void print_matrix(float *matrix, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%4.4f ", matrix[IDX2C(i,j,cols)]);
        }
        printf("\n");
    }
}

int main(void) {
    cublasHandle_t handle;
    int j;
    cudaError_t cudaStat;
    cublasStatus_t stat;
    float* a;
    float* b;
    float* c;
    a = (float *)malloc(m * k * sizeof(float));
    b = (float *)malloc(k * n * sizeof(float));
    c = (float *)malloc(m * n * sizeof(float));

    // Define an mxk (m=6, k=6) matrix A
    int ind = 11;
    for(j = 0; j < m*k; j++){
        a[j] = (float)ind++;
    }

    // Define a kxn (k=6, n=6) matrix B
    ind = 11;
    for(j = 0; j < k*n; j++){
        b[j] = (float)ind++;
    }

    printf("Matrix A:\n");
    print_matrix(a, m, k);

    printf("\nMatrix B:\n");
    print_matrix(b, k, n);

    // Allocate device memory
    float* d_a;
    cudaMalloc((void**)&d_a, m*k*sizeof(float));
    float* d_b;
    cudaMalloc((void**)&d_b, k*n*sizeof(float));
    float* d_c;
    cudaMalloc((void**)&d_c, m*n*sizeof(float));

    // Initialize the CUBLAS library
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    // Copy matrices to the device
    cublasSetMatrix(m, k, sizeof(float), a, lda, d_a, lda);
    cublasSetMatrix(k, n, sizeof(float), b, ldb, d_b, ldb);

    float al = 1.0f;
    float bet = 0.0f;

    // Perform operation using cublas
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, d_a, lda, d_b, ldb, &bet, d_c, ldc);

    // Retrieve the result matrix C from the device
    cublasGetMatrix(m, n, sizeof(float), d_c, ldc, c, ldc);

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
