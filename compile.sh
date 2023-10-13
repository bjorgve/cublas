#!/bin/sh
source saga.env
nvcc matrix_multiplication.cu -lcublas -o matrix_multiplication.x
nvcc matrix_multiplication_complex.cu -lcublas -o matrix_multiplication_complex.x
nvcc matrix_multiplication_complex_eigen.cu -lcublas -o matrix_multiplication_complex_eigen.x
