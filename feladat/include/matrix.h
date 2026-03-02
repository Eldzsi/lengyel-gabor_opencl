#ifndef MATRIX_H
#define MATRIX_H

void generate_matrix(float* matrix, int size);

void print_matrix(float* matrix, int size);

void calculate_determinant_gauss(float* matrix, int size, double* out_mantissa, long long* out_exponent, int* out_sign);

#endif
