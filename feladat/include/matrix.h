#ifndef MATRIX_H
#define MATRIX_H

#include "kernel_loader.h"

#include <CL/cl.h>

void generate_matrix(float* matrix, int size);

void print_matrix(float* matrix, int size);

void calculate_determinant_gauss(float* matrix, int size, float* out_mantissa, long long* out_exponent, int* out_sign);

void calculate_determinant_gauss_opencl(float* matrix, int size, float* out_mantissa, long long* out_exponent, int* out_sign, float* out_time_write, float* out_time_calc, float* out_time_read);

#endif
