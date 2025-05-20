#ifndef MATRIX_H
#define MATRIX_H

void print_matrix(int* matrix, int size);

void generate_matrix(int* matrix, int size);

void calculate_determinant_recursive(int* matrix, int size, long* det);

long calculate_determinant_iterative(int* matrix, int size);

#endif
