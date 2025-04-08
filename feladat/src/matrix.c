#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void print_matrix(int* matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", matrix[i * size + j]);
        }
        printf("\n");
    }
}


void generate_matrix(int* matrix, int size) {
    srand(time(NULL));

    for (int i = 0; i < size*size; i++) {
        matrix[i] = rand() % 11;
    }
}


void calculate_determinant(int* matrix, int size, long long* det) {
    if (size == 1) {
        *det = matrix[1];
    } else if (size == 2) {
        *det = (matrix[0] * matrix[3]) - (matrix[1] * matrix[2]);
    } else {
        *det = 0;
        int sign = 1;

        for (int col = 0; col < size; col++) {
            int* submatrix = malloc((size - 1) * (size - 1) * sizeof(int));
            if (submatrix == NULL) {
                printf("\n[ERROR] Failed to allocate memory for submatrix!");
                return;
            }

            for (int i = 1; i < size; i++) {
                for (int j = 0, sub_col = 0; j < size; j++) {
                    if (j == col) {
                        continue;
                    } else {
                        submatrix[(i - 1) * (size - 1) + sub_col] = matrix[i * size + j];
                        sub_col++;
                    }   
                }
            }

            long long subdet;
            calculate_determinant(submatrix, size - 1, &subdet);

            *det += sign * matrix[col] * subdet;

            sign = -sign;

            free(submatrix);
        }
    }
}


