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


void calculate_determinant_recursive(int* matrix, int size, long* det) {
    if (size == 1) {
        *det = matrix[0];
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

            long subdet;
            calculate_determinant_recursive(submatrix, size - 1, &subdet);

            *det += sign * matrix[col] * subdet;

            sign = -sign;

            free(submatrix);
        }
    }
}


long calculate_determinant_iterative(int* matrix, int size) {
    for (int col = 0; col < size; col++) {
        int submatrix[9];
        int idx = 0;
        for (int i = 1; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (j == col) continue;
                submatrix[idx++] = matrix[i * size + j];
            }
        }

        for (int i = 0; i < 9; i++) {
            printf("%4d", submatrix[i]);
            if ((i + 1) % 3 == 0) printf("\n");
        }
        printf("\n");

        for (int subcol = 0; subcol < 3; subcol++) {
            int subsubmatrix[4];
            int subidx = 0;
            for (int i = 1; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (j == subcol) continue;
                    subsubmatrix[subidx++] = submatrix[i * (size-1) + j];
                }
            }

            for (int i = 0; i < 4; i++) {
                printf("%4d", subsubmatrix[i]);
                if ((i + 1) % 2 == 0) printf("\n");

            }
            printf("\n");
            printf("Det: %d\n\n", (subsubmatrix[0] * subsubmatrix[3]) - (subsubmatrix[1] * subsubmatrix[2]));
        }
    }

    return 0;
}


/* long calculate_determinant_iterative(int* matrix, int size) {
    // (n-1)x(n-1) almátrixok (első sor Laplace)
    printf("\n%d x %d-es almátrixok:\n", size-1, size-1);
    for (int col = 0; col < size; col++) {
        int submatrix[(size-1)*(size-1)];
        int idx = 0;
        for (int i = 1; i < size; i++) { // első sort kihagyjuk
            for (int j = 0; j < size; j++) {
                if (j == col) continue;
                submatrix[idx++] = matrix[i * size + j];
            }
        }
        // Kiírás
        for (int i = 0; i < (size-1)*(size-1); i++) {
            printf("%4d", submatrix[i]);
            if ((i + 1) % (size-1) == 0) printf("\n");
        }
        printf("\n");

        // (n-2)x(n-2) almátrixok ebből (szintén első sor Laplace)
        if (size > 2) {
            printf("%d x %d-es almátrixok ebből:\n", size-2, size-2);
            for (int subcol = 0; subcol < size-1; subcol++) {
                int subsubmatrix[(size-2)*(size-2)];
                int subidx = 0;
                for (int i = 1; i < size-1; i++) { // első sort kihagyjuk
                    for (int j = 0; j < size-1; j++) {
                        if (j == subcol) continue;
                        subsubmatrix[subidx++] = submatrix[i * (size-1) + j];
                    }
                }
                // Kiírás
                for (int i = 0; i < (size-2)*(size-2); i++) {
                    printf("%4d", subsubmatrix[i]);
                    if ((i + 1) % (size-2) == 0) printf("\n");
                }
                printf("\n");
            }
        }
    }
    return 0;
} */