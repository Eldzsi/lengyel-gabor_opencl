#include "matrix.h"
#include "file.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main() {
    int size = 1;
    long long det;
    int* matrix = malloc(size * size * sizeof(int));
    if (matrix == NULL) {
        printf("[ERROR] Failed to allocate memory!");
        return -1;
    }
    clock_t start, end;

    generate_matrix(matrix, size);

    if (size < 10) {
        printf("\nMatrix:\n");
        print_matrix(matrix, size);
    }

    start = clock();
    calculate_determinant(matrix, size, &det);
    end = clock();
    printf("\nDet: %d", det);

    double exe_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nExecution time: %.4f s\n", exe_time);
    write_benchmark_to_file("sequential_benchmark.txt", size, exe_time);

    free(matrix);

    return 0;
}
