#include "file.h"

#include <stdio.h>


void write_benchmark_to_file(const char* file_name, int matrix_size, double exec_time) {
    FILE* file = fopen(file_name, "a");
    if (!file) {
        printf("\n[ERROR] Failed to open file: %s", file_name);
        return;
    }

    fprintf(file, "%dx%d, %.6f\n", matrix_size, matrix_size, exec_time);
    
    fclose(file);
}
