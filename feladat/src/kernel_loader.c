#include "kernel_loader.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

char* loadKernelFromFile(const char* filename, size_t* kernel_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("[ERROR] Failed to open kernel file: %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);  
    *kernel_size = ftell(file);
    rewind(file);              

    char* kernel_source = (char*)malloc(*kernel_size + 1);
    fread(kernel_source, 1, *kernel_size, file);
    kernel_source[*kernel_size] = '\0';

    fclose(file);
    return kernel_source;
}