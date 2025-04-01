#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

char* loadKernelFromFile(const char* filename, size_t* kernel_size);

#endif
