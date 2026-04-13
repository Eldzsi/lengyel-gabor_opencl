#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

char* load_kernel_source(const char* const path, int* error_code);

#endif
