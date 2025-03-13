#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>

const char* kernel_code =
    "__kernel void matrix_mult(__global int* A, __global int* B, __global int* C, int n) {\n"
    "   int row = get_global_id(0);\n"
    "   int col = get_global_id(1);\n"
    "   if (row < n && col < n) {\n"
    "       int sum = 0;\n"
    "       for (int k = 0; k < n; k++) {\n"
    "           sum += A[row * n + k] * B[k * n + col];\n"
    "       }\n"
    "       C[row * n + col] = sum;\n"
    "   }\n"
    "}\n";

#define MATRIX_SIZE 3

char* loadKernelFromFile(const char* filename, size_t* kernel_size);
void printMatrix(int* matrix, int size);
int randInt(int min, int max);


int main(void) {
    cl_int err;
    const int matrix_size = MATRIX_SIZE;

    int A[MATRIX_SIZE*MATRIX_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[MATRIX_SIZE*MATRIX_SIZE] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    int C[MATRIX_SIZE*MATRIX_SIZE];

    srand(time(NULL));

    int rand = randInt(1, 10);
    printf("Random: %d\n", rand);
    rand = randInt(1, 10);

    // OpenCL inicializálás
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint n_platforms, n_devices;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);

    // OpenCL kontextus létrehozása
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

    // Kernel fájlból betöltése
    size_t kernel_size;
    char* kernel_source = loadKernelFromFile("matrix_mult.cl", &kernel_size);
    if (!kernel_source) {
        return -1;
    }

    // Program és kernel létrehozása
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_size, &err);
    free(kernel_source);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matrix_mult", &err);

    // Create the device buffers
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(A), A, &err);
    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(B), B, &err);
    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(C), NULL, &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C);
    clSetKernelArg(kernel, 3, sizeof(int), &matrix_size);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);

    // Apply the kernel on the range
    size_t global_work_size[2] = {MATRIX_SIZE, MATRIX_SIZE};
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(command_queue, buffer_C, CL_TRUE, 0, sizeof(C), C, 0, NULL, NULL);

    printf("A =\n");
    printMatrix(A, MATRIX_SIZE);

    printf("B =\n");
    printMatrix(B, MATRIX_SIZE);

    printf("A * B =\n");
    printMatrix(C, MATRIX_SIZE);

    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C);

    return 0;
}


char* loadKernelFromFile(const char* filename, size_t* kernel_size) {
    FILE* file = fopen(filename, "r");
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


void printMatrix(int* matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%4d ", matrix[i * size + j]);
        }
        printf("\n");
    }
}


int randInt(int min, int max) {
    return min + rand() % (max - min + 1);
}

