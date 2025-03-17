#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>

#define MAX_SIZE 5000

char* loadKernelFromFile(const char* filename, size_t* kernel_size);
void printMatrix(int* matrix, int size);
int randInt(int min, int max);
void generateMatrix(int* matrix, int size, int min, int max);


int main(void) {
    cl_int err;

    srand(time(NULL));

    // Get platform
    cl_uint n_platforms;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
        return 0;
    }

    // Get device
    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_GPU,
        1,
        &device_id,
        &n_devices
    );
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
        return 0;
    }

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

    // Load kernel from file
    size_t kernel_size;
    char* kernel_source = loadKernelFromFile("matrix_mult.cl", &kernel_size);
    if (!kernel_source) {
        return -1;
    }

    // Build the program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_size, &err);
    free(kernel_source);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matrix_mult", &err);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);

    //for (int matrix_size = 100; matrix_size <= 5000; matrix_size += (matrix_size == 100 ? 400 : 500)) {
    for (int matrix_size = 2; matrix_size <= 4; matrix_size++) {
        printf("\n--- Matrix size: %dx%d ---\n", matrix_size, matrix_size);

        int* A = (int*)malloc(matrix_size * matrix_size * sizeof(int));
        int* B = (int*)malloc(matrix_size * matrix_size * sizeof(int));
        int* C = (int*)malloc(matrix_size * matrix_size * sizeof(int));

        if (!A || !B || !C) {
            printf("[ERROR] Memory allocation failed for matrix size %d\n", matrix_size);
            return -1;
        }

        generateMatrix(A, matrix_size, 1, 10);
        generateMatrix(B, matrix_size, 1, 10);

        cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrix_size * matrix_size * sizeof(int), A, &err);
        cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrix_size * matrix_size * sizeof(int), B, &err);
        cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrix_size * matrix_size * sizeof(int), NULL, &err);

        // Set kernel arguments
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C);
        clSetKernelArg(kernel, 3, sizeof(int), &matrix_size);

        size_t global_work_size[2] = {matrix_size, matrix_size};

        clock_t start = clock();
        clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
        clEnqueueReadBuffer(command_queue, buffer_C, CL_TRUE, 0, matrix_size * matrix_size * sizeof(int), C, 0, NULL, NULL);
        clock_t end = clock();

        printf("Execution time: %.3f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

        if (matrix_size <= 4) {
            printf("A =\n");
            printMatrix(A, matrix_size);
            printf("B =\n");
            printMatrix(B, matrix_size);
            printf("A * B =\n");
            printMatrix(C, matrix_size);
        }

        clReleaseMemObject(buffer_A);
        clReleaseMemObject(buffer_B);
        clReleaseMemObject(buffer_C);
        free(A);
        free(B);
        free(C);
    }

    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

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


void generateMatrix(int* matrix, int size, int min, int max) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = randInt(min, max);
    }
}