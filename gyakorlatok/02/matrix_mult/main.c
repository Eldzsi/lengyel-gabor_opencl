#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>

#define MAX_SIZE 10000

void printMatrix(int* matrix, int size);
int randInt(int min, int max);
void generateMatrix(int* matrix, int size, int min, int max);
void matrixMultiplySequential(int* A, int* B, int* C, int n);
void writeBenchmarkToFile(const char* filename, int matrix_size, double exec_time);


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
    char* kernel_source = loadKernelFromFile("kernels/matrix_mult.cl", &kernel_size);
    if (!kernel_source) {
        return -1;
    }

    // Build the program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_size, &err);
    free(kernel_source);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("[ERROR] Kernel build failed:\n%s\n", log);
        free(log);
        return -1;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_mult", &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateKernel failed with error code: %d\n", err);
        return -1;
    }

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);

    int testSizes[] = {500, 750, 1000, 1250, 1500, 1750, 2000};
    int numTests = sizeof(testSizes) / sizeof(testSizes[0]);

    FILE* file = fopen("sequential_benchmark.txt", "w");
    if (file) fclose(file);
    file = fopen("parallel_benchmark.txt", "w");
    if (file) fclose(file);

    for (int i = 0; i < numTests; i++) {
    //for (int i = numTests-1; i >= 0; i--) {

        int matrix_size = testSizes[i];
        printf("\n--- Matrix size: %dx%d ---\n", matrix_size, matrix_size);

        int* A = (int*)malloc(matrix_size * matrix_size * sizeof(int));
        int* B = (int*)malloc(matrix_size * matrix_size * sizeof(int));
        int* C = (int*)malloc(matrix_size * matrix_size * sizeof(int));

        if (!A || !B || !C) {
            printf("[ERROR] Memory allocation failed.\n");
            free(A);
            free(B);
            free(C);
            continue;
        }

        generateMatrix(A, matrix_size, 1, 10);
        generateMatrix(B, matrix_size, 1, 10);

        cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrix_size * matrix_size * sizeof(int), A, &err);
        cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrix_size * matrix_size * sizeof(int), B, &err);
        cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrix_size * matrix_size * sizeof(int), NULL, &err);

        if (err != CL_SUCCESS) {
            printf("[ERROR] clCreateBuffer failed: %d\n", err);
            clReleaseMemObject(buffer_A);
            clReleaseMemObject(buffer_B);
            clReleaseMemObject(buffer_C);
            free(A);
            free(B);
            free(C);
            continue;
        }

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C);
        clSetKernelArg(kernel, 3, sizeof(int), &matrix_size);

        size_t global_work_size[2] = {matrix_size, matrix_size};

        clock_t start, end;
        
        start = clock();
        matrixMultiplySequential(A, B, C, matrix_size);
        end = clock();
        double seq_time = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Sequential: %.3f seconds\n", seq_time);
        writeBenchmarkToFile("sequential_benchmark.txt", matrix_size, seq_time);

        start = clock();
        err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            printf("[ERROR] clEnqueueNDRangeKernel failed: %d\n", err);
        }
        clEnqueueReadBuffer(command_queue, buffer_C, CL_TRUE, 0, matrix_size * matrix_size * sizeof(int), C, 0, NULL, NULL);
        end = clock();
        double par_time = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Parallel: %.3f seconds\n", par_time);
        writeBenchmarkToFile("parallel_benchmark.txt", matrix_size, par_time);


        if (matrix_size <= 10000) {

            printf("%d\n", C[10 * matrix_size + 10]);
            /*
            printf("A =\n");
            printMatrix(A, matrix_size);
            printf("B =\n");
            printMatrix(B, matrix_size);
            printf("A * B =\n");
            printMatrix(C, matrix_size);
            */
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


void matrixMultiplySequential(int* A, int* B, int* C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}


void writeBenchmarkToFile(const char* filename, int matrix_size, double exec_time) {
    FILE* file = fopen(filename, "a");
    if (!file) {
        printf("[ERROR] Failed to open result file: %s\n", filename);
        return;
    }

    fprintf(file, "%dx%d, %.6f\n", matrix_size, matrix_size, exec_time);
    fclose(file);
}