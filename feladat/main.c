#define CL_TARGET_OPENCL_VERSION 220


#include "matrix.h"
#include "file.h"
#include "kernel_loader.h"

#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
    #include <direct.h>
    #define mkdir(path, mode) _mkdir(path)
#else
    #include <sys/stat.h>
    #include <sys/types.h>
#endif


const int SAMPLE_SIZE = 4;


int main() {
    long det;
    int* matrix = malloc(SAMPLE_SIZE * SAMPLE_SIZE * sizeof(int));
    if (matrix == NULL) {
        printf("[ERROR] Failed to allocate memory!\n");
        return -1;
    }

    int result = mkdir("outputs/", 0777);

    clock_t start, end;

    generate_matrix(matrix, SAMPLE_SIZE);

    //int matrix[16] = {2, 8, 7, 10, 0, 3, 0, 3, 10, 5, 7, 9, 7, 2, 4, 5};
    if (SAMPLE_SIZE < 10) {
        printf("\nMatrix:\n");
        print_matrix(matrix, SAMPLE_SIZE);
    }

    int num_submatrices = 4;
    int submatrix_size = 3;
    int* submatrices = malloc(num_submatrices * submatrix_size * submatrix_size * sizeof(int));

    for (int col = 0; col < SAMPLE_SIZE; col++) {
        int index = 0;
        for (int i = 1; i < SAMPLE_SIZE; i++) {
            for (int j = 0; j < SAMPLE_SIZE; j++) {
                if (j != col) {
                    submatrices[col * 9 + index] = matrix[i * SAMPLE_SIZE + j];
                    index++;
                }
            }
        }
    }

    printf("\nGenerated submatrices:\n");
    for (int i = 0; i < num_submatrices * submatrix_size * submatrix_size; i++) {
        printf("%4d ", submatrices[i]);
        if ((i + 1) % 3 == 0) printf("\n");
        if ((i + 1) % 9 == 0) printf("\n");
    }

    cl_int err;
    
    // Get platform
    cl_uint n_platforms;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clGetPlatformIDs failed: %d\n", err);
        return 1;
    }

    // Get device
    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clGetDeviceIDs failed: %d\n", err);
        return 1;
    }

    // Create context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateContext failed: %d\n", err);
        return 1;
    }

    // Load and build kernel
    int error_code;
    const char* kernel_code = load_kernel_source("kernel/sample.cl", &error_code);
    if (error_code != 0) {
        printf("[ERROR] Kernel source loading failed.\n");
        return 1;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateProgramWithSource failed: %d\n", err);
        return 1;
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* build_log = malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        printf("Build log:\n%s\n", build_log);
        free(build_log);
        return 1;
    }
    printf("OpenCL kernel compiled successfully.\n");

    // Create buffers
    cl_mem submatrix_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_submatrices * 9 * sizeof(int), submatrices, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateBuffer (submatrix_buffer) failed: %d\n", err);
        return 1;
    }

    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_submatrices * sizeof(long), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateBuffer (result_buffer) failed: %d\n", err);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "determinant_kernel", &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateKernel failed: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &submatrix_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_buffer);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clSetKernelArg failed: %d\n", err);
        return 1;
    }

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateCommandQueueWithProperties failed: %d\n", err);
        return 1;
    }

    // Launch kernel
    size_t global_work_size = num_submatrices;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clEnqueueNDRangeKernel failed: %d\n", err);
        return 1;
    }

    int submatrix_results[4];
    err = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, sizeof(submatrix_results), submatrix_results, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clEnqueueReadBuffer failed: %d\n", err);
        return 1;
    }

    clFinish(queue);

    start = clock();
    calculate_determinant_recursive(matrix, SAMPLE_SIZE, &det);
    end = clock();
    printf("\nDet (CPU): %ld", det);

    double exe_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nExecution time: %.4f s\n", exe_time);
    write_benchmark_to_file("outputs/sequential_benchmark.txt", SAMPLE_SIZE, exe_time);

    det = 0;
    int sign = 1;
    for (int i = 0; i < 4; i++) {
        printf("\n%d: %d", i+1, submatrix_results[i]);
        det += sign * matrix[i] * submatrix_results[i];
        sign = -sign;
    }
    printf("\nDet (OpenCL): %ld\n", det);

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(submatrix_buffer);
    clReleaseMemObject(result_buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(matrix);
    free(submatrices);

    return 0;
}
