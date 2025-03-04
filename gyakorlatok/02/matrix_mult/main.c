#include <stdio.h>
#include <stdlib.h>
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


void printMatrix(int* matrix, int size);


int main(void) {
    int i;
    cl_int err;

    int A[MATRIX_SIZE*MATRIX_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[MATRIX_SIZE*MATRIX_SIZE] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    int C[MATRIX_SIZE*MATRIX_SIZE];


    // Get platform
    cl_uint n_platforms;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
        return -1;
    }

    // Get device
    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
        return -1;
    }

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("[ERROR] Failed to create OpenCL context. Error code: %d\n", err);
        return -1;
    }

    // Build the program
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, &err);
    if (!program) {
        printf("[ERROR] Failed to create program. Error code: %d\n", err);
        return -1;
    }
    
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Build error! Code: %d\n", err);
        return -1;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_mult", &err);
    if (!kernel) {
        printf("[ERROR] Failed to create kernel. Error code: %d\n", err);
        return -1;
    }

    // Create the device buffers
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(A), A, &err);
    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(B), B, &err);
    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(C), NULL, &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_C);
    const int matrix_size = MATRIX_SIZE;
    clSetKernelArg(kernel, 3, sizeof(int), &matrix_size);
    

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Size specification
    size_t global_work_size[2] = {MATRIX_SIZE, MATRIX_SIZE};

    // Apply the kernel on the range
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


void printMatrix(int* matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%4d ", matrix[i * size + j]);
        }
        printf("\n");
    }
}
