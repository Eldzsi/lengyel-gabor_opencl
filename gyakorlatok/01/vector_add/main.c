#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const char* kernel_code =
    "__kernel void vector_add(__global int* A, __global int* B, __global int* C, int n) {\n"
    "   int id = get_global_id(0);\n"
    "   if (id < n) {\n"
    "       C[id] = A[id] + B[id];\n"
    "   }\n"
    "}\n";

const int VECTOR_SIZE = 3;

void printVector(int* vector);

int main(void) {
    int i;
    cl_int err;

    int* A = (int*)malloc(VECTOR_SIZE * sizeof(int));
    int* B = (int*)malloc(VECTOR_SIZE * sizeof(int));
    int* C = (int*)malloc(VECTOR_SIZE * sizeof(int));

    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return -1;
    }

    A[0] = 1; A[1] = 2; A[2] = 3;
    B[0] = 4; B[1] = 5; B[2] = 6;

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

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    if (!kernel) {
        printf("[ERROR] Failed to create kernel. Error code: %d\n", err);
        return -1;
    }

    // Create the device buffers
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VECTOR_SIZE * sizeof(int), A, &err);
    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VECTOR_SIZE * sizeof(int), B, &err);
    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(int), NULL, &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_C);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&VECTOR_SIZE);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Size specification
    size_t global_work_size = VECTOR_SIZE;

    // Apply the kernel on the range
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(command_queue, buffer_C, CL_TRUE, 0, VECTOR_SIZE * sizeof(int), C, 0, NULL, NULL);

    printf("A = ");
    printVector(A);

    printf("B = ");
    printVector(B);

    printf("A + B = ");
    printVector(C);

    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C);

    free(A);
    free(B);
    free(C);

    return 0;
}


void printVector(int* vector) {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        printf("%d " , vector[i]);
    }
    printf("\n");
}