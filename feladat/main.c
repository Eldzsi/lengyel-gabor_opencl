#define CL_TARGET_OPENCL_VERSION 220

#include "matrix.h"
#include "file.h"
#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>

const int SAMPLE_SIZE = 2;

int main() {
    long long det;
    int* matrix = malloc(SAMPLE_SIZE * SAMPLE_SIZE * sizeof(int));
    if (matrix == NULL) {
        printf("[ERROR] Failed to allocate memory!");
        return -1;
    }
    clock_t start, end;

    generate_matrix(matrix, SAMPLE_SIZE);

    if (SAMPLE_SIZE < 10) {
        printf("\nMatrix:\n");
        print_matrix(matrix, SAMPLE_SIZE);
    }

    int i;
    cl_int err;
    int error_code;

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
    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

    // Build the program
    const char* kernel_code = load_kernel_source("kernel/sample.cl", &error_code);
    if (error_code != 0) {
        printf("Source code loading error!\n");
        return 0;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Build error! Code: %d\n", err);
        return 0;
    }

    cl_kernel kernel = clCreateKernel(program, "sample_kernel", NULL);

    // Create the host buffer and initialize it
    int* host_buffer = (int*)malloc(SAMPLE_SIZE * sizeof(int));
    for (i = 0; i < SAMPLE_SIZE; ++i) {
        host_buffer[i] = i;
    }

    // Create the device buffer
    cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_buffer);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&SAMPLE_SIZE);

    // Create the command queue
    const cl_queue_properties props[] = {0};
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &err);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        device_buffer,
        CL_FALSE,
        0,
        SAMPLE_SIZE * sizeof(int),
        host_buffer,
        0,
        NULL,
        NULL
    );

    // Size specification
    size_t local_work_size = 256;
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    // Apply the kernel on the range
    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
    );

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        device_buffer,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(int),
        host_buffer,
        0,
        NULL,
        NULL
    );

    for (i = 0; i < SAMPLE_SIZE; ++i) {
        //printf("[%d] = %d, ", i, host_buffer[i]);
    }

    start = clock();
    calculate_determinant(matrix, SAMPLE_SIZE, &det);
    end = clock();
    printf("\nDet: %lld", det);

    double exe_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nExecution time: %.4f s\n", exe_time);
    write_benchmark_to_file("sequential_benchmark.txt", SAMPLE_SIZE, exe_time);

    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    free(host_buffer);
    free(matrix);

    return 0;
}
