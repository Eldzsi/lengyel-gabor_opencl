#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <CL/cl.h>

const int SAMPLE_SIZE = 1000;

void deviationSequential(float* data, int n, float* mean, float* deviation);


int main(void) {
    int i;
    cl_int err;

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

    // Load kernel from file
    size_t kernel_size;
    char* kernel_source = loadKernelFromFile("kernels/sample.cl", &kernel_size);
    if (!kernel_source) {
        printf("[ERROR] Failed to load kernel source from file.\n");
        return -1;
    }

    // Build the program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_size, &err);
    free(kernel_source);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* build_log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        printf("Build error log: %s\n", build_log);
        free(build_log);
        return 0;
    }
    cl_kernel mean_kernel = clCreateKernel(program, "mean_kernel", &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateKernel failed with error code: %d\n", err);
        return -1;
    }
    cl_kernel deviation_kernel = clCreateKernel(program, "deviation_kernel", &err);

    // Generate random numbers
    float* host_buffer = (float*)malloc(SAMPLE_SIZE * sizeof(float));
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        host_buffer[i] = (float)(rand() % 100) / 10.0f;
    }

    // Create the device buffer
    cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);
    cl_mem mean_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);
    cl_mem stddev_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);

    // Set kernel arguments
    clSetKernelArg(mean_kernel, 0, sizeof(cl_mem), &device_buffer);
    clSetKernelArg(mean_kernel, 1, sizeof(cl_mem), &mean_buffer);
    clSetKernelArg(mean_kernel, 2, sizeof(int), &SAMPLE_SIZE);

    clSetKernelArg(deviation_kernel, 0, sizeof(cl_mem), &device_buffer);
    clSetKernelArg(deviation_kernel, 1, sizeof(cl_mem), &mean_buffer);
    clSetKernelArg(deviation_kernel, 2, sizeof(cl_mem), &stddev_buffer);
    clSetKernelArg(deviation_kernel, 3, sizeof(int), &SAMPLE_SIZE);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(
        context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

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

    size_t global_size = SAMPLE_SIZE;
    clEnqueueNDRangeKernel(command_queue, mean_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clEnqueueNDRangeKernel(command_queue, deviation_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);

    float mean, deviation;
    clEnqueueReadBuffer(command_queue, mean_buffer, CL_TRUE, 0, sizeof(float), &mean, 0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, stddev_buffer, CL_TRUE, 0, sizeof(float), &deviation, 0, NULL, NULL);

    float mean_sequential, deviation_sequential;
    deviationSequential(host_buffer, SAMPLE_SIZE, &mean_sequential, &deviation_sequential);

    printf("\nAvg (sequential): %.4f", mean_sequential);
    printf("\nAvg (parallel): %.4f", mean);
    printf("\nDeviation (sequential): %.4f", deviation_sequential);
    printf("\nDeviation (parallel): %.4f", deviation);

    // Release the resources
    clReleaseMemObject(device_buffer);
    clReleaseMemObject(mean_buffer);
    clReleaseMemObject(stddev_buffer);
    clReleaseKernel(mean_kernel);
    clReleaseKernel(deviation_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    free(host_buffer);
}


void deviationSequential(float* data, int n, float* mean, float* deviation) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    *mean = sum/n;

    sum = 0.0;
    for (int i = 0; i < n; ++i) {
        float diff = data[i] - *mean;
        sum += diff * diff;
    }

    *deviation = sqrt(sum/n);
}