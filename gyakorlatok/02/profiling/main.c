#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

const int SAMPLE_SIZE = 1000;


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
    cl_kernel kernel = clCreateKernel(program, "hello_kernel", &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateKernel failed with error code: %d\n", err);
        return -1;
    }

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

    // Size specification
    size_t local_work_size = 256;
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    // Apply the kernel on the range
    cl_event event;
    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        &event
    );
    clFinish(command_queue);

    // Show profiling information
    cl_ulong queued, submit, start, end, time;
    err = clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_QUEUED,
        sizeof(queued),
        &queued,
        NULL
    );
    if (err == CL_PROFILING_INFO_NOT_AVAILABLE) {
        printf("Profiling info not available!\n");
        return 0;
    } else if (err != CL_SUCCESS) {
        printf("Error code: %d\n", err);
        return 0;
    }
    //printf("Queued : %lu\n", queued);

    clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_SUBMIT,
        sizeof(submit),
        &submit,
        NULL
    );
    //printf("Submit : %lu\n", submit);

    time = submit - queued;
    printf("Queued -> Submit: %.4f ms\n", time * 1e-6);

    clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_START,
        sizeof(start),
        &start,
        NULL
    );
    //printf("Start : %lu\n", start);

    time = start - submit;
    printf("Submit -> Start: %.4f ms\n", time * 1e-6);

    clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_END,
        sizeof(end),
        &end,
        NULL
    );
    //printf("End : %lu\n", end);

    time = end - start;
    printf("Start -> End: %.4f ms\n", time * 1e-6);

    time = end - queued;
    printf("Kernel execution time: %.4f ms\n", time * 1e-6);

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

    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    free(host_buffer);
}
