#define CL_TARGET_OPENCL_VERSION 220

#include "matrix.h"
#include "file.h"
#include "kernel_loader.h"

#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 64

void generate_matrix(float* matrix, int size) {
    srand(time(NULL));
    
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (float)(rand() % 10); 
        
        if (i % size == i / size) {
            matrix[i] += size * 10.0f; 
        }
    }
}

void print_matrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.2f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

void calculate_determinant_gauss(float* matrix, int size, float* out_mantissa, long long* out_exponent, int* out_sign) {
    int sign = 1;

    for (int k = 0; k < size - 1; k++) {
        float pivot = matrix[k * size + k];

        if (fabs(pivot) < 1e-12) {
            *out_mantissa = 0.0;
            *out_exponent = 0;
            *out_sign = 1;
            return;
        }
    
        for (int i = k + 1; i < size; i++) {
            float factor = matrix[i * size + k] / pivot;
            
            for (int j = k + 1; j < size; j++) {
                matrix[i * size + j] -= factor * matrix[k * size + j];
            }
            
            matrix[i * size + k] = 0.0f;
        }
    }

    float mantissa = 1.0;
    long long exponent = 0;

    for (int i = 0; i < size; i++) {
        float val = matrix[i * size + i];

        if (fabs(val) < 1e-12) {
            mantissa = 0.0;
            exponent = 0; 
            break;
        }

        if (val < 0) {
            sign = -sign;
            val = -val;
        }

        mantissa *= val;

        while (mantissa >= 10.0) {
            mantissa /= 10.0;
            exponent++;
        }
        while (mantissa < 1.0 && mantissa > 0.0) {
            mantissa *= 10.0;
            exponent--;
        }
    }

    *out_mantissa = mantissa;
    *out_exponent = exponent;
    *out_sign = sign;
}

void calculate_determinant_gauss_opencl(float* matrix, int size, float* out_mantissa, long long* out_exponent, int* out_sign, float* out_time_write, float* out_time_calc, float* out_time_read) {
    cl_int err;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint n_platforms, n_devices;

    clGetPlatformIDs(1, &platform_id, &n_platforms);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if (err != CL_SUCCESS) err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &n_devices);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, props, &err);

    int error_code;
    char* kernel_code = load_kernel_source("kernel/sample.cl", &error_code);
    if (error_code != 0) kernel_code = load_kernel_source("sample.cl", &error_code);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, NULL, &err);
    free(kernel_code);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    cl_kernel kernel_fact = clCreateKernel(program, "lu_factorize_block", &err);
    cl_kernel kernel_panel = clCreateKernel(program, "lu_update_panels", &err);
    cl_kernel kernel_trail = clCreateKernel(program, "lu_update_trailing_matrix", &err);

    cl_event write_event, read_event;
    
    cl_mem gpu_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, size * size * sizeof(float), NULL, &err);
    clEnqueueWriteBuffer(queue, gpu_matrix, CL_FALSE, 0, size * size * sizeof(float), matrix, 0, NULL, &write_event);
    
    cl_event calc_start_event, calc_end_event;
    clEnqueueMarkerWithWaitList(queue, 0, NULL, &calc_start_event);

    for (int k = 0; k < size; k += BLOCK_SIZE) {
        
        clSetKernelArg(kernel_fact, 0, sizeof(cl_mem), &gpu_matrix);
        clSetKernelArg(kernel_fact, 1, sizeof(int), &k);
        clSetKernelArg(kernel_fact, 2, sizeof(int), &size);
        
        size_t local_fact[2] = {BLOCK_SIZE, BLOCK_SIZE};
        size_t global_fact[2] = {BLOCK_SIZE, BLOCK_SIZE};
        clEnqueueNDRangeKernel(queue, kernel_fact, 2, NULL, global_fact, local_fact, 0, NULL, NULL);

        int remaining = size - k - BLOCK_SIZE;
        if (remaining > 0) {
            clSetKernelArg(kernel_panel, 0, sizeof(cl_mem), &gpu_matrix);
            clSetKernelArg(kernel_panel, 1, sizeof(int), &k);
            clSetKernelArg(kernel_panel, 2, sizeof(int), &size);
            
            size_t global_panel = remaining;
            clEnqueueNDRangeKernel(queue, kernel_panel, 1, NULL, &global_panel, NULL, 0, NULL, NULL);

            clSetKernelArg(kernel_trail, 0, sizeof(cl_mem), &gpu_matrix);
            clSetKernelArg(kernel_trail, 1, sizeof(int), &k);
            clSetKernelArg(kernel_trail, 2, sizeof(int), &size);
            
            size_t global_trail[2] = {remaining, remaining};
            clEnqueueNDRangeKernel(queue, kernel_trail, 2, NULL, global_trail, NULL, 0, NULL, NULL);
        }
    }
    
    clEnqueueMarkerWithWaitList(queue, 0, NULL, &calc_end_event);
    clFinish(queue);

    clEnqueueReadBuffer(queue, gpu_matrix, CL_TRUE, 0, size * size * sizeof(float), matrix, 0, NULL, &read_event);

    cl_ulong time_start, time_end;
    
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    float time_write_sec = (float)(time_end - time_start) / 1.0e9;

    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    float time_read_sec = (float)(time_end - time_start) / 1.0e9;

    clGetEventProfilingInfo(calc_start_event, CL_PROFILING_COMMAND_END, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(calc_end_event, CL_PROFILING_COMMAND_START, sizeof(time_end), &time_end, NULL);
    float gpu_calc = (float)(time_end - time_start) / 1.0e9;

    if (out_time_write != NULL) *out_time_write = time_write_sec;
    if (out_time_calc != NULL) *out_time_calc = gpu_calc;
    if (out_time_read != NULL) *out_time_read = time_read_sec;

    float mantissa = 1.0;
    long long exponent = 0;
    int sign = 1;

    for (int i = 0; i < size; i++) {
        float val = matrix[i * size + i];

        if (fabs(val) < 1e-12) {
            mantissa = 0.0;
            exponent = 0; 
            break;
        }

        if (val < 0) {
            sign = -sign;
            val = -val;
        }

        mantissa *= val;

        while (mantissa >= 10.0) {
            mantissa /= 10.0;
            exponent++;
        }
        while (mantissa < 1.0 && mantissa > 0.0) {
            mantissa *= 10.0;
            exponent--;
        }
    }

    *out_mantissa = mantissa;
    *out_exponent = exponent;
    *out_sign = sign;

    clReleaseMemObject(gpu_matrix);
    clReleaseKernel(kernel_fact);
    clReleaseKernel(kernel_panel);
    clReleaseKernel(kernel_trail);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
