#define CL_TARGET_OPENCL_VERSION 220

#include "matrix.h"
#include "file.h"
#include "kernel_loader.h"

#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
    #include <direct.h>
    #define mkdir(path, mode) _mkdir(path)
#else
    #include <sys/stat.h>
    #include <sys/types.h>
#endif

int MATRIX_SIZE = 1000;

int main(int argc, char* argv[]) {
    if (argc > 1) {
        MATRIX_SIZE = atoi(argv[1]);
    }

    float* matrix = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    if (matrix == NULL) return -1;

    #ifdef _WIN32
        _mkdir("outputs");
    #else
        mkdir("outputs", 0777);
    #endif

    srand(time(NULL));
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        float r = (float)rand() / (float)RAND_MAX; 
        matrix[i] = (r * 2.0f) + 1.0f; 
    }

    if (MATRIX_SIZE <= 10) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 9; j++) {
                printf("%3.0f ", matrix[i*MATRIX_SIZE+j]);
            }
            printf("\n");
        }
    }

    clock_t start_cpu = clock();
    
    double cpu_mantissa;
    long long cpu_exponent;
    int cpu_sign;
    
    calculate_determinant_gauss(matrix, MATRIX_SIZE, &cpu_mantissa, &cpu_exponent, &cpu_sign);
    
    clock_t end_cpu = clock();
    double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    printf("\n===================================\n");
    printf("CPU\n");
    printf("-----------------------------------\n");
    printf("Execution time (CPU): %.4f s\n", cpu_time);
    
    if (cpu_mantissa == 0.0) {
        printf("Determinant (CPU): 0\n");
    } else {
        printf("Determinant (CPU): %s%.4f * 10^%lld\n", cpu_sign < 0 ? "-" : "", cpu_mantissa, cpu_exponent);
    }
    
    write_benchmark_to_file("outputs/benchmark_cpu.txt", MATRIX_SIZE, cpu_time);

    cl_int err;
    cl_uint n_platforms, n_devices;
    cl_platform_id platform_id;
    cl_device_id device_id;

    clGetPlatformIDs(1, &platform_id, &n_platforms);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if (err != CL_SUCCESS) err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &n_devices);
    
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, props, &err);

    int error_code;
    const char* kernel_path = "kernel/sample.cl"; 
    char* kernel_code = load_kernel_source(kernel_path, &error_code);
    if (error_code != 0) {
        kernel_code = load_kernel_source("sample.cl", &error_code);
    }

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, NULL, &err);
    free(kernel_code);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "calculate_determinant_gauss", &err);
    
    cl_event write_event, read_event;
    cl_event* kernel_events = (cl_event*)malloc((MATRIX_SIZE - 1) * sizeof(cl_event));

    cl_mem gpu_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), NULL, &err);
    clEnqueueWriteBuffer(queue, gpu_matrix, CL_FALSE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), matrix, 0, NULL, &write_event);
    
    clock_t start_wall = clock();

    for (int k = 0; k < MATRIX_SIZE - 1; k++) {
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_matrix);
        clSetKernelArg(kernel, 1, sizeof(int), &k);
        clSetKernelArg(kernel, 2, sizeof(int), &MATRIX_SIZE);

        size_t global_work_size = MATRIX_SIZE - 1 - k;
        
        if (global_work_size > 0) {
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &kernel_events[k]);
        }
    }
    
    clFinish(queue); 
    clock_t end_wall = clock();
    double total_wall_time = (double)(end_wall - start_wall) / CLOCKS_PER_SEC;

    err = clEnqueueReadBuffer(queue, gpu_matrix, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), matrix, 0, NULL, &read_event);

    cl_ulong time_start, time_end;
    
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double time_write_sec = (double)(time_end - time_start) / 1.0e9;

    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double time_read_sec = (double)(time_end - time_start) / 1.0e9;

    cl_ulong total_kernel_ns = 0;
    for (int k = 0; k < MATRIX_SIZE - 1; k++) {
        if ((MATRIX_SIZE - 1 - k) > 0) {
            clGetEventProfilingInfo(kernel_events[k], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
            clGetEventProfilingInfo(kernel_events[k], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
            total_kernel_ns += (time_end - time_start);
            clReleaseEvent(kernel_events[k]);
        }
    }
    double pure_gpu_calc_time = (double)total_kernel_ns / 1.0e9;

    double api_overhead = total_wall_time - pure_gpu_calc_time;

    printf("===================================\n");
    printf("GPU\n");
    printf("-----------------------------------\n");
    printf("CPU -> GPU: %.4f s\n", time_write_sec);
    printf("GPU Computing: %.4f s\n", pure_gpu_calc_time);
    printf("GPU -> CPU: %.4f s\n", time_read_sec);
    printf("Total execution time (GPU): %.4f s\n", total_wall_time);

    double mantissa = 1.0;
    long long exponent = 0;
    int sign = 1;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        double val = matrix[i * MATRIX_SIZE + i];
        
        if (val == 0.0) { mantissa = 0.0; exponent = 0; break; }
        
        if (val < 0) { sign = -sign; val = -val; }

        mantissa *= val;

        while (mantissa >= 10.0) { mantissa /= 10.0; exponent++; }
        while (mantissa < 1.0 && mantissa > 0.0) { mantissa *= 10.0; exponent--; }
    }

    if (mantissa == 0.0) {
        printf("Determinant (GPU): 0\n");
    } else {
        printf("Determinant (GPU): %s%.4f * 10^%lld\n", sign < 0 ? "-" : "", mantissa, exponent);
    }
    printf("===================================\n");

    write_benchmark_to_file("outputs/benchmark_gpu.txt", MATRIX_SIZE, total_wall_time);

    free(kernel_events);
    clReleaseEvent(write_event);
    clReleaseEvent(read_event);
    clReleaseMemObject(gpu_matrix);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(matrix);

    return 0;
}