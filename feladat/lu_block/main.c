#define CL_TARGET_OPENCL_VERSION 220

#include "matrix.h"
#include "file.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
    #include <direct.h>
    #define mkdir(path, mode) _mkdir(path)
#else
    #include <sys/stat.h>
    #include <sys/types.h>
#endif

int MATRIX_SIZE = 1000;

#define MAX_MATRIX_SIZE_CPU 2000

int main(int argc, char* argv[]) {
    if (argc > 1) {
        MATRIX_SIZE = atoi(argv[1]);
    }

    float* matrix_gpu = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* matrix_cpu = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    if (matrix_gpu == NULL || matrix_cpu == NULL) {
        return -1;
    }

    #ifdef _WIN32
        _mkdir("outputs");
    #else
        mkdir("outputs", 0777);
    #endif

    generate_matrix(matrix_gpu, MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        matrix_cpu[i] = matrix_gpu[i];
    }

    if (MATRIX_SIZE <= 10) {
        printf("\nGenerated Matrix (%dx%d):\n", MATRIX_SIZE, MATRIX_SIZE);
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                printf("%2.0f ", matrix_gpu[i*MATRIX_SIZE+j]);
            }
            printf("\n");
        }
    }

    float cpu_mantissa = 0.0;
    long long cpu_exponent = 0;
    int cpu_sign = 1;

    printf("\n===================================\n");
    printf("CPU\n");
    printf("-----------------------------------\n");

    if (MATRIX_SIZE <= MAX_MATRIX_SIZE_CPU) {
        clock_t start_cpu = clock();
        
        calculate_determinant_gauss(matrix_cpu, MATRIX_SIZE, &cpu_mantissa, &cpu_exponent, &cpu_sign);
        
        clock_t end_cpu = clock();
        float cpu_time = (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

        printf("Execution time (CPU): %.4f s\n", cpu_time);
        
        if (cpu_mantissa == 0.0) {
            printf("Determinant (CPU): 0\n");
        } else {
            printf("Determinant (CPU): %s%.4f * 10^%lld\n", cpu_sign < 0 ? "-" : "", cpu_mantissa, cpu_exponent);
        }
        
        write_benchmark_to_file("outputs/benchmark_cpu.txt", MATRIX_SIZE, cpu_time);
    } else {
        printf("Skipped (Matrix size > %d)\n", MAX_MATRIX_SIZE_CPU);
    }

    printf("===================================\n");
    printf("GPU\n");
    printf("-----------------------------------\n");

    float gpu_mantissa;
    long long gpu_exponent;
    int gpu_sign;
    float gpu_time_write, gpu_time_calc, gpu_time_read;

    clock_t start_gpu = clock();
    
    calculate_determinant_gauss_opencl(matrix_gpu, MATRIX_SIZE, &gpu_mantissa, &gpu_exponent, &gpu_sign, &gpu_time_write, &gpu_time_calc, &gpu_time_read);
    
    clock_t end_gpu = clock();
    float gpu_time = (float)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

    if (gpu_mantissa == 0.0) {
        printf("Determinant (GPU): 0\n");
    } else {
        printf("Determinant (GPU): %s%.4f * 10^%lld\n", gpu_sign < 0 ? "-" : "", gpu_mantissa, gpu_exponent);
    }
    
    printf("CPU -> GPU: %.4f s\n", gpu_time_write);
    printf("GPU Computing: %.4f s\n", gpu_time_calc);
    printf("GPU -> CPU: %.4f s\n", gpu_time_read);
    printf("Total execution time (GPU): %.4f s\n", gpu_time);
    printf("===================================\n");
    
    if (MATRIX_SIZE <= MAX_MATRIX_SIZE_CPU) {
        printf("\nDiagonal comparison:\n");
        printf("%-5s | %-15s | %-15s | %-10s\n", "Index", "CPU Diagonal", "GPU Diagonal", "Diff");
        printf("------------------------------------------------------------\n");

        int limit = (MATRIX_SIZE < 10) ? MATRIX_SIZE : 10;
        for (int i = 0; i < limit; i++) {
            float c_val = matrix_cpu[i * MATRIX_SIZE + i];
            float g_val = matrix_gpu[i * MATRIX_SIZE + i];
            printf("%-5d | %-15.6f | %-15.6f | %-10.6e\n", i, c_val, g_val, fabs(c_val - g_val));
        }
        printf("===================================\n");

        if (cpu_mantissa != 0.0) {
            double gpu_part = (double)(gpu_sign * gpu_mantissa);
            double cpu_part = (double)(cpu_sign * cpu_mantissa);
            
            double ratio = (gpu_part / cpu_part) * pow(10.0, (double)(gpu_exponent - cpu_exponent));
            double error_percent = fabs(ratio - 1.0) * 100.0;
            
            printf("Relative Error: %.6f %%\n", error_percent);
            printf("===================================\n");
        }
    }

    write_benchmark_to_file("outputs/benchmark_gpu.txt", MATRIX_SIZE, gpu_time);

    free(matrix_gpu);
    free(matrix_cpu);

    return 0;
}
