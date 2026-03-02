#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void generate_matrix(float* matrix, int size) {
    srand(time(NULL));
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (float)(rand() % 11);
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

void calculate_determinant_gauss(float* matrix, int size, double* out_mantissa, long long* out_exponent, int* out_sign) {
    float* temp = (float*)malloc(size * size * sizeof(float));

    if (temp == NULL) {
        *out_mantissa = 0.0;
        *out_exponent = 0;
        *out_sign = 1;
        return;
    }
    
    for (int i = 0; i < size * size; i++) {
        temp[i] = matrix[i];
    }

    double mantissa = 1.0;
    long long exponent = 0;
    int sign = 1;

    for (int k = 0; k < size; k++) {
        float pivot = temp[k * size + k];
    
        for (int i = k + 1; i < size; i++) {
            float factor = temp[i * size + k] / pivot;
            for (int j = k; j < size; j++) {
                temp[i * size + j] -= factor * temp[k * size + j];
            }
        }
        
        double value = pivot;
        if (value < 0) {
            sign = -sign;
            value = -value;
        }

        mantissa *= value;

        while (mantissa >= 10.0) {
            mantissa /= 10.0;
            exponent++;
        }
        while (mantissa < 1.0 && mantissa > 0.0) {
            mantissa *= 10.0;
            exponent--;
        }
    }

    free(temp);
    
    *out_mantissa = mantissa;
    *out_exponent = exponent;
    *out_sign = sign;
}