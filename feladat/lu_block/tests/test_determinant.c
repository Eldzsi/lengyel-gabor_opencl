#define CL_TARGET_OPENCL_VERSION 220

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <stdint.h>
#include <cmocka.h>

#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
char *strtok_r(char *s, const char *delim, char **save_ptr) {
    char *token;
    if (s == NULL) s = *save_ptr;
    s += strspn(s, delim);
    if (*s == '\0') return NULL;
    token = s;
    s = strpbrk(token, delim);
    if (s == NULL) {
        *save_ptr = strchr(token, '\0');
    } else {
        *s = '\0';
        *save_ptr = s + 1;
    }
    return token;
}
#endif

static void test_cpu_determinant_4x4() {
    float test_matrix[16] = {
        4, 4, 4, 4,
        6, 4, 1, 9,
        5, 6, 6, 5,
        9, 2, 6, 8
    };

    float mantissa = 0.0f;
    long long int exponent = 0;
    int sign = 1;

    calculate_determinant_gauss(test_matrix, 4, &mantissa, &exponent, &sign);
    double result = (double)sign * (double)mantissa * pow(10.0, (double)exponent);

    assert_true(fabs(result - 36.0) < 0.0001);
}

static void test_cpu_determinant_5x5() {
    float test_matrix[25] = {
        2, 0, 0, 0, 0,
        0, 3, 0, 0, 0,
        0, 0, 4, 0, 0,
        0, 0, 0, 5, 0,
        0, 0, 0, 0, 6
    };

    float mantissa = 0.0f;
    long long int exponent = 0;
    int sign = 1;

    calculate_determinant_gauss(test_matrix, 5, &mantissa, &exponent, &sign);
    double result = (double)sign * (double)mantissa * pow(10.0, (double)exponent);

    assert_true(fabs(result - 720.0) < 0.0001);
}

static void test_cpu_determinant_6x6_zero() {
    float test_matrix[36] = {
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6,  
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1
    };

    float mantissa = 0.0f;
    long long int exponent = 0;
    int sign = 1;

    calculate_determinant_gauss(test_matrix, 6, &mantissa, &exponent, &sign);
    double result = (double)sign * (double)mantissa * pow(10.0, (double)exponent);

    assert_true(fabs(result - 0.0) < 0.0001);
}

static void test_gpu_determinant_4x4() {
    float test_matrix[16] = {
        4, 4, 4, 4,
        6, 4, 1, 9,
        5, 6, 6, 5,
        9, 2, 6, 8
    };

    float mantissa = 0.0f;
    long long int exponent = 0;
    int sign = 1;

    calculate_determinant_gauss_opencl(test_matrix, 4, &mantissa, &exponent, &sign, NULL, NULL, NULL);
    double result = (double)sign * (double)mantissa * pow(10.0, (double)exponent);

    assert_true(fabs(result - 36.0) < 0.0001);
}

static void test_gpu_determinant_5x5() {
    float test_matrix[25] = {
        2, 0, 0, 0, 0,
        0, 3, 0, 0, 0,
        0, 0, 4, 0, 0,
        0, 0, 0, 5, 0,
        0, 0, 0, 0, 6
    };

    float mantissa = 0.0f;
    long long int exponent = 0;
    int sign = 1;

    calculate_determinant_gauss_opencl(test_matrix, 5, &mantissa, &exponent, &sign, NULL, NULL, NULL);
    double result = (double)sign * (double)mantissa * pow(10.0, (double)exponent);

    assert_true(fabs(result - 720.0) < 0.0001);
}

static void test_gpu_determinant_6x6_zero() {
    float test_matrix[36] = {
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6,  
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1
    };

    float mantissa = 0.0f;
    long long int exponent = 0;
    int sign = 1;

    calculate_determinant_gauss_opencl(test_matrix, 6, &mantissa, &exponent, &sign, NULL, NULL, NULL);
    double result = (double)sign * (double)mantissa * pow(10.0, (double)exponent);

    assert_true(fabs(result - 0.0) < 0.0001);
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cpu_determinant_4x4),
        cmocka_unit_test(test_cpu_determinant_5x5),
        cmocka_unit_test(test_cpu_determinant_6x6_zero),
        cmocka_unit_test(test_gpu_determinant_4x4),
        cmocka_unit_test(test_gpu_determinant_5x5),
        cmocka_unit_test(test_gpu_determinant_6x6_zero),
    };

    printf("Matrix Determinant Tests\n");
    return cmocka_run_group_tests(tests, NULL, NULL);
}
