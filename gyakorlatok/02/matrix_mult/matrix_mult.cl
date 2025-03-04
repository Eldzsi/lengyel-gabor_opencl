__kernel void matrix_mult(__global int* A, __global int* B, __global int* C, int n) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}