__kernel void calculate_determinant_gauss(__global float* matrix, int k, int n) {
    int i = get_global_id(0) + k + 1;

    if (i >= n) return;

    float pivot = matrix[k * n + k];
    
    float factor = matrix[i * n + k] / pivot;

    for (int j = k; j < n; j++) {
        matrix[i * n + j] -= factor * matrix[k * n + j];
    }
}