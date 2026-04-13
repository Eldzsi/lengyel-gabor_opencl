__kernel void pivot_and_swap(__global float* matrix, int k, int n, __global int* sign) {
    int id = get_global_id(0);
    if (id != 0) {
        return; 
    }

    int max_row = k;
    float max_val = fabs(matrix[k * n + k]);
    
    for (int r = k + 1; r < n; r++) {
        float current_val = fabs(matrix[r * n + k]);
        if (current_val > max_val) {
            max_val = current_val;
            max_row = r;
        }
    }

    if (max_row != k) {
        for (int c = 0; c < n; c++) {
            float tmp = matrix[k * n + c];
            matrix[k * n + c] = matrix[max_row * n + c];
            matrix[max_row * n + c] = tmp;
        }
        *sign = -(*sign);
    }
}

__kernel void calculate_determinant_gauss(__global float* matrix, int k, int n) {
    int j = get_global_id(0) + k + 1;
    int i = get_global_id(1) + k + 1;
    
    if (i >= n || j >= n) {
        return;
    }
    
    float pivot = matrix[k * n + k];
    if (fabs(pivot) < 1e-12f) {
        return;
    }
    
    float factor = matrix[i * n + k] / pivot;
    matrix[i * n + j] -= factor * matrix[k * n + j];
}
