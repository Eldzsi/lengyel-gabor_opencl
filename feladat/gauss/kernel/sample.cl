__kernel void pivot_and_swap(__global float* matrix, int pivot_index, int size, __global int* sign) {
    int id = get_global_id(0);
    if (id != 0) {
        return; 
    }

    int max_row = pivot_index;
    float max_value = fabs(matrix[pivot_index * size + pivot_index]);
    
    for (int row = pivot_index + 1; row < size; row++) {
        float current_value = fabs(matrix[row * size + pivot_index]);
        if (current_value > max_value) {
            max_value = current_value;
            max_row = row;
        }
    }

    if (max_row != pivot_index) {
        for (int col = 0; col < size; col++) {
            float temp = matrix[pivot_index * size + col];
            matrix[pivot_index * size + col] = matrix[max_row * size + col];
            matrix[max_row * size + col] = temp;
        }
        *sign = -(*sign);
    }
}

__kernel void calculate_determinant_gauss(__global float* matrix, int pivot_index, int size) {
    int row = get_global_id(1) + pivot_index + 1;
    int col = get_global_id(0) + pivot_index + 1;
    
    if (row >= size || col >= size) {
        return;
    }
    
    float pivot = matrix[pivot_index * size + pivot_index];
    if (fabs(pivot) < 1e-12f) {
        return;
    }
    
    float factor = matrix[row * size + pivot_index] / pivot;
    matrix[row * size + col] -= factor * matrix[pivot_index * size + col];
}
