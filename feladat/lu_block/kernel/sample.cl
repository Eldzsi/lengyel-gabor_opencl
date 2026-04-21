#define BLOCK_SIZE 128

__kernel void lu_factorize_block(__global float* matrix, int block_offset, int matrix_size) {
    __local float local_block[BLOCK_SIZE][BLOCK_SIZE];
    
    int local_col = get_local_id(0);
    int local_row = get_local_id(1);
    
    int global_row = block_offset + local_row;
    int global_col = block_offset + local_col;

    if (global_row < matrix_size && global_col < matrix_size) {
        local_block[local_row][local_col] = matrix[global_row * matrix_size + global_col];
    } else {
        local_block[local_row][local_col] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int local_pivot_index = 0; local_pivot_index < BLOCK_SIZE; local_pivot_index++) {
        if (local_row > local_pivot_index && local_col == local_pivot_index) {
            float pivot = local_block[local_pivot_index][local_pivot_index];
            if (fabs(pivot) > 1e-12f) {
                local_block[local_row][local_col] /= pivot;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_row > local_pivot_index && local_col > local_pivot_index) {
            local_block[local_row][local_col] -= local_block[local_row][local_pivot_index] * local_block[local_pivot_index][local_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < matrix_size && global_col < matrix_size) {
        matrix[global_row * matrix_size + global_col] = local_block[local_row][local_col];
    }
}

__kernel void lu_update_panels(__global float* matrix, int block_offset, int matrix_size) {
    int id = get_global_id(0);
    int remaining_size = matrix_size - block_offset - BLOCK_SIZE;

    if (id >= remaining_size) {
        return;
    }

    int panel_row = block_offset + BLOCK_SIZE + id;
    int panel_col = block_offset + BLOCK_SIZE + id;

    for (int local_pivot_index = 0; local_pivot_index < BLOCK_SIZE; local_pivot_index++) {
        float pivot = matrix[(block_offset + local_pivot_index) * matrix_size + (block_offset + local_pivot_index)];
        if (fabs(pivot) > 1e-12f) {
            matrix[panel_row * matrix_size + (block_offset + local_pivot_index)] /= pivot;
        }
        float factor = matrix[panel_row * matrix_size + (block_offset + local_pivot_index)];
        
        for (int inner_col = local_pivot_index + 1; inner_col < BLOCK_SIZE; inner_col++) {
            matrix[panel_row * matrix_size + (block_offset + inner_col)] -= factor * matrix[(block_offset + local_pivot_index) * matrix_size + (block_offset + inner_col)];
        }
    }

    for (int local_pivot_index = 0; local_pivot_index < BLOCK_SIZE; local_pivot_index++) {
        for (int inner_row = local_pivot_index + 1; inner_row < BLOCK_SIZE; inner_row++) {
            float factor = matrix[(block_offset + inner_row) * matrix_size + (block_offset + local_pivot_index)];
            matrix[(block_offset + inner_row) * matrix_size + panel_col] -= factor * matrix[(block_offset + local_pivot_index) * matrix_size + panel_col];
        }
    }
}

__kernel void lu_update_trailing_matrix(__global float* matrix, int block_offset, int matrix_size) {
    int global_col = get_global_id(0) + block_offset + BLOCK_SIZE;
    int global_row = get_global_id(1) + block_offset + BLOCK_SIZE;

    if (global_row >= matrix_size || global_col >= matrix_size) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        sum += matrix[global_row * matrix_size + (block_offset + i)] * matrix[(block_offset + i) * matrix_size + global_col];
    }
    
    matrix[global_row * matrix_size + global_col] -= sum;
}
