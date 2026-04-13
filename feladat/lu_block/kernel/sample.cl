#define BLOCK_SIZE 64

__kernel void lu_factorize_block(__global float* matrix, int k, int n) {
    __local float local_block[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    
    int row = k + ty;
    int col = k + tx;

    if (row < n && col < n) {
        local_block[ty][tx] = matrix[row * n + col];
    } else {
        local_block[ty][tx] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int p = 0; p < BLOCK_SIZE; p++) {
        if (ty > p && tx == p) {
            float pivot = local_block[p][p];
            if (fabs(pivot) > 1e-12f) {
                local_block[ty][tx] /= pivot;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (ty > p && tx > p) {
            local_block[ty][tx] -= local_block[ty][p] * local_block[p][tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < n && col < n) {
        matrix[row * n + col] = local_block[ty][tx];
    }
}

__kernel void lu_update_panels(__global float* matrix, int k, int n) {
    int idx = get_global_id(0);
    int remaining = n - k - BLOCK_SIZE;

    if (idx >= remaining) {
        return;
    }

    int r = k + BLOCK_SIZE + idx;
    int c_u = k + BLOCK_SIZE + idx;

    for (int p = 0; p < BLOCK_SIZE; p++) {
        float pivot = matrix[(k + p) * n + (k + p)];
        if (fabs(pivot) > 1e-12f) {
            matrix[r * n + (k + p)] /= pivot;
        }
        float factor = matrix[r * n + (k + p)];
        
        for (int c = p + 1; c < BLOCK_SIZE; c++) {
            matrix[r * n + (k + c)] -= factor * matrix[(k + p) * n + (k + c)];
        }
    }

    for (int p = 0; p < BLOCK_SIZE; p++) {
        for (int row_u = p + 1; row_u < BLOCK_SIZE; row_u++) {
            float factor = matrix[(k + row_u) * n + (k + p)];
            matrix[(k + row_u) * n + c_u] -= factor * matrix[(k + p) * n + c_u];
        }
    }
}

__kernel void lu_update_trailing_matrix(__global float* matrix, int k, int n) {
    int col = get_global_id(0) + k + BLOCK_SIZE;
    int row = get_global_id(1) + k + BLOCK_SIZE;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int p = 0; p < BLOCK_SIZE; p++) {
        sum += matrix[row * n + (k + p)] * matrix[(k + p) * n + col];
    }
    
    matrix[row * n + col] -= sum;
}
