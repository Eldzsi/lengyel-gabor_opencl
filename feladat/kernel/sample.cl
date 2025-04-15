__kernel void determinant_kernel(__global int* submatrices, __global long* results) {
    int id = get_global_id(0);

    __global int* m = submatrices + id * 9;

    long det = 
        m[0] * (m[4] * m[8] - m[5] * m[7]) -
        m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);

    results[id] = det;
}
