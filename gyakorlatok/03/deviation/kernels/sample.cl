__kernel void mean_kernel(__global float* data, __global float* mean, int n) {
    int gid = get_global_id(0);
    float sum = 0.0f;
    
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    
    *mean = sum / n;
}


__kernel void deviation_kernel(__global float* data, __global float* mean, __global float* stddev, int n) {
    int gid = get_global_id(0);
    float local_mean = *mean;
    float sum = 0.0f;
    
    for (int i = 0; i < n; ++i) {
        float diff = data[i] - local_mean;
        sum += diff * diff;
    }

    *stddev = sqrt(sum / n);
}