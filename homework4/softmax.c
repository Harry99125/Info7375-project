#include "solve.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>


__global__ void exp_kernel(const float* input, float* exp_arr, float maxval, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        exp_arr[idx] = expf(input[idx] - maxval);
}



__global__ void softmax_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < N; i++){ sum += input[i];}
     
    if (idx < N)
        output[idx] = input[idx] / sum;

}
void solve(const float* input, float* output, int N) {

    float* input1 = (float*)malloc(N * sizeof(float));
    cudaMemcpy(input1, input, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxval = 0;
    for (int i = 0; i < N; i++)
        if (input1[i] > maxval) {
            maxval = input1[i];
        }


    float* exp_arr;
    cudaMalloc(&exp_arr, N * sizeof(float));
    int BLOCK = 256, GRID = (N + BLOCK - 1) / BLOCK;
    exp_kernel<<<GRID, BLOCK>>>(input, exp_arr, maxval, N);
    cudaDeviceSynchronize();
 
    softmax_kernel<<<GRID, BLOCK>>>(exp_arr, output, N);
    cudaFree(exp_arr);
    free(cpu_input);
}
