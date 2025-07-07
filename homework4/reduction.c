#include "solve.h"
#include <cuda_runtime.h>

const int BLOCK_SIZE = 128;

__global__ void reduction_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float data[BLOCK_SIZE];

    if(idx < N) {
        data[threadIdx.x] = input[idx];
    }
    else {
        data[threadIdx.x] = 0.0f; 
    }

    __syncthreads();
    float temp=0;
   for (int i=BLOCK_SIZE-1;i>=0;i--){
        temp+=data[i];
        __syncthreads();
   }
   data[0]=temp;
    if(threadIdx.x == 0) atomicAdd(output, data[0]);
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    int blocksPerGrid = (N+BLOCK_SIZE-1) / BLOCK_SIZE;
    cudaMemset(output, 0, sizeof(float));
    reduction_kernel<<<blocksPerGrid, BLOCK_SIZE>>>(input, output, N);
    cudaDeviceSynchronize();
}