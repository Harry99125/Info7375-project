#include <cstdio>
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
int main() {
    const int N=5;
    float h_in[N]={1,2,3,4,5}, h_out=0;
    float *d_in,*d_out;
    cudaMalloc(&d_in,sizeof(h_in));
    cudaMalloc(&d_out,sizeof(float));
    cudaMemcpy(d_in,h_in,sizeof(h_in),cudaMemcpyHostToDevice);

    solve(d_in,d_out,N);

    cudaMemcpy(&h_out,d_out,sizeof(float),cudaMemcpyDeviceToHost);
    printf("Sum = %f\n",h_out);
    printf("test finished");

    cudaFree(d_in); cudaFree(d_out);
}