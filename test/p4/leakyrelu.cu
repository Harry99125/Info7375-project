
#include <cuda_runtime.h>
#include <cstdio>
__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x; 
     if(row<N){
        if(input[row]<0){
            output[row]=0.01f*input[row];
        }
        else{
            output[row]=input[row];
        }
     }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

int main() {
    const int N = 5;
    float h_in[N] = {-2, -1, 0, 1, 2}, h_out[N];
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);

    solve(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0;i<N;i++) printf("%f -> %f\n", h_in[i], h_out[i]);
    printf("test passed");
    cudaFree(d_in); cudaFree(d_out);
}