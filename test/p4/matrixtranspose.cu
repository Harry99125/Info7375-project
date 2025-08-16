#include <cstdio>
#include <cuda_runtime.h>
#define BLOCK_SIZE 16


__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j

    if (row < rows && col < cols) {
      
        output[col*rows + row] = input[row*cols  + col];
    }
}

void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
int main() {
    int rows=2, cols=3;
    float h_in[6] = {1,2,3,4,5,6}, h_out[6];
    float *d_in,*d_out;
    cudaMalloc(&d_in,sizeof(h_in));
    cudaMalloc(&d_out,sizeof(h_out));
    cudaMemcpy(d_in,h_in,sizeof(h_in),cudaMemcpyHostToDevice);

    solve(d_in,d_out,rows,cols);

    cudaMemcpy(h_out,d_out,sizeof(h_out),cudaMemcpyDeviceToHost);
    for(int i=0;i<cols*rows;i++) printf("%f ",h_out[i]); printf("\n");
    printf("test passed");
    cudaFree(d_in); cudaFree(d_out);
}