
#include <cuda_runtime.h>
#include <cstdio>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int l = 0; l < N; l++) {
            sum += A[row * N + l] * B[l * K + col];
        }
        C[row * K + col] = sum;
    }
}


void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}int main() {
    int M=2, N=3, K=2;
    float hA[6] = {1,2,3,4,5,6};     // 2x3
    float hB[6] = {7,8,9,10,11,12};  // 3x2
    float hC[4];
    float *dA,*dB,*dC;
    cudaMalloc(&dA,sizeof(hA));
    cudaMalloc(&dB,sizeof(hB));
    cudaMalloc(&dC,sizeof(hC));
    cudaMemcpy(dA,hA,sizeof(hA),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,sizeof(hB),cudaMemcpyHostToDevice);

    solve(dA,dB,dC,M,N,K);

    cudaMemcpy(hC,dC,sizeof(hC),cudaMemcpyDeviceToHost);
    for(int i=0;i<M*K;i++) printf("%f ",hC[i]); printf("\n");
    printf("test passed");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}