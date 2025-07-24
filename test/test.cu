#include <cstdio>
#include <cuda_runtime.h>

// 直接声明外部函数，不需要头文件
extern void solve(const float* A, const float* B, float* C,
                  int M, int N, int K);

int main(){
    const int M=2,N=3,K=2;
    float A[M*N] = {1,2,3, 4,5,6};
    float B[N*K] = {7,8, 9,10, 11,12};
    float C[M*K];

    float *dA,*dB,*dC;
    cudaMalloc(&dA, M*N*sizeof(float));
    cudaMalloc(&dB, N*K*sizeof(float));
    cudaMalloc(&dC, M*K*sizeof(float));
    cudaMemcpy(dA, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N*K*sizeof(float), cudaMemcpyHostToDevice);

    // 调用 homework4/matmul.cu 里的 solve
    solve(dA, dB, dC, M, N, K);

    cudaMemcpy(C, dC, M*K*sizeof(float), cudaMemcpyDeviceToHost);
    printf("C =\n");
    for(int i=0;i<M;i++){
      for(int j=0;j<K;j++) printf("%4.1f ", C[i*K+j]);
      printf("\n");
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}