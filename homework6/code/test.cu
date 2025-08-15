#include <stdio.h>
#include <cuda_runtime.h>

// GPU 上的 kernel：把 *x 设为原值的平方
__global__ void test(int *x,int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N){
    x[i]+=1;
  }
}

int main() {
    int* arr=(int*)malloc(N*sizeof(int));
    for (int i = 0; i < N; i++) {
    arr[i] = i * 2;  
}
    int h = 5, *d;
    cudaMalloc(&d, sizeof(int));
    cudaMemcpy(d, &h, sizeof(int), cudaMemcpyHostToDevice);

    // 启动 1 个 block，1 个 thread
    square<<<1,1>>>(d);

    // 检查是否启动成功
    printf("launch error: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("result = %d\n", h);  // 应该打印 25

    cudaFree(d);
    return 0;
}