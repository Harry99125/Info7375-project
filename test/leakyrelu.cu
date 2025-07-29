#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h> 
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
    // 构造测试输入
    const int N = 16;
    float  h_input[N], h_output[N];
    for (int i = 0; i < N; ++i) {
        h_input[i] = float(i) - 8.0f;  // [-8, -7, ..., +7]
    }

    // 分配 GPU 内存
    float *d_input, *d_output;
    cudaMalloc(&d_input,  N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // 拷贝输入到 GPU
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 调用 solve（内部执行 kernel）
    solve(d_input, d_output, N);

    // 拷回结果
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印并校验
    printf(" idx |   input  |  output\n");
    printf("-----+----------+---------\n");
    for (int i = 0; i < N; ++i) {
        float in  = h_input[i];
        float out = h_output[i];
        float expect = (in < 0.f ? 0.01f * in : in);
        // 验证结果
        assert(fabs(out - expect) < 1e-6f);
        printf("%4d | %+7.3f | %+7.3f\n", i, in, out);
    }
    printf("\nAll results are correct! 🎉\n");

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}