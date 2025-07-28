/* Inference for Llama-2 Transformer model in pure C
 * With added CUDA support initially drawing from
 * https://github.com/ankan-ban/llama2.cu/blob/master/llama2.cu
 * and structured in a way that hopefully makes keeping it
 * up-to-date straightforward.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <float.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
 #define CUCHK(err) cuda_check((err), __FILE__, __LINE__)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

cublasHandle_t g_cublas_handle = nullptr;

void create_cublas_handle() {
    cublasStatus_t stat = cublasCreate(&g_cublas_handle);  // FIXME cublasDestroy
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
}
void destroy_cublas_handle() {
    cublasStatus_t stat = cublasDestroy(g_cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
}
#endif
  
typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;
      typedef struct {
     float* token_embedding_table;    // (vocab_size, dim)
     float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
     float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
     float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
     float* rms_final_weight; // (dim,)
     float* wcls;
} TransformerWeights;
     typedef struct {
     float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
#ifdef USE_CUDA
    float *logits_gpu; // output logits in GPU
#endif
    float *logits; // output logits in CPU
     float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
     int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

#ifdef USE_CUDA
void malloc_run_state(RunState* s, Config* p) {
     int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    CUCHK(cudaMalloc((void**)&s->x, p->dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->xb, p->dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->xb2, p->dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->q, p->dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->att, p->n_heads * p->seq_len * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(float)));
    s->logits = (float *)calloc(p->vocab_size, sizeof(float));
     if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits_gpu || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}
#else
void malloc_run_state(RunState* s, Config* p) {
     int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float *)calloc(p->dim, sizeof(float));
    s->xb = (float *)calloc(p->dim, sizeof(float));
    s->xb2 = (float *)calloc(p->dim, sizeof(float));
    s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
    s->q = (float *)calloc(p->dim, sizeof(float));
    s->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float *)calloc(p->vocab_size, sizeof(float));
     if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}
#endif

#ifdef USE_CUDA
void free_run_state(RunState* s) {
    CUCHK(cudaFree(s->x));
    CUCHK(cudaFree(s->xb));
    CUCHK(cudaFree(s->xb2));
    CUCHK(cudaFree(s->hb));
    CUCHK(cudaFree(s->hb2));
    CUCHK(cudaFree(s->q));
    CUCHK(cudaFree(s->att));
    CUCHK(cudaFree(s->logits_gpu));
    free(s->logits);
    CUCHK(cudaFree(s->key_cache));
    CUCHK(cudaFree(s->value_cache));
}
#else
void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}
#endif

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
     unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
     if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
     int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
     fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
     *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
#ifdef USE_CUDA
       float* weights_ptr;
    size_t weights_size = *file_size - sizeof(Config);
    CUCHK(cudaMalloc((void**)&weights_ptr, weights_size));
    CUCHK(cudaMemcpy(weights_ptr, *data + sizeof(Config)/sizeof(float), weights_size, cudaMemcpyHostToDevice));
#else
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
#endif
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
     read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
     malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
     if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
#ifdef USE_CUDA
      CUCHK(cudaFree(t->weights.token_embedding_table));
#endif
     free_run_state(&t->state);
}
  
#ifdef USE_CUDA
 int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

const int num_threads_lrg = 1024;
const int num_threads_med = 256;

__global__ void rmsnorm_kernel(float* o, float* x, float* weight, int size, int elementsPerThread) {
     float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_lrg;
        if (j < size)
            ss += x[j] * x[j];
    }
    using BlockReduce = cub::BlockReduce<float, num_threads_lrg>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss);
     __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;
     for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_lrg;
        if (j < size) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }
}
void rmsnorm(float* o, float* x, float* weight, int size) {
    int elementsPerThread = divUp(size, num_threads_lrg);
    rmsnorm_kernel <<<1, num_threads_lrg >>> (o, x, weight, size, elementsPerThread);
}
#else
void rmsnorm(float* o, float* x, float* weight, int size) {
     float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
     for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}
#endif

#ifdef USE_CUDA
__global__ void softmax_kernel(float* x, int size,float max_val,float sum) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    x[i]=expf(x[i]-max_val);
    sum+=x[i];
}
__global__ void norm_kernel(float* x, int size,float max_val,float sum) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    x[i]/=sum;
}
__device__ void softmax_gpu(float* x, int size) {
   
    if (threadIdx.x == 0) {
    
        float max_val = -FLT_MAX;
        for (int i = 0; i < size; ++i) {
            max_val = fmaxf(max_val, x[i]);
        }
    
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            float v = expf(x[i] - max_val);
            x[i] = v;
            sum += v;
        }
    
        for (int i = 0; i < size; ++i) {
            x[i] /= sum;
        }
    }

    __syncthreads();
}
#endif
void softmax(float* x, int size) {
     float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
     float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
     for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}


__global__ void matmul_kernel(float* xout, const float* x, const float* w, int n, int d) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;

    float val = 0.0f;
    for (int j = 0; j < n; j++) {
        val += w[i * n + j] * x[j];  
    }
    xout[i] = val;
}
 void matmul(float* xout, float* x, float* w, int n, int d) {
 



    
    int block = 32;
    int grid = (d + block - 1) / block;
    matmul_kernel<<<grid, block>>>(xout, x, w, n, d);
    CUCHK(cudaGetLastError());
    CUCHK(cudaDeviceSynchronize());
 
    
 
}
 #ifdef USE_CUDA
__global__ void RoPe_rotation_kernel(int pos, float *sq, float *sk, int kv_dim, int head_size) {
    int i = threadIdx.x * 2;
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
    for (int v = 0; v < rotn; v++) {
        float* vec = v == 0 ? sq : sk; // the vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i+1] = v0 * fci + v1 * fcr;
    }
}
void RoPe_rotation(int pos, RunState* s, int dim, int kv_dim, int head_size) {
    RoPe_rotation_kernel <<<1, dim/2 >>> (pos, s->q, s->k, kv_dim, head_size);
}
#else
void RoPe_rotation(int pos, RunState* s, int dim, int kv_dim, int head_size) { //s->q, s->k, freq_cis_real_row, freq_cis_imag_row, p->n_heads, head_size) {
    for (int i = 0; i < dim; i+=2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}
#endif
/*

#define TILE_K 8                             


template<int HEAD_SIZE>
__global__ void multi_head_kernel(int  pos, float*  q_all,int sq, float* out_all,  float*  k_cache,float* v_cache,
        int kv_dim, int kv_mul,int head_size, int loff)
{

    int h  = blockIdx.x;                
    int ld = threadIdx.x;                 
    int kt = threadIdx.y;                

 
    extern __shared__ float sm[];
    float* s_k = sm;                                         
    float* s_v = s_k + TILE_K * head_size;
     __shared__ float s_q[HEAD_SIZE];
     float* q = q_all + h * head_size;
    if (ld <  head_size) s_q[ld] = q[ld];
    __syncthreads();

 
    float m_prev = -FLT_MAX, l_prev = 0.f, acc = 0.f;

 
    for (int t0 = 0; t0 <= pos; t0 += TILE_K) {
        int t = t0 + kt;                                

    
        if (t <= pos && ld < head_size) {
            const float* k_glb = k_cache + loff + t*kv_dim + (h/kv_mul)* head_size;
            const float* v_glb = v_cache + loff + t*kv_dim + (h/kv_mul)* head_size;
            s_k[kt* head_size + ld] = k_glb[ld];
            s_v[kt* head_size + ld] = v_glb[ld];
        }
        __syncthreads();

        if (t <= pos && ld <  head_size) {
            float score = 0.f;
          
            for (int i = 0; i < head_size; ++i)
                score += s_q[i] * s_k[kt * head_size + i];
            score *= rsqrtf( head_size);            
            float m_new = fmaxf(m_prev, score);
            float l_new = __expf(m_prev - m_new) * l_prev +
                          __expf(score   - m_new);
            float coef_p = __expf(m_prev - m_new) / l_new;
            float coef_n = __expf(score   - m_new) / l_new;

            float v_val = s_v[kt* head_size + ld];
            acc = acc * coef_p + v_val * coef_n;

            m_prev = m_new;
            l_prev = l_new;
        }
        __syncthreads();
    }


    if (ld <  head_size) out_all[h *  head_size + ld] = acc;
}


inline void multi_head_attention(
        int pos, Config* p, RunState* s,
        int kv_dim, int kv_mul, int head_size, int loff)
{
    constexpr int HS =128;                   
   

    dim3 block(HS, TILE_K);                      
    dim3 grid(p->n_heads);
    size_t shmem = 2 * TILE_K * HS * sizeof(float); 

   multi_head_kernel<HS><<<grid, block, shmem>>>(pos,s->q, p->seq_len, s->xb,  s->key_cache, s->value_cache,kv_dim, kv_mul,head_size, loff);

}
*/
 __global__ void multi_head_attention_kernel(int pos, int seq_len, float *sq, float *satt, float *sxb, float *key_cache, float *value_cache, int kv_dim, int kv_mul, int head_size, int loff) {
    int h = blockIdx.x;
extern __shared__ float sm[]; 
float* q = sm;                         
float* s_k = sm + head_size;              

         
int ld = threadIdx.x;              
int kt = threadIdx.y;      

const float* g_q = sq + h * head_size;
const float* ta = satt + h * seq_len;
 float* xb = sxb + h * head_size;
if (ld < head_size){            
    q[ld] = g_q[ld];
}
int Tile=blockDim.y;
__syncthreads();  
 float* att = satt + h * seq_len;
for (int t0 = 0; t0 <= pos; t0 += blockDim.y) {
    int t = t0 + kt; 

   
    if (t <= pos && ld < head_size) {
        int base = loff + t * kv_dim + h * head_size;
        s_k[kt*head_size + ld] = key_cache[base + ld];
   
    }
   __syncthreads();  

  if (t <= pos && ld == 0) {
  float score = 0.f;
  for (int i = 0; i < head_size; ++i) {
    score += q[i] * s_k[i];
  }
  score *= rsqrtf((float)head_size);
  att[t] = score;
}
    
}
/*
   for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
          float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            
            score +=q[i] * s_k[i];
        }
        score /= sqrtf(head_size);
         att[t] = score;
    }

*/
    softmax_gpu(att, pos + 1);
    __syncthreads();


   
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float* v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
             float a = att[t];
            val += a *v[i];
        }
        xb[i] = val;
    }
}
void multi_head_attention(int pos, Config* p, RunState* s, int kv_dim, int kv_mul, int head_size, int loff) {
    
    multi_head_attention_kernel <<<p->n_heads, num_threads_lrg>>> (pos, p->seq_len, s->q, s->att, s->xb, s->key_cache, s->value_cache, kv_dim, kv_mul, head_size, loff);
}

__global__ void f_silu_elementwise_mul_w3_kernel(float *shb, float *shb2, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_dim) {
        float val = shb[i];
    
        val *= (1.0f / (1.0f + expf(-val)));
    
        val *= shb2[i];
        shb[i] = val;
    }
}
void f_silu_elementwise_mul_w3(RunState *s, int hidden_dim) {
    f_silu_elementwise_mul_w3_kernel<<<divUp(hidden_dim, num_threads_med), num_threads_med>>>(s->hb, s->hb2, hidden_dim);
}



#ifdef USE_CUDA
__global__ void accum_kernel(float* a, float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        a[i] += b[i];
    }
}
void accum(float *a, float *b, int size) {
    accum_kernel<<<divUp(size, num_threads_med), num_threads_med>>>(a,b,size);
}
#else
void accum(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}
#endif

float* forward(Transformer* transformer, int token, int pos) {
     Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
     float* content_row = w->token_embedding_table + token * dim;
#ifdef USE_CUDA
    CUCHK(cudaMemcpy(x, content_row, dim*sizeof(*x), cudaMemcpyHostToDevice));
#else
    memcpy(x, content_row, dim*sizeof(*x));
#endif
     for(unsigned long long l = 0; l < p->n_layers; l++) {
         rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
         int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;
         matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
         RoPe_rotation(pos, s, dim, kv_dim, head_size);
         multi_head_attention(pos, p, s, kv_dim, kv_mul, head_size, loff);
         matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
         accum(x, s->xb2, dim);
         rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
          matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
         f_silu_elementwise_mul_w3(s, hidden_dim);
         matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
         accum(x, s->xb, dim);
    }
     rmsnorm(x, x, w->rms_final_weight, dim);
 #ifdef USE_CUDA
    matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);
    CUCHK(cudaMemcpy(s->logits, s->logits_gpu, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
#else
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
#endif 
    return s->logits;
}
  
typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
     t->vocab_size = vocab_size;
     t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
     FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
     if (prev_token == 1 && piece[0] == ' ') { piece++; }
      unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
      if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
 #if defined USE_CUDA && defined _WIN32
     TokenIndex tok;
    tok.str = str;
#else
    TokenIndex tok = { .str = str }; // acts as the key to search for
#endif
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
      if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
         t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
      char* str_buffer = (char *)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;
     *n_tokens = 0;
     if (bos) tokens[(*n_tokens)++] = 1;
        if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char *)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }


    for (char *c = text; *c != '\0'; c++) {
             if ((*c & 0xC0) != 0x80) {
              str_len = 0;
        }
         str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';
          if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }
         int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
             tokens[(*n_tokens)++] = id;
        } else {
               for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }
     while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
             sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                 best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }
         tokens[best_idx] = best_id;
         for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }
     if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}
   
typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
     int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
      float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    
    int n0 = 0;
       const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);
     float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }
     float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
     sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
     *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
     int next;
    if (sampler->temperature == 0.0f) {
         next = sample_argmax(logits, sampler->vocab_size);
    } else {
         for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
         softmax(logits, sampler->vocab_size);
         float coin = random_f32(&sampler->rng_state);
         if (sampler->topp <= 0 || sampler->topp >= 1) {
             next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
             next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}
  
long time_in_ms() {
     struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}
  
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = (char *)"";
    if (prompt == NULL) { prompt = empty_prompt; }
     int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }
     long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {
         float* logits = forward(transformer, token, pos);
         if (pos < num_prompt_tokens - 1) {
             next = prompt_tokens[pos + 1];
        } else {
             next = sample(sampler, logits);
        }
        pos++;
         if (next == 1) { break; }
         char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;
         if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");
     if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
     printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}
     
void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {
      char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;
     int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {
         if (user_turn) {
             if (pos == 0) {
                 if (cli_system_prompt == NULL) {
                     read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                     strcpy(system_prompt, cli_system_prompt);
                }
            }
             if (pos == 0 && cli_user_prompt != NULL) {
                 strcpy(user_prompt, cli_user_prompt);
            } else {
                 read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
             if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
             encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }
         if (user_idx < num_prompt_tokens) {
             token = prompt_tokens[user_idx++];
        } else {
             token = next;
        }
         if (token == 2) { user_turn = 1; }
         float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
             char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}
  #ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
     char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = (char *)"tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = (char *)"generate";    // generate|chat
    char *system_prompt = (char *)NULL; // the (optional) system prompt to use in chat mode
     if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
         if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
         if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }
     if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;
     Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length
     Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
     Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

#ifdef USE_CUDA
    create_cublas_handle();
#endif
     if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }
     free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
#ifdef USE_CUDA
    destroy_cublas_handle();
#endif
    return 0;
}
#endif