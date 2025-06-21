#define TESTING
#include "run.c"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>   
#include <math.h>  
void rmsnorm(float* o, float* x, float* weight, int size);
void softmax(float* x, int size);
void matmul(float* xout, float* x, float* w, int n, int d);

void print_array(const char *name, const float *arr, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("% .6f%s", arr[i], i+1<n?", ":"");
    }
    printf("]\n");
}

void test_rmsnorm() {
    float x[4]      = {1,2,3,4};
    float weight[4] = {1,1,1,1};
    float out[4];

    rmsnorm(out, x, weight, 4);

 
    float rms = sqrtf((1+4+9+16)/4.0f + 1e-5f);
    float expect[4];
    for (int i = 0; i < 4; i++) {
        expect[i] = x[i] / rms;
    }

    
    printf("test_rmsnorm \n");
    print_array("  output  ", out,    4);
    print_array("  expect  ", expect, 4);
  

}

void test_softmax() {
    float orig[4] = {1,2,3,4};
    float x[4];
    memcpy(x, orig, sizeof(x));
    softmax(x, 4);
    float m = 4.0f;
    float sum = 0.0f, expv[4];
    for (int i = 0; i < 4; i++) {
        expv[i] = expf(orig[i] - m);
        sum += expv[i];
    }
    float expect[4];
    for (int i = 0; i < 4; i++) {
        expect[i] = expv[i] / sum;
    }
    printf("test_softmax \n");
    print_array("  output  ", x,      4);
    print_array("  expect  ", expect, 4);
   
  
}

void test_matmul() {
    float W[2*3] = {1,2,3, 4,5,6};
    float x[3]   = {1,1,1};
    float out[2] = {0,0};

    matmul(out, x, W, 3, 2);

    float expect[2] = {6.0f, 15.0f};

   
    printf("test_matmul \n");
    print_array("  output  ", out,    2);
    print_array("  expect  ", expect, 2);
   
  
}

void assert_eq(int a, int b) {
    if (a != b) {
        printf("Assertion failed: %d != %d\n", a, b);
        exit(EXIT_FAILURE);
    }
}

void test_prompt_encoding(Tokenizer* tokenizer, char* prompt, int* expected_tokens, int num_expected_tokens) {
    // encode
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    int num_prompt_tokens = 0; // the total number of prompt tokens
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    #if VERBOSITY == 1
    // print maybe
    printf("expected tokens:\n");
    for (int i = 0; i < num_expected_tokens; i++) printf("%d ", expected_tokens[i]);
    printf("\n");
    printf("actual tokens:\n");
    for (int i = 0; i < num_prompt_tokens; i++) printf("%d ", prompt_tokens[i]);
    printf("\n");
    #endif

    // verify
    assert_eq(num_prompt_tokens, num_expected_tokens);
    for (int i = 0; i < num_prompt_tokens; i++) {
        assert_eq(prompt_tokens[i], expected_tokens[i]);
    }

    #if VERBOSITY == 1
    printf("OK\n");
    printf("---\n");
    #endif
    free(prompt_tokens);
}

void test_prompt_encodings() {
    // let's verify that the Tokenizer works as expected

    char *tokenizer_path = "tokenizer.bin";
    int vocab_size = 32000;
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, vocab_size);

    // test 0 (test the empty string) (I added this as a simple case)
    char *prompt0 = "";
    int expected_tokens0[] = {1};
    test_prompt_encoding(&tokenizer, prompt0, expected_tokens0, sizeof(expected_tokens0) / sizeof(int));

    // the tests below are taken from the Meta Llama 2 repo example code
    // https://github.com/facebookresearch/llama/blob/main/example_text_completion.py
    // and the expected tokens come from me breaking in the debugger in Python

    // test 1
    char *prompt = "I believe the meaning of life is";
    int expected_tokens[] = {1, 306, 4658, 278, 6593, 310, 2834, 338};
    test_prompt_encoding(&tokenizer, prompt, expected_tokens, sizeof(expected_tokens) / sizeof(int));

    // test 2
    char* prompt2 = "Simply put, the theory of relativity states that ";
    int expected_tokens2[] = {1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871};
    test_prompt_encoding(&tokenizer, prompt2, expected_tokens2, sizeof(expected_tokens2) / sizeof(int));

    // test 3
    char* prompt3 = "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ";
    int expected_tokens3[] = {1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13, 13, 4706, 6324, 14332, 29892, 13, 13, 4706, 306, 925, 29871};
    test_prompt_encoding(&tokenizer, prompt3, expected_tokens3, sizeof(expected_tokens3) / sizeof(int));

    // test 4
    char* prompt4 = "Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>";
    int expected_tokens4[] = {1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 4706, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706, 923, 968, 1149};
    test_prompt_encoding(&tokenizer, prompt4, expected_tokens4, sizeof(expected_tokens4) / sizeof(int));

    // memory and file handles cleanup
    free_tokenizer(&tokenizer);
}

int main(int argc, char *argv[]) {
    test_prompt_encodings();
    test_rmsnorm();
    test_softmax();
    test_matmul();
    printf("ALL OK\n");
}
