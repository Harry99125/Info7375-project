To run this model, login to modal first, then goes to my cu directory, for example:C:\Users\yourname\xxx\project3
Then, run "modal run cuda_run.py --code-path code/run1.cu", it will run the code in "code" folder.
Since the llama2 modal is way too big (25g), I can't upload it to github, so I only use the stories15M from Karpathy.
If you have the converted llama2 model, you can go to "cuda_run.py" and change command line accordingly.
however, running 7b model is terrible slow, approximately less than 0.4 token/s

vllm vs original llama2.c vs cuda:
vllm 500 token/s running 7b chat model
original llama2.c running small story model:40 token/s
cuda version with matmul on gpu: 70token/s

