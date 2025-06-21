import modal

app = modal.App("cuda-nvcc-exec")

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Docker image with nvcc and CUDA
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
        .add_local_dir("code", remote_path="/root/code")
       
)

@app.function(image=image, gpu="A10G", timeout=300)
def compile_and_run_cuda(code_path: str):
    import subprocess

    file_path = f"/root/{code_path}"

    # Compile
   #compile_result = subprocess.run(
    #    ["nvcc", file_path, "-O3", "-Xcompiler", "-fopenmp", "-o", "llama2", "-lcublas"], 
   #     capture_output=True, text=True
   # )
    compile_result = subprocess.run(
    [
        "nvcc",
        "-DUSE_CUDA",
        "-O3",
        "-o", "llama2",
        file_path,      
        "-lm",
        "-lcublas"
    ],
    capture_output=True,
    text=True
)
    if compile_result.returncode != 0:
        print("Compilation failed:")
        print(compile_result.stderr)
        return

    # Run
    run_result = subprocess.run(["./llama2",
        f"/root/code/stories15M.bin",       
        "-z", f"/root/code/tokenizer.bin",  
        "-n", "0",
        "-m", "generate",
        "-i", "Lilly is a good girl"], capture_output=True, text=True)
    print("CUDA program output:")
    print(run_result.stdout)
    print(run_result.stderr)