import modal

app = modal.App("cuda-nvcc-exec")

cuda_version = "12.8.0"      # 不要高于宿主机的 CUDA 版本
flavor      = "devel"        # 包含完整的 CUDA toolkit
operating_sys= "ubuntu22.04"
tag         = f"{cuda_version}-{flavor}-{operating_sys}"

# 基于官方 nvidia/cuda 镜像，同时把本地 code 目录挂到 /root/code
image = (
    modal.Image
      .from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
      .add_local_dir("code", remote_path="/root/code")
)

@app.function(image=image, gpu="A10G", timeout=300)
def compile_and_run_cuda(code_file: str):
    import subprocess
    import os

    # code_file 是相对于 code 目录的文件名，比如 "leakyrelu.cu"
    src_path = f"/root/code/{code_file}"
    bin_path = "/root/code/a.out"

    # 1) 编译：调用 nvcc
    compile_cmd = [
        "nvcc", src_path,
        "-O3",
        "-o", bin_path,
    ]
    print("Compiling with:", " ".join(compile_cmd))
    comp = subprocess.run(compile_cmd, capture_output=True, text=True)
    if comp.returncode != 0:
        print("❌ Compilation failed:")
        print(comp.stderr)
        return

    # 2) 运行
    print("✅ Compiled successfully, running …")
    run = subprocess.run([bin_path], capture_output=True, text=True)
    print("---- stdout ----")
    print(run.stdout)
    print("---- stderr ----")
    print(run.stderr)
    if run.returncode != 0:
        print(f"❌ Program exited with code {run.returncode}")
    else:
        print("🎉 Program ran successfully")
