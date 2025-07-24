// run_temp.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
    if (argc < 1) {
        fprintf(stderr, "Usage: %s [test_file.cu ...]\n", argv[0]);
        return 1;
    }

    // 1) 创建临时目录
    char template[] = "/tmp/cuda_tmpXXXXXX";
    char *tmpdir = mkdtemp(template);
    if (!tmpdir) {
        perror("mkdtemp");
        return 1;
    }


    // 3) 额外拷贝 homework4/matmul.c
    const char *extra_src = "../homework4/matmul.c";
    if (access(extra_src, R_OK) == 0) {
        char cmd2[PATH_MAX];
        snprintf(cmd2, sizeof(cmd2),
                 "cp \"%s\" \"%s/\"",
                 extra_src, tmpdir);
        system(cmd2);
    } else {
        fprintf(stderr, "Warning: cannot find %s\n", extra_src);
    }

    // 4) 切换到临时目录
    char oldcwd[PATH_MAX];
    getcwd(oldcwd, sizeof(oldcwd));
    chdir(tmpdir);

    // 5) 编译所有 .cu 文件
    if (system("nvcc *.cu -o temp_exec") != 0) {
        fprintf(stderr, "nvcc compile failed\n");
    } else {
        // 6) 运行
        system("./temp_exec");
    }

    // 7) 清理
    chdir(oldcwd);
    char cleancmd[PATH_MAX];
    snprintf(cleancmd, sizeof(cleancmd),
             "rm -rf \"%s\"", tmpdir);
    system(cleancmd);

    return 0;
}
