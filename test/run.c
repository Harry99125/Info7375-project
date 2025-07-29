// run_choice.c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int choice;

    printf("enter which project you want to test：");
    if (scanf("%d", &choice) != 1) {
        fprintf(stderr, "wuxiao。\n");
        return EXIT_FAILURE;
    }

    switch (choice) {
        case 4:
         
            if (system("./test") != 0) {
                fprintf(stderr, "no ./my_program\n");
                return EXIT_FAILURE;
            }
            break;
        case 2:
          
            printf("你输入了 2，执行另外的逻辑。\n");
            break;
        default:
            printf("未定义的选项：%d\n", choice);
            break;
    }

    return EXIT_SUCCESS;
}
