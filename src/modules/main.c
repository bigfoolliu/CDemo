#include "arithmetic.h"

#include <assert.h>
#include <stdio.h>

int main(void)
{
    assert(arithmetic_add(2, 3) == 5);
    puts("多文件组织：头文件声明、源码定义与链接验证通过");
    return 0;
}
