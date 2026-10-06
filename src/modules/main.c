/** @file
 * @brief 多文件模块的声明、定义与链接。
 */
#include "arithmetic.h"

#include <assert.h>
#include <stdio.h>

/**
 * @brief 运行并验证多文件模块的声明、定义与链接。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    assert(arithmetic_add(2, 3) == 5);
    puts("多文件组织：头文件声明、源码定义与链接验证通过");
    return 0;
}
