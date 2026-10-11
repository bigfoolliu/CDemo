/** @file
 * @brief 基本类型、初始化与 sizeof。
 */
#include <assert.h>
#include <limits.h>
#include <stdio.h>

/**
 * @brief 运行并验证基本类型、初始化与 sizeof。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    /* 声明指定类型，初始化提供初始值；const 对象不可通过其名称修改。 */
    const int limit = 10;
    int count = 3;
    unsigned int positive = 4u;
    char letter = 'A';
    float fraction = 0.5f;
    double ratio = 1.5;
    bool enabled = true;
    count += 2;
    assert(count == 5 && count < limit && positive == 4u);
    assert(letter == 'A' && fraction == 0.5f && ratio == 1.5 && enabled);
    /* sizeof 的结果是 size_t，使用 %zu；字节不保证恰好是 8 位。 */
    printf("类型：int=%zu 字节，double=%zu 字节，每字节=%d 位\n", sizeof(int), sizeof(double),
           CHAR_BIT);
    return 0;
}
