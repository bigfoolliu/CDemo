/** @file
 * @brief 基本类型、初始化与 sizeof。
 */
/* 提供 assert 宏，用于检查程序运行时应当成立的条件。 */
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
    /* C23 将 bool、true 和 false 纳入语言；bool 表示真或假。 */
    bool enabled = true;
    count += 2;

    /*
     * assert 用来检查程序内部的预期：条件为真时继续运行；为假时打印
     * 条件和源码位置，并通过 abort() 终止程序，便于尽早发现错误。
     */
    assert(count == 5 && count < limit && positive == 4u);
    /* 这里检查字符、浮点数和 bool 的值是否符合预期。 */
    assert(letter == 'A' && fraction == 0.5f && ratio == 1.5 && enabled);

    /*
     * 若在包含 <assert.h> 前定义 NDEBUG，assert 会被关闭，表达式也不会
     * 执行；因此不要把修改变量或读取输入等必要操作写在断言里。assert
     * 用于检查程序预期，不适合代替对外部输入和运行时错误的处理。
     */

    /* sizeof 返回类型为 size_t 的字节数，使用 %zu；字节不保证恰好是 8 位。 */
    printf("类型：int=%zu 字节，double=%zu 字节，每字节=%d 位\n", sizeof(int), sizeof(double),
           CHAR_BIT);
    return 0;
}
