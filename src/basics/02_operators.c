/** @file
 * @brief 基本运算符与类型转换。
 */
#include <assert.h>
#include <stdio.h>

/**
 * @brief 运行并验证基本运算符与类型转换。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    int a = 7;
    int b = 2;
    assert(a + b == 9 && a - b == 5 && a * b == 14);
    assert(a / b == 3 && a % b == 1); /* 整数除法向零截断。 */
    assert((double)a / b == 3.5);     /* 强制转换改变参与运算的类型。 */

    unsigned int bits = 5u;
    assert((bits & 1u) == 1u && (bits | 2u) == 7u);
    assert((bits ^ 1u) == 4u && (bits << 1) == 10u && (bits >> 1) == 2u);
    assert((~bits & 7u) == 2u);

    int changes = 0;
    bool allowed = false;
    if (allowed && ++changes > 0) {
        puts("不会执行");
    }
    assert(changes == 0); /* && 与 || 短路；按位 & 与 | 不短路。 */

    int maximum = a > b ? a : b;
    int previous = a++; /* 后缀产生旧值，前缀 ++a 产生新值。 */
    assert(maximum == 7 && previous == 7 && a == 8 && !allowed);
    puts("运算符：算术、比较、逻辑、位运算、赋值与转换验证通过");

    return 0;
}
