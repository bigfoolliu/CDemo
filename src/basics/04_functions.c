/** @file
 * @brief 函数原型、递归与按值传参。
 */
#include <assert.h>
#include <stdio.h>

/**
 * @brief 声明整数加法接口。
 * @param a 第一个加数。
 * @param b 第二个加数；两数之和须在 int 范围内。
 * @return 两个加数的和。
 */
static int add(int a, int b); /* 原型使调用前即可检查参数类型。 */

/**
 * @brief 演示带原型的整数加法。
 * @param a 第一个加数。
 * @param b 第二个加数；结果须在 int 范围内。
 * @return a 与 b 的和。
 */
static int add(int a, int b)
{
    return a + b;
}

/**
 * @brief 递归计算阶乘。
 * @param n 非负整数；调用者须保证阶乘能用 unsigned int 表示。
 * @return n 的阶乘，0 的阶乘为 1。
 */
static unsigned int factorial(unsigned int n)
{
    /* 递归需要终止条件；此示例只使用不会溢出的小输入。 */
    return n <= 1 ? 1u : n * factorial(n - 1);
}

/**
 * @brief 演示按值传参不会修改调用者变量。
 * @param value 待修改的整数副本。
 */
static void change_copy(int value)
{
    value = 99; /* C 的参数按值传递，这里仅修改局部副本。 */
    assert(value == 99);
}

/**
 * @brief 运行并验证函数原型、递归与按值传参。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    int original = 7;
    change_copy(original);

    int (*operation)(int, int) = add; /* 函数指针可用于回调。 */
    assert(original == 7 && operation(3, 4) == 7);

    assert(factorial(0) == 1 && factorial(5) == 120);

    puts("函数：原型、按值传参、递归、函数指针验证通过");

    return 0;
}
