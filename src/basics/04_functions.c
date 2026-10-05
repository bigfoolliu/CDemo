#include <assert.h>
#include <stdio.h>

static int add(int a, int b); /* 原型使调用前即可检查参数类型。 */

static int add(int a, int b)
{
    return a + b;
}

static unsigned int factorial(unsigned int n)
{
    /* 递归需要终止条件；此示例只使用不会溢出的小输入。 */
    return n <= 1 ? 1u : n * factorial(n - 1);
}

static void change_copy(int value)
{
    value = 99; /* C 的参数按值传递，这里仅修改局部副本。 */
    assert(value == 99);
}

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
