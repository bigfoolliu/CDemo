/** @file
 * @brief 指针、解引用与指针运算。
 */
#include <assert.h>
#include <stdio.h>

/**
 * @brief 交换两个整数对象的值。
 * @param left 可读写的第一个整数地址。
 * @param right 可读写的第二个整数地址；允许与 left 相同。
 */
static void swap(int *left, int *right)
{
    int temporary = *left; /* * 解引用；调用者必须提供有效地址。 */
    *left = *right;
    *right = temporary;
}

/**
 * @brief 累加整数数组。
 * @param values 包含 length 个元素的只读数组；length 为零时可为空。
 * @param length 元素数量；累加结果须在 int 范围内。
 * @return 元素总和，空数组返回 0。
 */
static int sum(const int *values, size_t length)
{
    int result = 0;
    for (size_t i = 0; i < length; ++i) {
        result += values[i];
    }
    return result;
}

/**
 * @brief 运行并验证指针、解引用与指针运算。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    int a = 1;
    int b = 2;
    swap(&a, &b); /* & 获取地址，指针参数允许修改调用者的对象。 */

    int values[] = {10, 20, 30};
    const int *cursor = values; /* 数组在多数表达式中转换为首元素指针。 */
    assert(*(cursor + 1) == 20 && sum(values, 3) == 60);

    const int *end = values + 3; /* 尾后指针可用于比较，不能解引用。 */
    assert(end - cursor == 3 && a == 2 && b == 1);

    int *const fixed = &a; /* 指针自身不可改；const int * 则限制通过指针改值。 */
    *fixed = 9;
    assert(a == 9);

    printf("指针：a 的地址=%p\n", (void *)&a);

    return 0;
}
