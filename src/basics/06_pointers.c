/** @file
 * @brief 指针、数组关系、限定符与函数指针。
 */
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

/**
 * @brief 交换两个整数对象的值。
 * @param left 第一个可写整数的地址；不转移所有权，可与 right 相同。
 * @param right 第二个可写整数的地址；不转移所有权。
 * @return 两个指针均有效时返回 true；有空指针时返回 false 且不修改数据。
 */
static bool swap(int *left, int *right)
{
    if (left == nullptr || right == nullptr) {
        return false;
    }

    int temporary = *left;
    *left = *right;
    *right = temporary;
    return true;
}

/**
 * @brief 累加整数数组并通过输出参数返回结果。
 * @param values 只读数组；length 大于零时须指向至少 length 个有效元素。
 * @param length 元素数量；可为零，此时 values 可为空。
 * @param result 可写结果地址；不得为空。数组元素和须能用 int 表示。
 * @return 参数满足约定时返回 true 并写入总和；否则返回 false 且不写 result。
 */
static bool sum(const int *values, size_t length, int *result)
{
    if (result == nullptr || (length > 0 && values == nullptr)) {
        return false;
    }

    int total = 0;
    for (size_t i = 0; i < length; ++i) {
        total += values[i];
    }

    *result = total;
    return true;
}

/**
 * @brief 返回两个整数之和，供函数指针示例使用。
 * @param left 第一个加数；本例保证结果不溢出 int。
 * @param right 第二个加数；本例保证结果不溢出 int。
 * @return 两个加数之和。
 */
static int add(int left, int right)
{
    return left + right;
}

/**
 * @brief 运行并验证指针的基本规则。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    int a = 1;
    int b = 2;
    int *pointer = &a; /* pointer 保存 a 的地址；*pointer 访问 a 本身。 */
    *pointer = 3;
    assert(a == 3);

    bool swapped = swap(&a, &b); /* & 取地址，函数通过指针修改调用者对象。 */
    assert(swapped);
    assert(a == 2 && b == 3);
    bool rejected_null = !swap(nullptr, &a); /* 空指针可比较，但不能解引用。 */
    assert(rejected_null);

    int values[] = {10, 20, 30};
    size_t length = sizeof values / sizeof values[0];
    int *cursor = values; /* 此处数组表达式转换为首元素地址。 */
    int (*whole_array_pointer)[3] = &values; /* 指向整个 int[3] 数组。 */
    const int *read_only_cursor = values; /* 不能通过此指针修改元素。 */
    int *end = values + length; /* 尾后指针可比较或相减，不能解引用。 */
    int total = 0;
    int empty_total = -1;

    assert(cursor == &values[0]);
    assert(*(cursor + 1) == values[1]); /* 下标访问等价于 *(cursor + 1)。 */
    assert((*whole_array_pointer)[1] == 20);
    bool summed_array = sum(values, length, &total);
    assert(summed_array && total == 60);
    bool summed_empty_array = sum(nullptr, 0, &empty_total);
    assert(summed_empty_array && empty_total == 0);
    assert(read_only_cursor[2] == 30 && end - cursor == 3);
    assert(sizeof values == 3 * sizeof values[0]);
    assert(sizeof cursor == sizeof(int *)); /* sizeof cursor 得到指针对象大小，不是数组长度。 */

    int *const fixed_pointer = &a; /* 指针本身固定，但可以修改它指向的对象。 */
    *fixed_pointer = 9;
    const int *const fixed_read_only = &b; /* 指针和所指对象都不能经此名修改。 */
    assert(a == 9 && *fixed_read_only == 3);

    int value = 41;
    int *value_pointer = &value;
    int **pointer_to_pointer = &value_pointer; /* 二级指针保存一级指针的地址。 */
    **pointer_to_pointer = 42;
    assert(value == 42);

    void *generic_pointer = &value; /* void * 可保存对象指针，但不能直接解引用。 */
    int *restored_pointer = generic_pointer; /* C 允许 void * 转回对象指针。 */
    assert(*restored_pointer == 42);

    int (*operation)(int, int) = add; /* 函数指针与对象指针是不同类型。 */
    assert(operation(2, 3) == 5);

    printf("指针：a 地址=%p，数组元素数=%zu，总和=%d\n", (void *)&a, length, total);
    return 0;
}
