/** @file
 * @brief 泛型选择与可变参数。
 */
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

#define TYPE_NAME(value) _Generic((value), int: "int", double: "double", default: "其他")

/**
 * @brief 演示使用 va_list 累加可变参数。
 * @param count 后续整数的数量，须非负。
 * @param ... 恰好 count 个按默认提升规则传入的 int；总和须在 int 范围内。
 * @return 参数总和，count 为零时返回 0。
 */
static int sum_many(int count, ...)
{
    va_list arguments;
    va_start(arguments, count);
    int result = 0;
    for (int i = 0; i < count; ++i) {
        result += va_arg(arguments, int); /* 调用者必须传入匹配类型。 */
    }
    va_end(arguments);
    return result;
}

/**
 * @brief 运行并验证泛型选择与可变参数。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    assert(sum_many(0) == 0 && sum_many(3, 1, 2, 3) == 6);
    printf("泛型选择与可变参数：%s、%s\n", TYPE_NAME(1), TYPE_NAME(1.0));
    /* 可变参数中 float 提升为 double，窄整数通常提升为 int。
     * _Generic 是按类型选择表达式，不是 C++ 模板。 */
    return 0;
}
