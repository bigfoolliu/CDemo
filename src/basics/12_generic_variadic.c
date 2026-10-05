#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

#define TYPE_NAME(value) _Generic((value), int: "int", double: "double", default: "其他")

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

int main(void)
{
    assert(sum_many(0) == 0 && sum_many(3, 1, 2, 3) == 6);
    printf("泛型选择与可变参数：%s、%s\n", TYPE_NAME(1), TYPE_NAME(1.0));
    /* 可变参数中 float 提升为 double，窄整数通常提升为 int。
     * _Generic 是按类型选择表达式，不是 C++ 模板。 */
    return 0;
}
