#include <assert.h>
#include <stdio.h>

static void swap(int *left, int *right)
{
    int temporary = *left; /* * 解引用；调用者必须提供有效地址。 */
    *left = *right;
    *right = temporary;
}

static int sum(const int *values, size_t length)
{
    int result = 0;
    for (size_t i = 0; i < length; ++i) {
        result += values[i];
    }
    return result;
}

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
