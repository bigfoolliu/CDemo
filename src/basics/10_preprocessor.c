#include <assert.h>
#include <stdio.h>

#define ARRAY_LENGTH(array) (sizeof(array) / sizeof((array)[0]))
#define LABEL "预处理"

/* 预处理按记号替换，不进行函数调用的类型检查。
 * 宏参数即使加括号，也不能防止重复求值；计算优先使用函数。 */
static inline int square(int value)
{
    return value * value;
}

int main(void)
{
    int values[] = {1, 2, 3};
    assert(ARRAY_LENGTH(values) == 3); /* 此宏不能用于数组退化后的指针。 */
    int value = 3;
    assert(square(value++) == 9 && value == 4);
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
    puts(LABEL "：当前编译器声明采用 C23 标准版本");
#else
    puts(LABEL "：当前编译器尚未声明正式 C23 版本值");
#endif
    return 0;
}
