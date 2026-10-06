/** @file
 * @brief 预处理宏与内联函数。
 */
#include <assert.h>
#include <stdio.h>

#define ARRAY_LENGTH(array) (sizeof(array) / sizeof((array)[0]))
#define LABEL "预处理"

/**
 * @brief 计算整数平方，参数仅求值一次。
 * @param value 待平方的整数；平方须在 int 范围内。
 * @return value 的平方。
 * @note 预处理按记号替换，不进行函数调用的类型检查。
 *       宏参数即使加括号，也不能防止重复求值；计算优先使用函数。
 */
static inline int square(int value)
{
    return value * value;
}

/**
 * @brief 运行并验证预处理宏与内联函数。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
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
