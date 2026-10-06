/** @file
 * @brief 使用相同的比较规则排序和二分查找整数。
 *
 * 学习范围：<stdlib.h> 通用工具库（按用途列出主要接口，不含可选扩展）。
 * 接口概览：
 * - 数值转换：atoi/atol/atoll/atof、strtol/strtoll/strtoul/strtoull、
 *   strtof/strtod/strtold、strfromf/strfromd/strfroml（C23）。
 * - 内存管理：malloc、calloc、realloc、free、aligned_alloc；
 *   free_sized/free_aligned_sized（C23，平台支持需单独验证）。
 * - 排序搜索：qsort、bsearch；整数算术：abs/labs/llabs、div/ldiv/lldiv。
 * - 随机序列：rand、srand；环境交互：getenv、system。
 * - 退出管理：exit、_Exit、abort、atexit、quick_exit、at_quick_exit。
 * - 多字节转换：mblen、mbtowc、wctomb、mbstowcs、wcstombs。
 * - C23 同步/对齐接口：call_once、memalignment；依赖实现提供。
 * - 类型与宏：size_t、wchar_t、div_t/ldiv_t/lldiv_t、once_flag、
 *   NULL、EXIT_SUCCESS/EXIT_FAILURE、RAND_MAX、MB_CUR_MAX。
 * 本文件已演示：qsort、bsearch，借助 <limits.h> 的 INT_MIN/INT_MAX 验证极值比较。
 * 本文件未演示：通用工具库的其他分类；内存与转换分别见 03_memory.c、04_conversion.c。
 * 观察重点：比较器返回顺序关系；搜索数组须按相同规则排序；重复值位置不保证。
 */
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief 按整数值升序比较，避免相减导致有符号溢出。
 * @param left 第一个只读 int 对象地址，查找时是 key 地址。
 * @param right 第二个只读 int 对象地址。
 * @return 负数、零、正数分别表示小于、等于、大于。
 */
static int compare_ints(const void *left, const void *right)
{
    int a = *(const int *)left;
    int b = *(const int *)right;
    return (a > b) - (a < b);
}

/** @brief 验证极值、重复元素、查找命中与未命中。 @return 验证完成返回 0。 */
int main(void)
{
    int values[] = {3, INT_MAX, -1, 3, INT_MIN};
    size_t count = sizeof(values) / sizeof(values[0]);
    qsort(values, count, sizeof(values[0]), compare_ints);
    assert(values[0] == INT_MIN && values[count - 1] == INT_MAX);
    for (size_t i = 1; i < count; ++i) {
        assert(values[i - 1] <= values[i]);
    }
    int key = 3;
    const int *found = bsearch(&key, values, count, sizeof(values[0]), compare_ints);
    assert(found != nullptr && *found == 3); /* 重复值的具体位置不保证。 */
    key = 2;
    found = bsearch(&key, values, count, sizeof(values[0]), compare_ints);
    assert(found == nullptr);
    int singleton[] = {7};
    qsort(singleton, 1, sizeof(singleton[0]), compare_ints);
    found = bsearch(&key, singleton, 0, sizeof(singleton[0]), compare_ints);
    assert(found == nullptr); /* 零元素仍传有效的 key 和 base 指针。 */
    puts("algorithms：安全比较、排序、重复值与二分搜索验证通过");
    return 0;
}
