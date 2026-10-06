/** @file
 * @brief 学习分配失败检查、扩容和内存所有权。
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
 * 本文件已演示：malloc、calloc、realloc、free，使用 nullptr 检查失败；
 * 借助 <stdint.h> 的 SIZE_MAX 检查分配大小，并使用退出状态宏。
 * 本文件未演示：其他通用工具；本专区 04_conversion.c 演示转换，07_sort_search.c 演示排序搜索。
 * 观察重点：谁拥有内存、何时初始化、realloc 失败后谁负责释放原分配。
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief 分配整数数组并以递增序列初始化。
 * @param count 所需元素数；零或大小乘法溢出时拒绝分配。
 * @return 成功返回由调用者 free 的数组，失败返回 nullptr。
 */
static int *allocate_sequence(size_t count)
{
    if (count == 0 || count > SIZE_MAX / sizeof(int) || count > 1000) {
        return nullptr; /* 示例限制输入规模，使索引转换为 int 安全。 */
    }
    int *values = malloc(count * sizeof(*values));
    if (values != nullptr) {
        for (size_t i = 0; i < count; ++i) {
            values[i] = (int)i;
        }
    }
    return values;
}

/** @brief 验证 malloc、calloc、realloc 和 free。 @return 成功为 0，分配失败为非零。 */
int main(void)
{
    int *empty = allocate_sequence(0);
    int *overflow = allocate_sequence(SIZE_MAX);
    assert(empty == nullptr && overflow == nullptr);
    int *values = allocate_sequence(3);
    if (values == nullptr) {
        fputs("malloc 分配失败。\n", stderr);
        return EXIT_FAILURE;
    }
    int *grown = realloc(values, 5 * sizeof(*values));
    if (grown == nullptr) {
        free(values); /* 失败时旧分配仍归调用者所有。 */
        fputs("realloc 扩容失败。\n", stderr);
        return EXIT_FAILURE;
    }
    values = grown; /* 成功后不再读取旧指针，即使地址看起来相同。 */
    values[3] = 3;
    values[4] = 4;
    assert(values[0] == 0 && values[2] == 2 && values[4] == 4);
    free(values);
    values = nullptr;
    free(values); /* free(nullptr) 是合法空操作。 */

    int *zeros = calloc(3, sizeof(*zeros));
    if (zeros == nullptr) {
        fputs("calloc 分配失败。\n", stderr);
        return EXIT_FAILURE;
    }
    assert(zeros[0] == 0 && zeros[2] == 0);
    /* calloc 清零字节；不能推广为浮点数或指针的语义零。 */
    free(zeros);
    puts("memory：分配边界、扩容保留数据、字节清零与释放验证通过");
    return EXIT_SUCCESS;
}
