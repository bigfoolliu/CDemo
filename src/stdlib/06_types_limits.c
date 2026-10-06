/** @file
 * @brief 常用库类型、整数范围和可移植格式宏。
 *
 * 学习范围：类型、整数格式与数值范围；这里的 API 主要是类型和宏，不是函数。
 * 接口概览（按头文件列出代表性名称与接口家族）：
 * - <stddef.h>：size_t、ptrdiff_t、wchar_t、max_align_t、offsetof、NULL；nullptr_t（C23）。
 * - <stdint.h>：intN_t/uintN_t（实现提供时）、int_leastN_t/uint_leastN_t、
 *   int_fastN_t/uint_fastN_t、intmax_t/uintmax_t、intptr_t/uintptr_t（实现提供时），
 *   对应的 MIN/MAX、INTN_C/UINTN_C 和 SIZE_MAX 等宏；C23 增加位宽宏。
 * - <inttypes.h>：PRI* 和 SCN* 格式宏；imaxabs、imaxdiv、strtoimax、strtoumax、
 *   wcstoimax、wcstoumax，以及 imaxdiv_t 类型。
 * - <limits.h>：CHAR_BIT、CHAR_MIN/MAX、SCHAR_MIN/MAX、UCHAR_MAX、
 *   SHRT/INT/LONG/LLONG 等整数范围宏、MB_LEN_MAX；C23 增加整数位宽宏。
 * - <float.h>：FLT/DBL/LDBL 的 MIN/MAX/EPSILON/DIG/MANT_DIG 等宏，
 *   FLT_RADIX、FLT_ROUNDS、FLT_EVAL_METHOD，以及实现支持时的扩展浮点宏。
 * 本文件已演示：size_t、ptrdiff_t、offsetof、uint_least32_t、UINT32_C、
 * PRIuLEAST32、CHAR_BIT、INT_MIN、INT_MAX、SIZE_MAX、DBL_MAX、DBL_EPSILON。
 * 本文件未演示：最大宽度整数转换、SCN* 输入、其他宽度类型、C23 位宽及扩展浮点宏。
 * 观察重点：格式与类型一致；不硬编码整数位数或结构填充。N 表示具体位数，* 表示宏家族。
 */
#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    char kind;
    int value;
} Record;

/** @brief 验证元素计数、指针差值、结构偏移及范围宏。 @return 验证完成返回 0。 */
int main(void)
{
    int values[] = {10, 20, 30};
    size_t count = sizeof(values) / sizeof(values[0]);
    ptrdiff_t distance = (values + count) - values; /* 必须在同一数组范围内相减。 */
    assert(count == 3 && distance == 3);
    assert(offsetof(Record, value) < sizeof(Record)); /* 不硬编码填充与对齐。 */
    assert(CHAR_BIT >= 8 && INT_MIN < 0 && INT_MAX >= 32767);
    assert(SIZE_MAX >= count && DBL_MAX >= 1 && DBL_EPSILON > 0);
    uint_least32_t identifier = UINT32_C(123456);
    printf("types：size_t=%zu，ptrdiff_t=%td，整数=%" PRIuLEAST32 "，CHAR_BIT=%d\n",
           count, distance, identifier, CHAR_BIT);
    /* uint_least32_t 保证至少 32 位；精确宽度 uint32_t 依赖实现提供。 */
    return 0;
}
