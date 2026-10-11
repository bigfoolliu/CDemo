/** @file
 * @brief 使用 strtol/strtod 检查数值文本、尾随字符和范围错误。
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
 * 本文件已演示：strtol（十进制与十六进制）、strtod；借助 <errno.h> 的
 * errno/ERANGE、<limits.h> 的 INT_MIN/INT_MAX 和 <math.h> 的 isfinite 校验结果。
 * 本文件未演示：无符号/long long 转换、strtof/strtold、数值反向格式化和其他通用工具。
 * 观察重点：endptr 是否到达末尾、是否超范围；不要用 atoi 的结果判断错误。
 */
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief 解析十进制 int，采用 strtol 的规则允许前导空白和正负号。
 * @param text 有效的 NUL 结尾文本；不允许尾随空白或其他字符。
 * @param output 有效的输出地址；仅成功时修改。
 * @return 完整解析且在 int 范围内返回 true，否则返回 false。
 */
static bool parse_decimal(const char *text, int *output)
{
    char *end;
    errno = 0; /* errno 不能用成功调用自动清零来推断。 */
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || value < INT_MIN || value > INT_MAX) {
        return false;
    }
    *output = (int)value;
    return true;
}

/**
 * @brief 将整个字符串转换为有限浮点数。
 * @param text 有效的 NUL 结尾文本，遵循当前 locale 的 strtod 语法。
 * @param output 有效的输出地址；仅成功时修改。
 * @return 解析完整、无范围错误且结果有限时返回 true。
 */
static bool parse_finite(const char *text, double *output)
{
    char *end;
    errno = 0;
    double value = strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !isfinite(value)) {
        return false;
    }
    *output = value;
    return true;
}

/** @brief 验证合法输入、空输入、尾随字符和数值范围。 @return 验证完成返回 0。 */
int main(void)
{
    int number = 7;
    bool ok = parse_decimal(" -42", &number);
    assert(ok && number == -42);
    // clang-format off
    const char *invalid[] = {"", "x", "12x", "12 ", "999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999"};
    // clang-format on
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        ok = parse_decimal(invalid[i], &number);
        assert(!ok && number == -42);
    }
    char boundary[sizeof(int) * CHAR_BIT + 3];
    int length = snprintf(boundary, sizeof(boundary), "%d", INT_MAX);
    assert(length > 0 && (size_t)length < sizeof(boundary));
    ok = parse_decimal(boundary, &number);
    assert(ok && number == INT_MAX);
    char *end;
    errno = 0;
    long hexadecimal = strtol("ff", &end, 16);
    assert(hexadecimal == 255 && *end == '\0' && errno == 0);
    double value = 0;
    ok = parse_finite("1.25e2", &value);
    assert(ok && value == 125);
    ok = parse_finite("nan", &value);
    assert(!ok && value == 125);
    ok = parse_finite("1.2x", &value);
    assert(!ok && value == 125);
    puts("conversion：十进制/十六进制、endptr、errno 与有限浮点数验证通过");
    return 0;
}
