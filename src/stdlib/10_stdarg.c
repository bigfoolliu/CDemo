/** @file
 * @brief 使用 va_copy 保留参数列表，并检测格式化缓冲区不足。
 *
 * 学习范围：<stdarg.h> 可变参数库。
 * 接口概览（全部核心接口）：
 * - 参数列表类型：va_list。
 * - 初始化：va_start；获取下一个参数：va_arg；复制状态：va_copy；清理：va_end。
 * 本文件已演示：va_list、va_start、va_copy、va_end；使用 <stdio.h> 的
 * vsnprintf 消费参数列表，先测量长度再格式化。
 * 本文件未直接演示：va_arg；手动逐个取整数参数的示例见 ../basics/12_generic_variadic.c。
 * 观察重点：默认参数提升、格式与类型匹配；每次初始化或复制都配套 va_end。
 * va_* 是宏接口，不要将 va_list 当作普通指针复制，也不要复用已消费的列表。
 */
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief 先计算所需长度，再写入完整的格式化字符串。
 * @param output 至少 capacity 字节的可写数组；本示例要求有效地址。
 * @param capacity 输出容量，包含 NUL 终止符，须大于零。
 * @param format 有效的 printf 格式字符串，不能来自未经校验的用户输入。
 * @param ... 与格式字符串要求一致的参数类型与数量。
 * @return 完整写入返回 true；格式错误或空间不足返回 false，空间不足时不改输出。
 */
static bool format_message(char *output, size_t capacity, const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    va_list copy;
    va_copy(copy, arguments); /* 不能用普通赋值复制 va_list。 */
    int needed = vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    bool ok = needed >= 0 && (size_t)needed < capacity;
    if (ok) {
        ok = vsnprintf(output, capacity, format, arguments) == needed;
    }
    va_end(arguments); /* 每次 va_start/va_copy 都须匹配 va_end。 */
    return ok;
}

/** @brief 验证完整格式化、默认参数提升及空间不足。 @return 验证完成返回 0。 */
int main(void)
{
    char message[32] = "unchanged";
    float fraction = 1.5f; /* 传入 ... 时提升为 double。 */
    bool ok = format_message(message, sizeof(message), "%s:%d:%.1f", "C", 23, fraction);
    assert(ok && strcmp(message, "C:23:1.5") == 0);
    char small[4] = "old";
    ok = format_message(small, sizeof(small), "%s", "long message");
    assert(!ok && strcmp(small, "old") == 0);
    ok = format_message(small, sizeof(small), "%s", "");
    assert(ok && small[0] == '\0');
    puts("stdarg：默认参数提升、va_copy、两次格式化与容量检查验证通过");
    return 0;
}
