/** @file
 * @brief 区分 NUL 结尾字符串与按长度处理的字节缓冲区。
 *
 * 学习范围：<string.h> 字符串与字节处理库。
 * 接口概览（主要常规接口，不含可选边界检查接口）：
 * - 复制与拼接：memcpy、memmove、strcpy、strncpy、strcat、strncat。
 * - 比较：memcmp、strcmp、strncmp、strcoll；排序键转换：strxfrm。
 * - 查找与分割：memchr、strchr、strrchr、strstr、strpbrk、strspn、strcspn、strtok。
 * - 长度、填充与错误文本：strlen、memset、strerror。
 * - C23 新增：memccpy、strdup、strndup、memset_explicit；需单独验证平台支持。
 * - 类型与宏：size_t、NULL，以及标准版本相关宏。
 * 本文件已演示：strlen、strcmp、strncmp、strchr、strstr、memcpy、memmove、
 * memset、memcmp、memchr，含空字符串、未命中与重叠复制。
 * 本文件未演示：拼接、分割、locale 比较、动态字符串复制和显式内存擦除。
 * 观察重点：字符串以 NUL 结尾；字节缓冲区按长度处理；memcpy 不允许重叠。
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>

/** @brief 验证字符串查找、比较及重叠内存移动。 @return 验证完成返回 0。 */
int main(void)
{
    const char text[] = "C library";
    assert(strlen(text) == 9 && strlen("") == 0);
    assert(strcmp(text, "C library") == 0 && strcmp("a", "b") < 0);
    assert(strncmp("apple", "application", 4) == 0);
    const char *space = strchr(text, ' ');
    const char *word = strstr(text, "library");
    assert(space == text + 1 && word == text + 2);
    assert(strchr(text, '!') == nullptr && strstr(text, "xyz") == nullptr);
    assert(strchr(text, '\0') == text + strlen(text));

    char copied[sizeof(text)];
    memcpy(copied, text, sizeof(text)); /* 包括终止符；源与目标不重叠。 */
    assert(strcmp(copied, text) == 0);
    char overlap[16] = "abcde";
    memmove(overlap + 1, overlap, strlen(overlap) + 1);
    assert(strcmp(overlap, "aabcde") == 0); /* 重叠区域必须用 memmove。 */

    unsigned char bytes[4];
    memset(bytes, 0x7f, sizeof(bytes)); /* 填充字节，不是给任意对象赋值。 */
    assert(bytes[0] == 0x7f && memchr(bytes, 0x7f, sizeof(bytes)) == bytes);
    assert(memchr(bytes, 0, sizeof(bytes)) == nullptr);
    assert(memcmp(bytes, bytes, sizeof(bytes)) == 0);
    puts("string：长度、比较、查找、字节复制与重叠移动验证通过");
    return 0;
}
