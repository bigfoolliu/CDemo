/** @file
 * @brief 在默认 C locale 下分类字符并转换大小写。
 *
 * 学习范围：<ctype.h> 字符分类与大小写映射库。
 * 接口概览（全部常规函数）：
 * - 字符分类：isalnum、isalpha、isblank、iscntrl、isdigit、isgraph、islower、
 *   isprint、ispunct、isspace、isupper、isxdigit。
 * - 大小写转换：tolower、toupper。
 * 本文件已演示：isdigit、isalpha、isalnum、isspace、isxdigit、ispunct、
 * tolower、toupper，并检查空文本和 EOF；<stdio.h> 提供 EOF 宏。
 * 本文件未演示：isblank、iscntrl、isgraph、islower、isprint、isupper。
 * 观察重点：分类结果为零或非零；参数须可表示为 unsigned char 或等于 EOF。
 * 这是字节字符接口；行为受 locale 影响，不是 Unicode 文本分类器。
 */
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief 将字节字符串中的字母原地转换为小写。
 * @param text 可写的 NUL 结尾字节字符串；不是 Unicode 大小写转换接口。
 */
static void lowercase(char *text)
{
    for (size_t i = 0; text[i] != '\0'; ++i) {
        /* char 可能有符号；ctype 参数必须可表示为 unsigned char 或等于 EOF。 */
        text[i] = (char)tolower((unsigned char)text[i]);
    }
}

/** @brief 验证字符分类、空文本和大小写转换。 @return 验证完成返回 0。 */
int main(void)
{
    assert(isdigit('7') != 0 && isdigit('x') == 0);
    assert(isalpha('A') != 0 && isalnum('8') != 0);
    assert(isspace('\t') != 0 && isspace('\n') != 0);
    assert(isxdigit('f') != 0 && ispunct('!') != 0);
    assert(toupper('a') == 'A' && tolower('7') == '7');
    assert(tolower(EOF) == EOF); /* 从 fgetc 得到的 EOF 不要先转成 unsigned char。 */
    char text[] = "C23 API!";
    lowercase(text);
    assert(strcmp(text, "c23 api!") == 0);
    char empty[] = "";
    lowercase(empty);
    assert(empty[0] == '\0');
    puts("ctype：字符分类、unsigned char 参数与大小写转换验证通过");
    return 0;
}
