#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

enum ParseStatus { PARSE_OK, PARSE_EMPTY, PARSE_INVALID, PARSE_RANGE };

/* 严格解析十进制 int：接受一个可选正负号，不接受空白和尾随字符。
 * 仅成功时写入 output；不使用 atoi，因为它无法可靠报告错误。
 * text 必须是有效的 NUL 结尾字符串，output 必须指向有效 int。 */
static enum ParseStatus parse_int(const char *text, int *output)
{
    const unsigned char *cursor = (const unsigned char *)text;
    if (*cursor == '\0') {
        return PARSE_EMPTY;
    }
    bool negative = *cursor == '-';
    if (*cursor == '-' || *cursor == '+') {
        ++cursor;
    }
    if (*cursor == '\0') {
        return PARSE_INVALID;
    }
    /* 以负数累积，避免先构造无法表示的正数 abs(INT_MIN)。 */
    int limit = negative ? INT_MIN : -INT_MAX;
    int accumulated = 0;
    while (*cursor != '\0') {
        if (*cursor < '0' || *cursor > '9') {
            return PARSE_INVALID;
        }
        int digit = *cursor - '0';
        if (accumulated < limit / 10) {
            return PARSE_RANGE;
        }
        accumulated *= 10;
        if (accumulated < limit + digit) {
            return PARSE_RANGE;
        }
        accumulated -= digit;
        ++cursor;
    }
    *output = negative ? accumulated : -accumulated;
    return PARSE_OK;
}

int main(void)
{
    const struct {
        const char *text;
        enum ParseStatus status;
        int value;
    } cases[] = {
        {"0", PARSE_OK, 0}, {"-0", PARSE_OK, 0}, {"+42", PARSE_OK, 42},
        {"-17", PARSE_OK, -17}, {"", PARSE_EMPTY, 0},
        {"+", PARSE_INVALID, 0}, {"12x", PARSE_INVALID, 0},
        {" 1", PARSE_INVALID, 0}, {"--1", PARSE_INVALID, 0}
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int value = 99;
        enum ParseStatus status = parse_int(cases[i].text, &value);
        assert(status == cases[i].status);
        assert(value == (status == PARSE_OK ? cases[i].value : 99));
    }
    char text[sizeof(int) * CHAR_BIT + 3];
    int value = 0;
    snprintf(text, sizeof(text), "%d", INT_MIN);
    enum ParseStatus status = parse_int(text, &value);
    assert(status == PARSE_OK && value == INT_MIN);
    snprintf(text, sizeof(text), "%d", INT_MAX);
    status = parse_int(text, &value);
    assert(status == PARSE_OK && value == INT_MAX);
    /* 最大值的十进制文本追加 0，必定超范围。 */
    size_t length = strlen(text);
    text[length] = '0';
    text[length + 1] = '\0';
    status = parse_int(text, &value);
    assert(status == PARSE_RANGE && value == INT_MAX);
    puts("整数解析：完整消费输入、INT_MIN/MAX、溢出前检查验证通过");
    return 0;
}
