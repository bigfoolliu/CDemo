/** @file
 * @brief 数组与字符串。
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief 运行并验证数组与字符串。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    int values[] = {1, 2, 3};
    size_t length = sizeof(values) / sizeof(values[0]);
    int total = 0;
    for (size_t i = 0; i < length; ++i) {
        total += values[i];
    }
    int matrix[2][2] = {{1, 2}, {3, 4}};
    char word[] = "hello"; /* 自动包含末尾的 '\0'，这个数组可以修改。 */
    word[0] = 'H';
    const char *literal = "world"; /* 不要修改字符串字面量。 */
    char message[32];
    int written = snprintf(message, sizeof(message), "%s %s", word, literal);
    assert(written >= 0 && (size_t)written < sizeof(message));
    assert(total == 6 && matrix[1][0] == 3);
    assert(sizeof(word) == 6 && strlen(word) == 5);
    assert(strcmp(message, "Hello world") == 0);
    printf("数组与字符串：%s\n", message);
    return 0;
}
