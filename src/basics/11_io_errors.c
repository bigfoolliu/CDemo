#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    const char *input = "123";
    char *end;
    errno = 0;
    long number = strtol(input, &end, 10);
    assert(errno == 0 && end != input && *end == '\0' && number == 123);
    const char *invalid = "abc";
    errno = 0;
    (void)strtol(invalid, &end, 10);
    assert(end == invalid); /* 没有读到数字；仅检查 errno 不够。 */

    FILE *stream = tmpfile(); /* 临时文件关闭后自动删除，不覆盖用户文件。 */
    if (stream == nullptr) {
        perror("创建临时文件失败");
        return EXIT_FAILURE;
    }
    if (fputs("hello\n", stream) == EOF || fflush(stream) == EOF ||
        fseek(stream, 0, SEEK_SET) != 0) {
        perror("写入或定位失败");
        fclose(stream);
        return EXIT_FAILURE;
    }
    char buffer[16];
    if (fgets(buffer, sizeof(buffer), stream) == nullptr) {
        fclose(stream);
        return EXIT_FAILURE;
    }
    assert(strcmp(buffer, "hello\n") == 0);
    assert(fgetc(stream) == EOF && feof(stream) && !ferror(stream));
    if (fclose(stream) == EOF) {
        return EXIT_FAILURE;
    }
    puts("输入输出：字符串解析、文件读写、EOF 与错误检查验证通过");
    return EXIT_SUCCESS;
}
