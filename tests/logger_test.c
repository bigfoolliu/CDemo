/** @file
 * @brief 日志格式、双输出与失败路径验证。
 */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#include "../src/common/logger.h"
#include <assert.h>
#include <unistd.h>

/**
 * @brief 运行并验证日志格式、双输出与失败路径验证。
 * @return 正常完成返回 0；资源操作失败可能返回非零，断言失败会终止程序。
 */
int main(void)
{
    int made = mkdir("logs", 0755);
    assert(made == 0 || errno == EEXIST);
    char directory[] = "logs/logger-test-XXXXXX";
    char *created = mkdtemp(directory);
    assert(created != nullptr);
    char path[256];
    snprintf(path, sizeof(path), "%s/nested/test.log", directory);
    FILE *console = tmpfile();
    assert(console != nullptr);
    Logger logger;
    bool ok = logger_open(&logger, path, console);
    assert(ok);
    int call_line = __LINE__ + 1;
    logger_write(&logger, "first\n\nlast");
    logger_write(&logger, "值=%d\n", 42);
    /* 长消息验证动态格式化不会截断。 */
    char long_message[7001];
    memset(long_message, 'x', sizeof(long_message) - 1);
    long_message[7000] = '\0';
    logger_write(&logger, "%s", long_message);
    ok = logger_close(&logger);
    assert(ok);
    FILE *file = fopen(path, "r");
    assert(file != nullptr);
    rewind(console);
    int a, b;
    do {
        a = fgetc(file);
        b = fgetc(console);
        assert(a == b);
    } while (a != EOF);
    rewind(file);
    char line[8192], first_stamp[24], location[64];
    snprintf(location, sizeof(location), " | logger_test.c:%d | ", call_line);
    for (int i = 0; i < 5; ++i) {
        char *read = fgets(line, sizeof(line), file);
        assert(read != nullptr);
        int year, month, day, hour, minute, second, milliseconds;
        int fields = sscanf(line, "%d-%d-%d %d:%d:%d,%d", &year, &month, &day,
                            &hour, &minute, &second, &milliseconds);
        assert(fields == 7 && month >= 1 && month <= 12 && milliseconds >= 0 && milliseconds <= 999);
        if (i == 0) {
            memcpy(first_stamp, line, 23);
            first_stamp[23] = '\0';
        }
        if (i < 3) {
            assert(strncmp(first_stamp, line, 23) == 0 && strstr(line, location) != nullptr);
        }
        if (i == 1) {
            assert(strstr(line, " | \n") != nullptr); /* 空行也有前缀。 */
        }
        if (i == 4) {
            assert(strlen(line) > 7000);
        }
    }
    assert(fgetc(file) == EOF);
    int closed = fclose(file);
    assert(closed == 0);
    ok = logger_open(&logger, path, nullptr);
    assert(ok);
    logger_write(&logger, "append");
    ok = logger_close(&logger);
    assert(ok);
    file = fopen(path, "r");
    assert(file != nullptr);
    size_t lines = 0;
    while (fgets(line, sizeof(line), file) != nullptr) {
        ++lines;
    }
    assert(lines == 6);
    closed = fclose(file);
    assert(closed == 0);
    ok = logger_open(&logger, "logs/../forbidden.log", nullptr);
    assert(!ok && logger.failed);
    ok = logger_open(&logger, directory, nullptr); /* 目录不能当日志文件。 */
    assert(!ok && logger.failed);
    logger = (Logger){.file = fopen(path, "r")};
    assert(logger.file != nullptr);
    logger_write(&logger, "write to read-only stream");
    ok = logger_close(&logger);
    assert(!ok && logger.failed);
    closed = fclose(console); /* 记录器没有关闭借用流。 */
    assert(closed == 0);
    int removed = remove(path);
    assert(removed == 0);
    snprintf(path, sizeof(path), "%s/nested", directory);
    removed = rmdir(path);
    assert(removed == 0);
    removed = rmdir(directory);
    assert(removed == 0);
    puts("日志测试通过：毫秒时间/调用位置、多行、长消息、双输出一致、追加、路径及写入失败。");
    return 0;
}
