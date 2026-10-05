#ifndef CDEMO_LOGGER_H
#define CDEMO_LOGGER_H

#include <stdarg.h>
#include <stdio.h>

/* 单线程日志记录器：拥有文件流，借用终端流；错误状态持续保留。
 * 使用追加模式保留此前运行记录；每次写入后刷新以便实时查看。
 * static inline 使独立示例无需额外链接一个实现文件。 */
typedef struct {
    FILE *file;
    FILE *console;
    bool failed;
} Logger;

static inline bool logger_open(Logger *logger, const char *path, FILE *console)
{
    *logger = (Logger){.console = console};
    logger->file = fopen(path, "a");
    if (logger->file == nullptr) {
        logger->failed = true;
        return false;
    }
    return true;
}

static inline void logger_write(Logger *logger, const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    /* vfprintf 会消费 va_list，两个输出目标必须分别使用副本。 */
    FILE *streams[] = {logger->console, logger->file};
    for (size_t i = 0; i < sizeof(streams) / sizeof(streams[0]); ++i) {
        if (streams[i] == nullptr) {
            continue;
        }
        va_list copy;
        va_copy(copy, arguments);
        if (vfprintf(streams[i], format, copy) < 0) {
            logger->failed = true;
        }
        va_end(copy);
        if (fflush(streams[i]) == EOF) {
            logger->failed = true;
        }
    }
    va_end(arguments);
}

static inline bool logger_close(Logger *logger)
{
    if (logger->file != nullptr) {
        if (fclose(logger->file) == EOF) {
            logger->failed = true;
        }
        logger->file = nullptr;
    }
    return !logger->failed;
}

#endif
