/** @file
 * @brief 单线程双输出日志接口。
 */
#ifndef CDEMO_LOGGER_H
#define CDEMO_LOGGER_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

/* 单线程日志记录器：拥有文件流，借用终端流；错误状态持续保留。
 * 使用追加模式保留此前运行记录；每次写入后刷新以便实时查看。
 * static inline 使独立示例无需额外链接一个实现文件。 */
typedef struct {
    FILE *file;
    FILE *console;
    bool failed;
} Logger;

/**
 * @brief 校验日志路径并创建缺失的父目录。
 * @param path 有效的 NUL 结尾相对路径，必须位于 logs/ 内且无回退组件。
 * @return 路径有效且父目录就绪返回 true；失败返回 false 并设置 errno。
 * @note 日志使用 logs/ 下的相对路径；拒绝目录回退，自动创建父目录。
 *       mkdir 是 POSIX 接口，其余记录与格式化逻辑使用标准 C。
 */
static inline bool logger_prepare_path(const char *path)
{
    if (strncmp(path, "logs/", 5) != 0 || strlen(path) >= 1024 || path[5] == '\0') {
        errno = EINVAL;
        return false;
    }
    char directory[1024];
    strcpy(directory, path);
    for (char *component = directory; component != nullptr;) {
        char *slash = strchr(component, '/');
        if (slash != nullptr) {
            *slash = '\0';
        }
        if (*component == '\0' || strcmp(component, "..") == 0 || strcmp(component, ".") == 0) {
            errno = EINVAL;
            return false;
        }
        if (slash == nullptr) {
            break;
        }
        *slash = '/';
        component = slash + 1;
    }
    for (char *cursor = directory; *cursor != '\0'; ++cursor) {
        if (*cursor != '/') {
            continue;
        }
        *cursor = '\0';
        if (mkdir(directory, 0755) != 0 && errno != EEXIST) {
            return false;
        }
        struct stat status;
        if (stat(directory, &status) != 0 || !S_ISDIR(status.st_mode)) {
            errno = ENOTDIR;
            return false;
        }
        *cursor = '/';
    }
    return true;
}

/**
 * @brief 初始化单线程记录器并以追加模式打开日志。
 * @param logger 有效的输出地址，不应覆盖仍持有文件流的记录器。
 * @param path 有效的 logs/ 下相对日志路径。
 * @param console 借用的可写流；nullptr 表示仅写文件，记录器不关闭它。
 * @return 成功返回 true；失败返回 false 并设置 failed。
 * @note console 为 nullptr 时只写文件；已有文件采用追加模式。
 */
static inline bool logger_open(Logger *logger, const char *path, FILE *console)
{
    *logger = (Logger){.console = console};
    if (!logger_prepare_path(path)) {
        logger->failed = true;
        return false;
    }
    logger->file = fopen(path, "a");
    if (logger->file == nullptr) {
        logger->failed = true;
        return false;
    }
    return true;
}

/**
 * @brief 格式化消息并为每行添加时间及调用位置。
 * @param logger 已初始化的可读写记录器，失败状态会累计保留。
 * @param source 有效的 NUL 结尾调用源码路径，通常由宏自动传入。
 * @param line 调用处源码行号，通常由宏自动传入。
 * @param format 有效的 printf 格式字符串。
 * @param ... 与 format 中转换说明匹配的可变参数。
 * @note 一次调用是一条消息：每个物理行都有前缀，没有末尾换行时自动补齐。
 *       同一消息的所有行及输出目标共用时间戳，格式化只执行一次。
 */
static inline void logger_write_at(Logger *logger, const char *source, int line, const char *format, ...)
{
    struct timespec now;
    char timestamp[32];
    if (timespec_get(&now, TIME_UTC) != TIME_UTC) {
        logger->failed = true;
        return;
    }
    struct tm *local = localtime(&now.tv_sec);
    if (local == nullptr || strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", local) == 0) {
        logger->failed = true;
        return;
    }
    const char *filename = source;
    for (const char *cursor = source; *cursor != '\0'; ++cursor) {
        if (*cursor == '/' || *cursor == '\\') {
            filename = cursor + 1;
        }
    }
    va_list arguments;
    va_start(arguments, format);
    va_list copy;
    va_copy(copy, arguments);
    int length = vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    char *message = length >= 0 ? malloc((size_t)length + 1) : nullptr;
    if (message == nullptr) {
        va_end(arguments);
        logger->failed = true;
        return;
    }
    int written = vsnprintf(message, (size_t)length + 1, format, arguments);
    va_end(arguments);
    if (written != length) {
        free(message);
        logger->failed = true;
        return;
    }
    FILE *streams[] = {logger->console, logger->file};
    for (size_t i = 0; i < sizeof(streams) / sizeof(streams[0]); ++i) {
        if (streams[i] == nullptr) {
            continue;
        }
        size_t start = 0;
        for (size_t end = 0; end <= (size_t)length; ++end) {
            if (end < (size_t)length && message[end] != '\n') {
                continue;
            }
            /* 已有末尾换行不额外生成空行；显式空白行仍记录前缀。 */
            if (end == (size_t)length && start == end && length > 0) {
                break;
            }
            if (fprintf(streams[i], "%s,%03ld | %s:%d | ", timestamp,
                        now.tv_nsec / 1000000L, filename, line) < 0 ||
                fwrite(message + start, 1, end - start, streams[i]) != end - start ||
                fputc('\n', streams[i]) == EOF) {
                logger->failed = true;
            }
            start = end + 1;
        }
        if (fflush(streams[i]) == EOF) {
            logger->failed = true;
        }
    }
    free(message);
}

/* 宏在调用处捕获文件名和行号，而不是 logger.h 内部的位置。 */
#define logger_write(logger, ...) logger_write_at((logger), __FILE__, __LINE__, __VA_ARGS__)

/**
 * @brief 关闭拥有的文件流并报告累计写入状态。
 * @param logger 有效的可读写记录器；不关闭借用的 console 流。
 * @return 所有记录与关闭操作均成功返回 true，否则返回 false。
 * @note 只关闭拥有的文件，不关闭借用的终端流；返回累计的成功状态。
 */
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
