/** @file @brief 截图命名与目录创建和 BMP 编码分离；不依赖 SDL。 */
#include "bmp_capture.h"
#include "bmp.h"
#include <errno.h>
#include <sys/stat.h>
#include <time.h>

/** @copydoc gfx_bmp_capture */
bool gfx_bmp_capture(const GfxCanvas *canvas, const char *name, char path[BMP_CAPTURE_PATH_SIZE])
{
    if (canvas == nullptr || name == nullptr || name[0] == '\0' || path == nullptr) {
        errno = EINVAL;
        return false;
    }
    for (const char *c = name; *c != '\0'; ++c) {
        if (!((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') || (*c >= '0' && *c <= '9') ||
              *c == '_' || *c == '-')) {
            errno = EINVAL;
            return false;
        }
    }
    /* mkdir 是 POSIX 扩展。已有目录可复用，已有同名普通文件则明确失败。 */
    if (mkdir("exports", 0755) != 0) {
        if (errno != EEXIST) {
            return false;
        }
        struct stat info;
        if (stat("exports", &info) != 0) {
            return false;
        }
        if (!S_ISDIR(info.st_mode)) {
            errno = ENOTDIR;
            return false;
        }
    }
    time_t now = time(nullptr);
    if (now == (time_t)-1) {
        errno = EIO;
        return false;
    }
    struct tm *local = localtime(&now); /* 单线程教学演示，不用于并发共享此静态结果。 */
    char stamp[32];
    if (local == nullptr || strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", local) == 0) {
        errno = EIO;
        return false;
    }
    /* 独占创建配合序号重试：同秒多次截图、程序重启或并发进程都不覆盖旧图。 */
    for (unsigned int serial = 0; serial < 10000; ++serial) {
        int length =
            snprintf(path, BMP_CAPTURE_PATH_SIZE, "exports/%s-%s-%04u.bmp", name, stamp, serial);
        if (length < 0 || length >= BMP_CAPTURE_PATH_SIZE) {
            errno = ENAMETOOLONG;
            return false;
        }
        if (gfx_bmp_save(canvas, path)) {
            return true;
        }
        if (errno != EEXIST) {
            return false;
        }
    }
    errno = EEXIST;
    return false;
}
