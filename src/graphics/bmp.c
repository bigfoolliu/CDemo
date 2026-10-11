/** @file
 * @brief BMP = 14 字节文件头 + 40 字节信息头 + BGR 像素行。
 * 字段逐字节编码，避免直接 fwrite 结构体造成填充或主机字节序问题。
 */
#include "bmp.h"
#include <errno.h>
#include <limits.h>

_Static_assert(CHAR_BIT == 8, "BMP 文件编码要求 8 位字节");

/**
 * @brief 将无符号整数的低 n 个字节按小端顺序存入数组。
 * @param out 可写的 n 字节区域。
 * @param value 要编码的数值。
 * @param n 字节数，本模块只传 2 或 4。
 */
static void little_endian(unsigned char *out, uint32_t value, unsigned int n)
{
    for (unsigned int i = 0; i < n; ++i) {
        out[i] = (unsigned char)(value >> (8 * i));
    }
}

/** @copydoc gfx_bmp_write */
bool gfx_bmp_write(FILE *stream, const GfxCanvas *canvas, int width, int height)
{
    if (stream == nullptr || canvas == nullptr || width <= 0 || width > GFX_WIDTH || height <= 0 ||
        height > GFX_HEIGHT) {
        errno = EINVAL;
        return false;
    }
    /* 24 位像素占三字节，每行向上补齐到四字节倍数。
     * 例：宽 1 → 3 字节颜色 + 1 字节填充；宽 2 → 6 + 2。
     * 尺寸被限定在固定画布范围，以下长度计算不会溢出 uint32_t。
     */
    uint32_t stride = ((uint32_t)width * 3 + 3) / 4 * 4;
    uint32_t image_size = stride * (uint32_t)height;
    unsigned char header[54] = {0};
    header[0] = 'B';
    header[1] = 'M';
    little_endian(header + 2, 54 + image_size, 4); /* 文件总长度。 */
    little_endian(header + 10, 54, 4);             /* 像素从文件偏移 54 开始，无调色板。 */
    little_endian(header + 14, 40, 4);             /* BITMAPINFOHEADER 的字节长度。 */
    little_endian(header + 18, (uint32_t)width, 4);
    little_endian(header + 22, (uint32_t)height, 4); /* 正高度表示自下而上。 */
    little_endian(header + 26, 1, 2);                /* 色彩平面数量固定为一。 */
    little_endian(header + 28, 24, 2);               /* 每像素位数。 */
    /* 偏移 30 的压缩方式保持为零，即 BI_RGB；其他未使用字段也保持零。 */
    little_endian(header + 34, image_size, 4);
    if (fwrite(header, 1, sizeof header, stream) != sizeof header) {
        goto failure;
    }
    unsigned char row[(GFX_WIDTH * 3 + 3) / 4 * 4];
    for (int y = height - 1; y >= 0; --y) {
        /* BMP 第一行是画面的最下方，画布第一行是最上方，所以逆序读取行。
         * 通过移位取通道，不依赖 uint32_t 在内存中的实际字节排列。
         */
        for (int x = 0; x < width; ++x) {
            uint32_t color = canvas->pixels[y][x];
            row[x * 3] = (unsigned char)color;
            row[x * 3 + 1] = (unsigned char)(color >> 8);
            row[x * 3 + 2] = (unsigned char)(color >> 16);
        }
        for (uint32_t i = (uint32_t)width * 3; i < stride; ++i) {
            row[i] = 0;
        }
        if (fwrite(row, 1, stride, stream) != stride) {
            goto failure;
        }
    }
    /* stdio 有缓冲，fwrite 成功不代表底层写入成功；刷新时仍可能失败。 */
    if (fflush(stream) == 0) {
        return true;
    }
failure:
    if (errno == 0) {
        errno = EIO;
    }
    return false;
}

/** @copydoc gfx_bmp_save */
bool gfx_bmp_save(const GfxCanvas *canvas, const char *path)
{
    if (canvas == nullptr || path == nullptr || path[0] == '\0') {
        errno = EINVAL;
        return false;
    }
    /* x 要求独占创建，已有同名文件时失败，避免“检查存在再创建”的竞争窗口。 */
    FILE *stream = fopen(path, "wbx");
    if (stream == nullptr) {
        return false;
    }
    errno = 0;
    bool ok = gfx_bmp_write(stream, canvas, GFX_WIDTH, GFX_HEIGHT);
    int saved_error = errno;
    if (fclose(stream) != 0 && ok) {
        ok = false;
        saved_error = errno != 0 ? errno : EIO;
    }
    if (!ok) {
        /* 只有本次独占创建成功的路径才会清理；保留最初的写入/关闭错误。 */
        remove(path);
        errno = saved_error;
    }
    return ok;
}
