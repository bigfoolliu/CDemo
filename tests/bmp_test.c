/** @file @brief BMP 字节验证及临时目录中的文件保护、命名和失败路径测试。 */
#include "../src/graphics/bmp_capture.h"
#include "../src/graphics/bmp.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * @brief 解码一个小端字段，测试端不使用生产编码函数。
 * @param bytes 有效的只读字节数组。
 * @param count 字节数，最多四个。
 * @return 解码后的无符号数值。
 */
static uint32_t read_le(const unsigned char *bytes, unsigned int count)
{
    uint32_t value = 0;
    for (unsigned int i = 0; i < count; ++i) { value |= (uint32_t)bytes[i] << (8 * i); }
    return value;
}

/** @brief 宽 1..4 覆盖全部行填充长度，同时验证 RGB、上下方向与 alpha 忽略。 */
static void test_bytes(void)
{
    static GfxCanvas canvas;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 4; ++x) {
            canvas.pixels[y][x] = gfx_rgb((uint8_t)(20 + x), (uint8_t)(70 + y), (uint8_t)(110 + x + y));
        }
    }
    canvas.pixels[0][0] &= UINT32_C(0x00ffffff); /* alpha 为零，仍导出其 RGB。 */
    for (int width = 1; width <= 4; ++width) {
        FILE *stream = tmpfile();
        assert(stream != nullptr && gfx_bmp_write(stream, &canvas, width, 2));
        long stride = (width * 3 + 3) / 4 * 4;
        assert(ftell(stream) == 54 + 2 * stride);
        rewind(stream);
        unsigned char bytes[128];
        size_t length = fread(bytes, 1, sizeof bytes, stream);
        assert(length == (size_t)(54 + 2 * stride));
        assert(bytes[0] == 'B' && bytes[1] == 'M');
        assert(read_le(bytes + 2, 4) == length && read_le(bytes + 10, 4) == 54);
        assert(read_le(bytes + 14, 4) == 40 && read_le(bytes + 18, 4) == (uint32_t)width);
        assert(read_le(bytes + 22, 4) == 2 && read_le(bytes + 26, 2) == 1);
        assert(read_le(bytes + 28, 2) == 24 && read_le(bytes + 30, 4) == 0);
        assert(read_le(bytes + 34, 4) == (uint32_t)(2 * stride));
        for (int row = 0; row < 2; ++row) {
            int y = 1 - row;
            for (int x = 0; x < width; ++x) {
                size_t offset = 54 + (size_t)row * (size_t)stride + (size_t)x * 3;
                assert(bytes[offset] == 110 + x + y);
                assert(bytes[offset + 1] == 70 + y);
                assert(bytes[offset + 2] == 20 + x);
            }
            for (long pad = width * 3; pad < stride; ++pad) {
                assert(bytes[54 + row * stride + pad] == 0);
            }
        }
        assert(fclose(stream) == 0);
    }
    FILE *stream = tmpfile();
    assert(stream != nullptr);
    assert(!gfx_bmp_write(stream, &canvas, 0, 2) && errno == EINVAL);
    assert(!gfx_bmp_write(stream, &canvas, 1, GFX_HEIGHT + 1) && errno == EINVAL);
    assert(ftell(stream) == 0); /* 非法尺寸在写文件头之前被拒绝。 */
    assert(fclose(stream) == 0);
}

/** @brief 所有持久文件测试都在临时目录中，不创建或清理用户已有图片。 */
static void test_files(void)
{
    char previous[4096];
    assert(getcwd(previous, sizeof previous) != nullptr);
    char directory[] = "/tmp/cdemo-bmp-XXXXXX";
    assert(mkdtemp(directory) != nullptr && chdir(directory) == 0);
    static GfxCanvas canvas;
    gfx_clear(&canvas, gfx_rgb(1, 2, 3));
    FILE *file = fopen("existing.bmp", "wb");
    assert(file != nullptr && fputs("keep", file) >= 0 && fclose(file) == 0);
    assert(!gfx_bmp_save(&canvas, "existing.bmp") && errno == EEXIST);
    file = fopen("existing.bmp", "rb");
    char marker[5] = {0};
    assert(file != nullptr && fread(marker, 1, 4, file) == 4 && strcmp(marker, "keep") == 0);
    errno = 0;
    assert(!gfx_bmp_write(file, &canvas, 1, 1)); /* 只读流必须报告写入失败。 */
    assert(fclose(file) == 0);
    assert(!gfx_bmp_save(&canvas, "missing/image.bmp"));
    assert(!gfx_bmp_save(&canvas, "") && errno == EINVAL);
    char first[BMP_CAPTURE_PATH_SIZE], second[BMP_CAPTURE_PATH_SIZE];
    assert(!gfx_bmp_capture(&canvas, "../escape", first) && errno == EINVAL);
    /* exports 若是普通文件，目录创建须明确失败，不破坏该文件。 */
    file = fopen("exports", "wb");
    assert(file != nullptr && fclose(file) == 0);
    assert(!gfx_bmp_capture(&canvas, "test", first) && errno == ENOTDIR);
    assert(remove("exports") == 0);
    assert(gfx_bmp_capture(&canvas, "test", first));
    assert(gfx_bmp_capture(&canvas, "test", second));
    assert(strcmp(first, second) != 0 && strncmp(first, "exports/test-", 13) == 0);
    file = fopen(first, "rb");
    assert(file != nullptr && fseek(file, 0, SEEK_END) == 0 && ftell(file) == 230454);
    assert(fclose(file) == 0);
    assert(remove(first) == 0 && remove(second) == 0 && remove("existing.bmp") == 0);
    assert(rmdir("exports") == 0 && chdir(previous) == 0 && rmdir(directory) == 0);
}

/** @brief 运行全部 BMP 格式与失败行为断言。 @return 成功返回零。 */
int main(void)
{
    test_bytes(); test_files();
    puts("BMP 测试通过：文件头、BGR、上下方向、行填充、追加截图与失败路径。");
    return 0;
}
