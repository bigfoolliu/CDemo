/**
 * @file
 * @brief 图像滤镜实验室：生成确定的测试图，观察五种灰度滤镜并独占导出BMP。
 */
#include "image_filters.h"
#include "../graphics/bmp_capture.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief 绘制渐变、棋盘格、色块、圆与细线，不需要外部图片。
 * @param canvas 有效可写画布，覆盖全部像素。
 */
static void draw_scene(GfxCanvas *canvas)
{
    for (int row = 0; row < GFX_HEIGHT; ++row) {
        for (int column = 0; column < GFX_WIDTH; ++column) {
            uint8_t gray = (uint8_t)(25 + column * 100 / (GFX_WIDTH - 1));
            canvas->pixels[row][column] = gfx_rgb(gray, gray, gray);
        }
    }
    for (int row = 0; row < 8; ++row) {
        for (int column = 0; column < 10; ++column) {
            uint8_t gray = (row + column) % 2 == 0 ? 230 : 20;
            gfx_rect(canvas, column * 10, row * 10, 10, 10, true, gfx_rgb(gray, gray, gray));
        }
    }
    gfx_rect(canvas, 120, 20, 50, 60, true, gfx_rgb(255, 45, 35));
    gfx_rect(canvas, 180, 20, 50, 60, true, gfx_rgb(35, 230, 60));
    gfx_rect(canvas, 240, 20, 50, 60, true, gfx_rgb(30, 60, 255));
    gfx_circle(canvas, 70, 160, 42, gfx_rgb(245, 245, 245));
    gfx_circle(canvas, 70, 160, 30, gfx_rgb(5, 5, 5));
    gfx_rect(canvas, 140, 120, 70, 75, true, gfx_rgb(230, 230, 230));
    gfx_rect(canvas, 155, 135, 40, 45, true, gfx_rgb(15, 15, 15));
    for (int i = 0; i < 6; ++i) {
        gfx_line(canvas, 230, 105 + i * 20, 319, 125 + i * 20, gfx_rgb(255, 255, 255));
    }
}

/**
 * @brief 按真实灰度范围打印响应最小/最大值，显示之前不裁剪。
 * @param name 有效只读滤镜名称。
 * @param image 有效只读有限图像。
 */
static void print_range(const char *name, const VisionImage *image)
{
    double low = image->pixels[0][0];
    double high = low;
    for (size_t row = 0; row < image->height; ++row) {
        for (size_t column = 0; column < image->width; ++column) {
            double value = image->pixels[row][column];
            if (value < low) {
                low = value;
            }
            if (value > high) {
                high = value;
            }
        }
    }
    printf("%-14s 浮点响应范围 [%8.3f, %8.3f]；BMP显示限幅到[0,255]\n", name, low, high);
}

/**
 * @brief 独占创建BMP，打印实际路径，失败保留已经导出的文件。
 * @param canvas 有效只读画布。
 * @param name 有效ASCII文件名前缀。
 * @return 写入成功为true，失败输出错误原因。
 */
static bool export_canvas(const GfxCanvas *canvas, const char *name)
{
    char path[BMP_CAPTURE_PATH_SIZE];
    if (!gfx_bmp_capture(canvas, name, path)) {
        fprintf(stderr, "滤镜BMP导出失败：%s\n", strerror(errno));
        return false;
    }
    printf("已导出：%s\n", path);
    return true;
}

/**
 * @brief 默认计算并报告，--export保存七张图片，--zero选择零填充。
 * @param argc 参数数量。
 * @param argv 接受各一次--export、--zero；未知或重复选项失败且不导出。
 * @return 所有处理及输出成功为EXIT_SUCCESS，否则为EXIT_FAILURE。
 */
int main(int argc, char **argv)
{
    bool exporting = false;
    VisionBorder border = VISION_BORDER_CLAMP;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--export") == 0 && !exporting) {
            exporting = true;
        } else if (strcmp(argv[i], "--zero") == 0 && border != VISION_BORDER_ZERO) {
            border = VISION_BORDER_ZERO;
        } else {
            fprintf(stderr, "用法：%s [--export] [--zero]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    static GfxCanvas canvas;
    static VisionImage original;
    static VisionImage filtered;
    draw_scene(&canvas);
    printf("图像滤镜实验室：320×240，3×3核，边界=%s\n",
           border == VISION_BORDER_CLAMP ? "边缘复制" : "零填充");
    if (!vision_from_canvas(&canvas, &original)) {
        fputs("灰度转换失败\n", stderr);
        return EXIT_FAILURE;
    }
    if (exporting && !export_canvas(&canvas, "filters-original")) {
        return EXIT_FAILURE;
    }
    print_range("grayscale", &original);
    if (!vision_to_canvas(&original, &canvas) ||
        (exporting && !export_canvas(&canvas, "filters-grayscale"))) {
        return EXIT_FAILURE;
    }
    const char *names[VISION_FILTER_COUNT] = {"filters-box", "filters-gaussian", "filters-sharpen",
                                              "filters-emboss", "filters-sobel"};
    for (int filter = VISION_BOX_BLUR; filter < VISION_FILTER_COUNT; ++filter) {
        /* 各滤镜读取同一份原始灰度图，不把上一滤镜结果当作下一滤镜输入。 */
        if (!vision_filter(&original, (VisionFilter)filter, border, &filtered) ||
            !vision_to_canvas(&filtered, &canvas)) {
            fputs("滤镜处理失败：输入、内存或浮点运算无效\n", stderr);
            return EXIT_FAILURE;
        }
        print_range(names[filter], &filtered);
        if (exporting && !export_canvas(&canvas, names[filter])) {
            return EXIT_FAILURE;
        }
    }
    return fflush(stdout) == 0 && !ferror(stdout) ? EXIT_SUCCESS : EXIT_FAILURE;
}
