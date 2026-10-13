/**
 * @file
 * @brief 互相关、两种边界采样和五种预置滤镜的纯C实现。
 */
#include "image_filters.h"
#include <math.h>
#include <stdlib.h>

/**
 * @brief 检查尺寸与有效区域，未使用像素不参与验证或运算。
 * @param image 可空只读图像。
 * @return 尺寸非零且未超过容量、有效像素全部有限时为true。
 */
static bool image_valid(const VisionImage *image)
{
    if (image == nullptr || image->width == 0 || image->height == 0 || image->width > GFX_WIDTH ||
        image->height > GFX_HEIGHT) {
        return false;
    }
    for (size_t row = 0; row < image->height; ++row) {
        for (size_t column = 0; column < image->width; ++column) {
            if (!isfinite(image->pixels[row][column])) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief 越界坐标按指定规则采样，绝不在检查前索引数组。
 * @param image 有效只读图像，尺寸已验证，可安全转换为int。
 * @param x 水平采样坐标，可以越界。
 * @param y 垂直采样坐标，可以越界。
 * @param border 已验证的合法模式。
 * @return 有限源像素或零。
 */
static double sample_pixel(const VisionImage *image, int x, int y, VisionBorder border)
{
    int width = (int)image->width;
    int height = (int)image->height;
    if (border == VISION_BORDER_ZERO && (x < 0 || y < 0 || x >= width || y >= height)) {
        return 0;
    }
    if (x < 0) {
        x = 0;
    } else if (x >= width) {
        x = width - 1;
    }
    if (y < 0) {
        y = 0;
    } else if (y >= height) {
        y = height - 1;
    }
    return image->pixels[y][x];
}

/**
 * @brief 计算一个像素的互相关，kernel[4]对应当前位置。
 * @param source 有效只读图像。
 * @param kernel 九个已验证有限系数。
 * @param x 图像内横坐标。
 * @param y 图像内纵坐标。
 * @param border 已验证边界模式。
 * @return 未限幅响应，极端数值可能非有限，提交前由调用者检查。
 */
static double correlate_pixel(const VisionImage *source, const double *kernel, int x, int y,
                              VisionBorder border)
{
    double sum = 0;
    for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
            sum += kernel[ky * 3 + kx] * sample_pixel(source, x + kx - 1, y + ky - 1, border);
        }
    }
    return sum;
}

bool vision_from_canvas(const GfxCanvas *canvas, VisionImage *output)
{
    if (canvas == nullptr || output == nullptr) {
        return false;
    }

    output->width = GFX_WIDTH;
    output->height = GFX_HEIGHT;
    for (size_t row = 0; row < GFX_HEIGHT; ++row) {
        for (size_t column = 0; column < GFX_WIDTH; ++column) {
            uint32_t pixel = canvas->pixels[row][column];
            double red = (pixel >> 16) & 255;
            double green = (pixel >> 8) & 255;
            double blue = pixel & 255;
            output->pixels[row][column] = (299 * red + 587 * green + 114 * blue) / 1000;
        }
    }
    return true;
}

/**
 * @brief 共用处理通路，一次分配完整候选结果，全部检查成功才提交。
 * @param source 有效只读输入，已验证尺寸和像素。
 * @param kernel 九个有限系数，主核。
 * @param second 可空第二核；非空时计算两响应的hypot，忽略bias。
 * @param bias 主响应的有限偏移。
 * @param border 已验证边界模式。
 * @param output 有效可写结果，允许与source相同。
 * @return 分配及运算成功为true，否则释放临时内存且保留output。
 */
static bool apply_kernels(const VisionImage *source, const double *kernel, const double *second,
                          double bias, VisionBorder border, VisionImage *output)
{
    VisionImage *next = calloc(1, sizeof(*next));
    if (next == nullptr) {
        return false;
    }
    next->width = source->width;
    next->height = source->height;
    for (size_t row = 0; row < source->height; ++row) {
        for (size_t column = 0; column < source->width; ++column) {
            double value = correlate_pixel(source, kernel, (int)column, (int)row, border);
            if (second != nullptr) {
                double other = correlate_pixel(source, second, (int)column, (int)row, border);
                value = hypot(value, other);
            } else {
                value += bias;
            }
            if (!isfinite(value)) {
                free(next);
                return false;
            }
            next->pixels[row][column] = value;
        }
    }
    *output = *next;
    free(next);
    return true;
}

bool vision_correlate3(const VisionImage *source, const double kernel[9], double bias,
                       VisionBorder border, VisionImage *output)
{
    if (!image_valid(source) || kernel == nullptr || output == nullptr || !isfinite(bias) ||
        (border != VISION_BORDER_CLAMP && border != VISION_BORDER_ZERO)) {
        return false;
    }
    for (size_t i = 0; i < 9; ++i) {
        if (!isfinite(kernel[i])) {
            return false;
        }
    }

    return apply_kernels(source, kernel, nullptr, bias, border, output);
}

bool vision_filter(const VisionImage *source, VisionFilter filter, VisionBorder border,
                   VisionImage *output)
{
    if (!image_valid(source) || output == nullptr || filter < VISION_BOX_BLUR ||
        filter >= VISION_FILTER_COUNT ||
        (border != VISION_BORDER_CLAMP && border != VISION_BORDER_ZERO)) {
        return false;
    }

    const double kernels[VISION_FILTER_COUNT][9] = {
        {1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9},
        {1.0 / 16, 2.0 / 16, 1.0 / 16, 2.0 / 16, 4.0 / 16, 2.0 / 16, 1.0 / 16, 2.0 / 16, 1.0 / 16},
        {0, -1, 0, -1, 5, -1, 0, -1, 0},
        {-2, -1, 0, -1, 0, 1, 0, 1, 2},
        {-1, 0, 1, -2, 0, 2, -1, 0, 1}};
    const double sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    const double *second = filter == VISION_SOBEL ? sobel_y : nullptr;
    double bias = filter == VISION_EMBOSS ? 128 : 0;
    return apply_kernels(source, kernels[filter], second, bias, border, output);
}

bool vision_to_canvas(const VisionImage *image, GfxCanvas *canvas)
{
    if (!image_valid(image) || canvas == nullptr) {
        return false;
    }

    gfx_clear(canvas, gfx_rgb(0, 0, 0));
    for (size_t row = 0; row < image->height; ++row) {
        for (size_t column = 0; column < image->width; ++column) {
            double value = fmax(0, fmin(255, image->pixels[row][column]));
            uint8_t gray = (uint8_t)lround(value);
            canvas->pixels[row][column] = gfx_rgb(gray, gray, gray);
        }
    }
    return true;
}
