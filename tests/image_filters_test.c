/**
 * @file
 * @brief 验证手算响应、核方向、边界、原地处理和失败不修改。
 */
#include "../src/vision/image_filters.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief 比较手算结果，允许浮点舍入误差。
 * @param actual 实际有限响应。
 * @param expected 手算期望值。
 */
static void close_number(double actual, double expected)
{
    assert(isfinite(actual) && fabs(actual - expected) < 1e-9);
}

/**
 * @brief 恒定图验证归一化、零和核、Sobel及零填充的边缘变暗。
 */
static void test_constant_and_borders(void)
{
    static VisionImage input;
    static VisionImage output;
    input = (VisionImage){.width = 3, .height = 3};
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            input.pixels[row][column] = 90;
        }
    }
    const double expected[] = {90, 90, 90, 128, 0};
    for (int filter = 0; filter < VISION_FILTER_COUNT; ++filter) {
        assert(vision_filter(&input, (VisionFilter)filter, VISION_BORDER_CLAMP, &output));
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                close_number(output.pixels[row][column], expected[filter]);
            }
        }
    }
    assert(vision_filter(&input, VISION_BOX_BLUR, VISION_BORDER_ZERO, &output));
    close_number(output.pixels[0][0], 40);
    close_number(output.pixels[0][1], 60);
    close_number(output.pixels[1][1], 90);
    input.width = 1;
    input.height = 1;
    assert(vision_filter(&input, VISION_GAUSSIAN_BLUR, VISION_BORDER_CLAMP, &output));
    close_number(output.pixels[0][0], 90);
    assert(vision_filter(&input, VISION_GAUSSIAN_BLUR, VISION_BORDER_ZERO, &output));
    close_number(output.pixels[0][0], 22.5);
}

/**
 * @brief 脉冲和非对称核验证系数、核不翻转、负值保留与原地处理。
 */
static void test_impulse_and_alias(void)
{
    static VisionImage input;
    static VisionImage output;
    static VisionImage inplace;
    input = (VisionImage){.width = 3, .height = 3};
    input.pixels[1][1] = 16;
    assert(vision_filter(&input, VISION_GAUSSIAN_BLUR, VISION_BORDER_ZERO, &output));
    close_number(output.pixels[0][0], 1);
    close_number(output.pixels[0][1], 2);
    close_number(output.pixels[1][1], 4);
    assert(vision_filter(&input, VISION_SHARPEN, VISION_BORDER_ZERO, &output));
    close_number(output.pixels[1][1], 80);
    close_number(output.pixels[0][1], -16);
    const double right_neighbor[9] = {0, 0, 0, 0, 0, 1, 0, 0, 0};
    assert(vision_correlate3(&input, right_neighbor, 3, VISION_BORDER_ZERO, &output));
    close_number(output.pixels[1][0], 19);
    close_number(output.pixels[1][2], 3);
    for (int filter = 0; filter < VISION_FILTER_COUNT; ++filter) {
        inplace = input;
        assert(vision_filter(&input, (VisionFilter)filter, VISION_BORDER_CLAMP, &output));
        assert(vision_filter(&inplace, (VisionFilter)filter, VISION_BORDER_CLAMP, &inplace));
        assert(memcmp(&inplace, &output, sizeof(output)) == 0);
    }
}

/**
 * @brief 水平正反与垂直斜坡验证Sobel符号与幅值，无提前限幅。
 */
static void test_sobel(void)
{
    static VisionImage input;
    static VisionImage output;
    const double sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    input = (VisionImage){.width = 3, .height = 3};
    for (int direction = -1; direction <= 1; direction += 2) {
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                input.pixels[row][column] = 100 + direction * (double)column * 10;
            }
        }
        assert(vision_correlate3(&input, sobel_x, 0, VISION_BORDER_CLAMP, &output));
        close_number(output.pixels[1][1], direction * 80);
        assert(vision_filter(&input, VISION_SOBEL, VISION_BORDER_CLAMP, &output));
        close_number(output.pixels[1][1], 80);
    }
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            input.pixels[row][column] = 10 * (double)row;
        }
    }
    assert(vision_filter(&input, VISION_SOBEL, VISION_BORDER_CLAMP, &output));
    close_number(output.pixels[1][1], 80);
}

/**
 * @brief 颜色灰度权重、显示限幅与非有限失败不清空画布。
 */
static void test_conversion(void)
{
    static GfxCanvas canvas;
    static GfxCanvas saved;
    static VisionImage image;
    gfx_clear(&canvas, gfx_rgb(255, 255, 255));
    canvas.pixels[0][0] = gfx_rgb(255, 0, 0);
    canvas.pixels[0][1] = gfx_rgb(0, 255, 0);
    canvas.pixels[0][2] = gfx_rgb(0, 0, 255);
    assert(vision_from_canvas(&canvas, &image));
    close_number(image.pixels[0][0], 76.245);
    close_number(image.pixels[0][1], 149.685);
    close_number(image.pixels[0][2], 29.07);
    close_number(image.pixels[1][1], 255);
    image.width = 3;
    image.height = 1;
    image.pixels[0][0] = -100;
    image.pixels[0][1] = 100.5;
    image.pixels[0][2] = 1000;
    assert(vision_to_canvas(&image, &canvas));
    assert(canvas.pixels[0][0] == gfx_rgb(0, 0, 0));
    assert(canvas.pixels[0][1] == gfx_rgb(101, 101, 101));
    assert(canvas.pixels[0][2] == gfx_rgb(255, 255, 255));
    assert(canvas.pixels[1][0] == gfx_rgb(0, 0, 0));
    saved = canvas;
    image.pixels[0][0] = NAN;
    assert(!vision_to_canvas(&image, &canvas));
    assert(memcmp(&saved, &canvas, sizeof(canvas)) == 0);
    assert(!vision_from_canvas(nullptr, &image));
    assert(!vision_from_canvas(&canvas, nullptr));
}

/**
 * @brief 非法尺寸、参数与极端数值失败时保留整个输出。
 */
static void test_failures(void)
{
    static VisionImage input;
    static VisionImage output;
    static VisionImage saved;
    input = (VisionImage){.width = 1, .height = 1};
    output = (VisionImage){.width = 1, .height = 1};
    output.pixels[0][0] = 77;
    saved = output;
    double kernel[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
    assert(!vision_correlate3(nullptr, kernel, 0, VISION_BORDER_CLAMP, &output));
    assert(!vision_correlate3(&input, nullptr, 0, VISION_BORDER_CLAMP, &output));
    assert(!vision_correlate3(&input, kernel, 0, VISION_BORDER_CLAMP, nullptr));
    assert(!vision_correlate3(&input, kernel, NAN, VISION_BORDER_CLAMP, &output));
    assert(!vision_correlate3(&input, kernel, 0, (VisionBorder)99, &output));
    assert(!vision_filter(&input, VISION_FILTER_COUNT, VISION_BORDER_CLAMP, &output));
    assert(!vision_filter(&input, (VisionFilter)-1, VISION_BORDER_CLAMP, &output));
    kernel[0] = INFINITY;
    assert(!vision_correlate3(&input, kernel, 0, VISION_BORDER_CLAMP, &output));
    kernel[0] = 0;
    input.pixels[0][0] = NAN;
    assert(!vision_filter(&input, VISION_BOX_BLUR, VISION_BORDER_CLAMP, &output));
    input.pixels[0][0] = DBL_MAX;
    kernel[4] = 2;
    assert(!vision_correlate3(&input, kernel, 0, VISION_BORDER_CLAMP, &output));
    input.pixels[0][0] = 0;
    input.width = GFX_WIDTH + 1;
    assert(!vision_filter(&input, VISION_BOX_BLUR, VISION_BORDER_CLAMP, &output));
    input.width = 0;
    assert(!vision_filter(&input, VISION_BOX_BLUR, VISION_BORDER_CLAMP, &output));
    assert(memcmp(&saved, &output, sizeof(output)) == 0);
}

/**
 * @brief 执行无文件写入的确定性核心测试。
 * @return 所有断言通过时为零。
 */
int main(void)
{
    test_constant_and_borders();
    test_impulse_and_alias();
    test_sobel();
    test_conversion();
    test_failures();
    puts("滤镜测试通过：手算响应、互相关方向、边界、Sobel、原地处理与失败保留。");
    return 0;
}
