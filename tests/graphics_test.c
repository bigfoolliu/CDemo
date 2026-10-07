/** @file
 * @brief 无窗口验证像素、栅格化边界及变换顺序；不依赖 SDL。
 */
#include "../src/graphics/graphics.h"
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

static GfxCanvas canvas;
static const uint32_t INK = UINT32_C(0xff123456);

/**
 * @brief 统计被绘制的像素，检测范围外请求是否改动画布。
 * @return 当前颜色等于 INK 的像素数。
 */
static size_t ink_count(void)
{
    size_t count = 0;
    for (int y = 0; y < GFX_HEIGHT; ++y) {
        for (int x = 0; x < GFX_WIDTH; ++x) {
            count += canvas.pixels[y][x] == INK;
        }
    }
    return count;
}

/** @brief 验证颜色、全部像素清屏和画布边界。 */
static void test_pixels(void)
{
    assert(gfx_rgb(0x12, 0x34, 0x56) == INK);
    gfx_clear(&canvas, INK);
    assert(ink_count() == (size_t)GFX_WIDTH * GFX_HEIGHT);
    gfx_clear(&canvas, 0);
    gfx_pixel(&canvas, 0, 0, INK);
    gfx_pixel(&canvas, GFX_WIDTH - 1, GFX_HEIGHT - 1, INK);
    gfx_pixel(&canvas, -1, 0, INK);
    gfx_pixel(&canvas, GFX_WIDTH, 0, INK);
    gfx_pixel(&canvas, 0, -1, INK);
    gfx_pixel(&canvas, 0, GFX_HEIGHT, INK);
    gfx_pixel(&canvas, INT_MIN, INT_MAX, INK);
    assert(ink_count() == 2);
}

/** @brief 验证八个方向、单点、端点和部分可见直线。 */
static void test_lines(void)
{
    const int offsets[][2] = {{8, 3}, {3, 8}, {-3, 8}, {-8, 3},
                             {-8, -3}, {-3, -8}, {3, -8}, {8, -3},
                             {8, 0}, {0, 8}, {-8, 0}, {0, -8}, {8, 8}};
    for (size_t i = 0; i < sizeof(offsets) / sizeof(offsets[0]); ++i) {
        gfx_clear(&canvas, 0);
        int x = 20 + offsets[i][0], y = 20 + offsets[i][1];
        gfx_line(&canvas, 20, 20, x, y, INK);
        assert(canvas.pixels[20][20] == INK && canvas.pixels[y][x] == INK);
        assert(ink_count() == 9);
    }
    gfx_clear(&canvas, 0);
    gfx_line(&canvas, 2, 2, 2, 2, INK);
    assert(ink_count() == 1);
    gfx_clear(&canvas, 0);
    gfx_line(&canvas, -5, 10, 5, 10, INK);
    assert(ink_count() == 6);
    for (int x = 0; x <= 5; ++x) {
        assert(canvas.pixels[10][x] == INK);
    }
    gfx_clear(&canvas, 0);
    gfx_line(&canvas, INT_MIN, 0, INT_MAX, 0, INK);
    assert(ink_count() == 0);
    gfx_line(&canvas, -GFX_COORD_LIMIT, 0, GFX_COORD_LIMIT, 0, INK);
    assert(ink_count() == GFX_WIDTH);
}

/** @brief 验证描边、填充、单像素、非正尺寸和范围外矩形。 */
static void test_rectangles(void)
{
    gfx_clear(&canvas, 0);
    gfx_rect(&canvas, 10, 10, 5, 4, false, INK);
    assert(ink_count() == 14 && canvas.pixels[11][11] == 0);
    gfx_rect(&canvas, 10, 10, 5, 4, true, INK);
    assert(ink_count() == 20 && canvas.pixels[11][11] == INK);
    gfx_clear(&canvas, 0);
    gfx_rect(&canvas, -2, -2, 4, 4, true, INK);
    assert(ink_count() == 4);
    gfx_clear(&canvas, 0);
    gfx_rect(&canvas, 0, 0, 1, 1, false, INK);
    assert(ink_count() == 1);
    gfx_clear(&canvas, 0);
    gfx_rect(&canvas, 0, 0, 0, 10, true, INK);
    gfx_rect(&canvas, 0, 0, 10, -1, false, INK);
    gfx_rect(&canvas, INT_MAX, 0, INT_MAX, 1, true, INK);
    gfx_rect(&canvas, GFX_COORD_LIMIT, 0, 2, 1, false, INK);
    assert(ink_count() == 0);
}

/** @brief 验证圆的八向对称、轴交点、零半径和裁切。 */
static void test_circles(void)
{
    gfx_clear(&canvas, 0);
    gfx_circle(&canvas, 20, 20, 3, INK);
    assert(canvas.pixels[20][23] == INK && canvas.pixels[23][20] == INK);
    for (int y = -3; y <= 3; ++y) {
        for (int x = -3; x <= 3; ++x) {
            uint32_t color = canvas.pixels[20 + y][20 + x];
            assert(color == canvas.pixels[20 - y][20 + x]);
            assert(color == canvas.pixels[20 + y][20 - x]);
            assert(color == canvas.pixels[20 + x][20 + y]);
        }
    }
    assert(canvas.pixels[20][20] == 0); /* 圆周而非填充圆。 */
    gfx_clear(&canvas, 0);
    gfx_circle(&canvas, 0, 0, 0, INK);
    assert(ink_count() == 1);
    gfx_clear(&canvas, 0);
    gfx_circle(&canvas, -2, 0, 2, INK);
    assert(canvas.pixels[0][0] == INK);
    gfx_clear(&canvas, 0);
    gfx_circle(&canvas, 0, 0, -1, INK);
    gfx_circle(&canvas, INT_MAX, INT_MIN, INT_MAX, INK);
    assert(ink_count() == 0);
}

/** @brief 验证缩放、旋转、平移顺序以及失败不改输出。 */
static void test_transform(void)
{
    GfxPoint result = {0};
    bool ok = gfx_transform_point((GfxPoint){1, 0}, (GfxPoint){10, 20}, acos(-1) / 2, 2, &result);
    assert(ok && fabs(result.x - 10) < 1e-10 && fabs(result.y - 22) < 1e-10);
    ok = gfx_transform_point((GfxPoint){2, 3}, (GfxPoint){4, 5}, 0, -1, &result);
    assert(ok && result.x == 2 && result.y == 2);
    ok = gfx_transform_point((GfxPoint){2, 3}, (GfxPoint){4, 5}, 0, 0, &result);
    assert(ok && result.x == 4 && result.y == 5);
    ok = gfx_transform_point((GfxPoint){NAN, 0}, (GfxPoint){0}, 0, 1, &result);
    assert(!ok && result.x == 4 && result.y == 5);
    ok = gfx_transform_point((GfxPoint){DBL_MAX, 0}, (GfxPoint){0}, 0, DBL_MAX, &result);
    assert(!ok && result.x == 4 && result.y == 5);
    ok = gfx_transform_point((GfxPoint){GFX_COORD_LIMIT + 1, 0}, (GfxPoint){0}, 0, 1, &result);
    assert(!ok && result.x == 4 && result.y == 5);
}

/** @brief 运行无窗口测试。 @return 正常完成为 0，断言失败则终止程序。 */
int main(void)
{
    test_pixels();
    test_lines();
    test_rectangles();
    test_circles();
    test_transform();
    puts("graphics：颜色、裁切、直线、矩形、圆和二维变换验证通过");
    return 0;
}
