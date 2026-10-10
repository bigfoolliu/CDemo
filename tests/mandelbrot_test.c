/** @file @brief 逃逸次数、视图坐标、锚点不变、失败安全与像素测试。 */
#include "../src/graphics/mandelbrot.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

/** @brief 验证解析可知的点及最后一步逃逸，不将达到上限等同于逃逸失败。 */
static void test_escape(void)
{
    MandelbrotEscape result;
    const GfxPoint bounded[] = {{0,0},{-1,0},{-2,0}};
    for (size_t i = 0; i < sizeof bounded / sizeof bounded[0]; ++i) {
        assert(mandelbrot_escape(bounded[i], 128, &result));
        assert(!result.escaped && result.iterations == 128);
    }
    assert(mandelbrot_escape((GfxPoint){3,0}, 1, &result) && result.escaped && result.iterations == 1);
    assert(mandelbrot_escape((GfxPoint){2,0}, 1, &result) && !result.escaped);
    assert(mandelbrot_escape((GfxPoint){2,0}, 2, &result) && result.escaped && result.iterations == 2);
    assert(mandelbrot_escape((GfxPoint){1,0}, 3, &result) && result.escaped && result.iterations == 3);
    for (int i = -10; i <= 10; ++i) {
        MandelbrotEscape upper, lower;
        assert(mandelbrot_escape((GfxPoint){-0.7, i / 10.0}, 128, &upper));
        assert(mandelbrot_escape((GfxPoint){-0.7, -i / 10.0}, 128, &lower));
        assert(upper.iterations == lower.iterations && upper.escaped == lower.escaped);
    }
    result = (MandelbrotEscape){7,true};
    assert(!mandelbrot_escape((GfxPoint){NAN,0}, 128, &result));
    assert(!mandelbrot_escape((GfxPoint){DBL_MAX,0}, 128, &result));
    assert(!mandelbrot_escape((GfxPoint){0,0}, 0, &result));
    assert(!mandelbrot_escape((GfxPoint){0,0}, 1025, &result));
    assert(result.iterations == 7 && result.escaped);
}

/**
 * @brief 比较视图字段，避免比较结构体填充字节。
 * @param a 第一个视图。
 * @param b 第二个视图。
 * @return 字段全部相同时 true。
 */
static bool same_view(MandelbrotView a, MandelbrotView b)
{
    return a.center_real == b.center_real && a.center_imag == b.center_imag &&
           a.span == b.span && a.max_iterations == b.max_iterations;
}

/** @brief 坐标映射、像素比例、锚点缩放、平移与极端输入不变性。 */
static void test_view(void)
{
    MandelbrotView view;
    mandelbrot_reset(&view);
    GfxPoint point;
    assert(mandelbrot_map(&view, 0.5, 0.5, &point) && point.x == -0.5 && point.y == 0);
    assert(mandelbrot_map(&view, 0, 0, &point) && point.x == -2.25 && point.y == 1.3125);
    assert(mandelbrot_map(&view, 1, 1, &point) && point.x == 1.25 && point.y == -1.3125);
    GfxPoint x_step, y_step, center;
    assert(mandelbrot_map(&view, 0.5, 0.5, &center));
    assert(mandelbrot_map(&view, 0.5 + 1.0 / GFX_WIDTH, 0.5, &x_step));
    assert(mandelbrot_map(&view, 0.5, 0.5 - 1.0 / GFX_HEIGHT, &y_step));
    assert(fabs((x_step.x - center.x) - (y_step.y - center.y)) < 1e-14);
    GfxPoint before, after;
    assert(mandelbrot_map(&view, 0.25, 0.75, &before));
    assert(mandelbrot_zoom(&view, 0.25, 0.75, 0.5) && view.span == 1.75);
    assert(mandelbrot_map(&view, 0.25, 0.75, &after));
    assert(before.x == after.x && before.y == after.y);
    assert(mandelbrot_zoom(&view, 0.25, 0.75, 2) && view.span == 3.5 && view.center_real == -0.5);
    assert(mandelbrot_pan(&view, 0.25, 0.25) && view.center_real == 0.375 && view.center_imag == 0.65625);
    MandelbrotView previous = view;
    assert(!mandelbrot_zoom(&view, 0.5, 0.5, 2));
    assert(!mandelbrot_zoom(&view, 0.5, 0.5, DBL_MAX));
    assert(!mandelbrot_zoom(&view, 0.5, 0.5, 0));
    assert(!mandelbrot_zoom(&view, NAN, 0.5, 0.5));
    assert(!mandelbrot_pan(&view, DBL_MAX, 0));
    assert(!mandelbrot_pan(&view, 0, INFINITY));
    assert(same_view(view, previous));
    view = (MandelbrotView){-.75,0,1e-10,128};
    previous = view;
    assert(!mandelbrot_zoom(&view, 0.5, 0.5, 0.5) && same_view(view, previous));
    assert(mandelbrot_map(&view, 0.5, 0.5, &center));
    assert(mandelbrot_map(&view, 0.5 + 1.0 / GFX_WIDTH, 0.5, &x_step) && x_step.x > center.x);
    point = (GfxPoint){7,8};
    view.span = NAN;
    assert(!mandelbrot_map(&view, 0.5, 0.5, &point) && point.x == 7 && point.y == 8);
}

/** @brief 默认画布共轭对称、有内外颜色，并验证非法视图不改画布。 */
static void test_pixels(void)
{
    MandelbrotView view;
    mandelbrot_reset(&view);
    static GfxCanvas a, b;
    assert(mandelbrot_draw(&view, &a) && mandelbrot_draw(&view, &b));
    size_t black = 0, colored = 0;
    for (int y = 0; y < GFX_HEIGHT; ++y) {
        for (int x = 0; x < GFX_WIDTH; ++x) {
            assert(a.pixels[y][x] == b.pixels[y][x]);
            assert(a.pixels[y][x] == a.pixels[GFX_HEIGHT - 1 - y][x]);
            if (a.pixels[y][x] == gfx_rgb(0,0,0)) { ++black; } else { ++colored; }
        }
    }
    assert(black > 0 && colored > 0 && a.pixels[120][205] == gfx_rgb(0,0,0));
    view.max_iterations = 0;
    assert(!mandelbrot_draw(&view, &b));
    for (int y = 0; y < GFX_HEIGHT; ++y) {
        for (int x = 0; x < GFX_WIDTH; ++x) { assert(a.pixels[y][x] == b.pixels[y][x]); }
    }
    /* 验证恰在第 32 次逃逸的像素也有外部颜色，不能误画为黑色。 */
    mandelbrot_reset(&view); view.max_iterations = 32;
    assert(mandelbrot_draw(&view, &b));
    bool found_last = false;
    for (int y = 0; y < GFX_HEIGHT && !found_last; ++y) {
        for (int x = 0; x < GFX_WIDTH; ++x) {
            double step = view.span / GFX_WIDTH;
            GfxPoint c = {view.center_real + (x + 0.5 - GFX_WIDTH / 2.0) * step,
                          view.center_imag + (GFX_HEIGHT / 2.0 - y - 0.5) * step};
            MandelbrotEscape result;
            assert(mandelbrot_escape(c, 32, &result));
            if (result.escaped && result.iterations == 32) {
                assert(b.pixels[y][x] == gfx_rgb(255,220,120));
                found_last = true;
                break;
            }
        }
    }
    assert(found_last);
    /* 视图移到 c≈1.1，上限 32；此处应绘成外部颜色，而不是黑色。 */
    view = (MandelbrotView){1.1,0,1e-3,32};
    assert(mandelbrot_draw(&view, &b) && b.pixels[120][160] != gfx_rgb(0,0,0));
}

/** @brief 执行全部分形数值与像素断言。 @return 成功返回零。 */
int main(void)
{
    test_escape(); test_view(); test_pixels();
    puts("曼德勃罗测试通过：逃逸次数、坐标、锚点、边界、失败安全与像素。");
    return 0;
}
