/** @file @brief 曲线端点、插值、凸包边界、编辑失败与退化绘图验证。 */
#include "../src/graphics/bezier.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/** @brief 验证近似相等。 @param a 实际数值。 @param b 期望数值。 */
static void close_number(double a, double b)
{
    assert(fabs(a - b) < 1e-9);
}

/** @brief 用独立的 Bernstein 多项式检查插值和端点，不重复实现算法。 */
static void test_interpolation(void)
{
    BezierCurve curve;
    bezier_reset(&curve);
    for (int sample = 0; sample <= 100; ++sample) {
        double t = sample / 100.0, u = 1 - t;
        double weights[] = {u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t};
        GfxPoint expected = {0}, actual;
        for (int i = 0; i < BEZIER_POINT_COUNT; ++i) {
            expected.x += weights[i] * curve.points[i].x;
            expected.y += weights[i] * curve.points[i].y;
        }
        assert(bezier_evaluate(&curve, t, &actual));
        close_number(actual.x, expected.x);
        close_number(actual.y, expected.y);
        assert(actual.x >= 30 && actual.x <= 290 && actual.y >= 30 && actual.y <= 210);
    }
    curve = (BezierCurve){.points = {{0, 0}, {100, 60}, {200, 120}, {300, 180}}};
    GfxPoint point;
    assert(bezier_evaluate(&curve, 0.25, &point));
    close_number(point.x, 75);
    close_number(point.y, 45);
    assert(bezier_evaluate(&curve, 1, &curve.points[0])); /* 输出允许别名。 */
    close_number(curve.points[0].x, 300);
}

/** @brief 无效输入保持调用者状态；命中选择应最近且稳定。 */
static void test_editing(void)
{
    BezierCurve curve;
    bezier_reset(&curve);
    BezierCurve before = curve;
    GfxPoint output = {7, 9};
    const double invalid[] = {-0.1, 1.1, NAN, INFINITY};
    for (unsigned i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        assert(!bezier_evaluate(&curve, invalid[i], &output));
        assert(output.x == 7 && output.y == 9);
    }
    assert(!bezier_evaluate(nullptr, 0, &output));
    assert(!bezier_evaluate(&curve, 0, nullptr));
    assert(!bezier_move(&curve, -1, (GfxPoint){0, 0}));
    assert(!bezier_move(&curve, 4, (GfxPoint){0, 0}));
    assert(!bezier_move(&curve, 0, (GfxPoint){NAN, 0}));
    assert(!bezier_move(&curve, 0, (GfxPoint){320, 239}));
    assert(!bezier_move(&curve, 0, (GfxPoint){0, -1}));
    assert(memcmp(&before, &curve, sizeof curve) == 0);
    assert(bezier_move(&curve, 0, (GfxPoint){0, 0}));
    assert(bezier_move(&curve, 3, (GfxPoint){319, 239}));
    assert(bezier_hit(&curve, (GfxPoint){0, 0}, 0) == 0);
    assert(bezier_hit(&curve, (GfxPoint){10, 10}, 1) == -1);
    assert(bezier_hit(&curve, (GfxPoint){0, 0}, NAN) == -1);
    assert(bezier_hit(&curve, (GfxPoint){0, 0}, -1) == -1);
    curve.points[1] = (GfxPoint){2, 0};
    assert(bezier_hit(&curve, (GfxPoint){1, 0}, 1) == 0); /* 等距选较小编号。 */
    assert(bezier_hit(&curve, (GfxPoint){1.5, 0}, 2) == 1);
}

/** @brief 边缘和重合控制点可以绘制；非法输入不得清空已有画布。 */
static void test_drawing(void)
{
    static GfxCanvas canvas, before;
    BezierCurve curve = {.points = {{0, 0}, {0, 0}, {0, 0}, {0, 0}}};
    GfxPoint point;
    assert(bezier_evaluate(&curve, 0.5, &point));
    assert(point.x == 0 && point.y == 0);
    assert(bezier_hit(&curve, point, 0) == 0);
    assert(bezier_draw(&curve, 0, &canvas));
    assert(canvas.pixels[0][0] == gfx_rgb(240, 110, 110));
    before = canvas;
    assert(!bezier_draw(&curve, 4, &canvas));
    assert(!bezier_draw(nullptr, -1, &canvas));
    curve.points[2].y = INFINITY;
    assert(!bezier_draw(&curve, -1, &canvas));
    assert(memcmp(&before, &canvas, sizeof canvas) == 0);
    bezier_reset(&curve);
    assert(bezier_draw(&curve, -1, &canvas));
    assert(canvas.pixels[190][30] == gfx_rgb(80, 220, 130));
    assert(canvas.pixels[50][290] == gfx_rgb(240, 110, 110));
    assert(canvas.pixels[120][160] == gfx_rgb(80, 220, 240));
}

/** @brief 运行无需 SDL 的测试。 @return 成功返回零，失败由断言终止。 */
int main(void)
{
    test_interpolation();
    test_editing();
    test_drawing();
    puts("贝塞尔曲线核心测试通过。");
    return 0;
}
