/** @file @brief 逐级线性插值构造三次曲线，采样后用折线近似显示。 */
#include "bezier.h"
#include <math.h>

/** @brief 验证点为画布内有限坐标。 @param point 待验证的坐标。 */
static bool valid_point(GfxPoint point)
{
    return isfinite(point.x) && isfinite(point.y) &&
           point.x >= 0 && point.x <= GFX_WIDTH - 1 &&
           point.y >= 0 && point.y <= GFX_HEIGHT - 1;
}

/** @brief 验证全部控制点。 @param curve 待验证曲线，可以为 nullptr。 */
static bool valid_curve(const BezierCurve *curve)
{
    if (curve == nullptr) { return false; }
    for (int i = 0; i < BEZIER_POINT_COUNT; ++i) {
        if (!valid_point(curve->points[i])) { return false; }
    }
    return true;
}

/** @copydoc bezier_reset */
void bezier_reset(BezierCurve *curve)
{
    if (curve != nullptr) {
        *curve = (BezierCurve){.points = {{30, 190}, {80, 30}, {240, 210}, {290, 50}}};
    }
}

/** @brief 插值核心，调用前须验证。
 * @param curve 有效曲线。
 * @param t 有效的 [0,1] 参数。
 * @return 曲线上的坐标；局部数组保证输出别名不会破坏计算。
 */
static GfxPoint evaluate_unchecked(const BezierCurve *curve, double t)
{
    GfxPoint work[BEZIER_POINT_COUNT];
    for (int i = 0; i < BEZIER_POINT_COUNT; ++i) { work[i] = curve->points[i]; }
    /* 四点 -> 三点 -> 两点 -> 一点，每层都使用相同参数 t。
     * 加权平均保证曲线位于控制点凸包内，不会超出画布。
     */
    for (int count = BEZIER_POINT_COUNT - 1; count > 0; --count) {
        for (int i = 0; i < count; ++i) {
            work[i].x = (1 - t) * work[i].x + t * work[i + 1].x;
            work[i].y = (1 - t) * work[i].y + t * work[i + 1].y;
        }
    }
    return work[0];
}

/** @copydoc bezier_evaluate */
bool bezier_evaluate(const BezierCurve *curve, double t, GfxPoint *output)
{
    if (!valid_curve(curve) || output == nullptr || !isfinite(t) || t < 0 || t > 1) {
        return false;
    }
    *output = evaluate_unchecked(curve, t);
    return true;
}

/** @copydoc bezier_move */
bool bezier_move(BezierCurve *curve, int index, GfxPoint point)
{
    if (!valid_curve(curve) || index < 0 || index >= BEZIER_POINT_COUNT || !valid_point(point)) {
        return false;
    }
    curve->points[index] = point;
    return true;
}

/** @copydoc bezier_hit */
int bezier_hit(const BezierCurve *curve, GfxPoint point, double radius)
{
    if (!valid_curve(curve) || !valid_point(point) || !isfinite(radius) ||
        radius < 0 || radius > GFX_COORD_LIMIT) { return -1; }
    int found = -1;
    double best = radius * radius;
    for (int i = 0; i < BEZIER_POINT_COUNT; ++i) {
        double dx = point.x - curve->points[i].x, dy = point.y - curve->points[i].y;
        double distance = dx * dx + dy * dy;
        if (distance <= best && (found == -1 || distance < best)) {
            found = i;
            best = distance;
        }
    }
    return found;
}

/** @copydoc bezier_draw */
bool bezier_draw(const BezierCurve *curve, int selected, GfxCanvas *canvas)
{
    if (!valid_curve(curve) || canvas == nullptr || selected < -1 || selected >= BEZIER_POINT_COUNT) {
        return false;
    }
    gfx_clear(canvas, gfx_rgb(18, 22, 32));
    for (int i = 1; i < BEZIER_POINT_COUNT; ++i) {
        gfx_line(canvas, (int)lround(curve->points[i - 1].x), (int)lround(curve->points[i - 1].y),
                 (int)lround(curve->points[i].x), (int)lround(curve->points[i].y), gfx_rgb(90, 100, 120));
    }
    GfxPoint previous = curve->points[0];
    /* 均匀参数采样不是均匀弧长采样；128 段是固定画布上的显示近似。 */
    for (int i = 1; i <= BEZIER_SEGMENTS; ++i) {
        GfxPoint next = evaluate_unchecked(curve, (double)i / BEZIER_SEGMENTS);
        gfx_line(canvas, (int)lround(previous.x), (int)lround(previous.y),
                 (int)lround(next.x), (int)lround(next.y), gfx_rgb(80, 220, 240));
        previous = next;
    }
    for (int i = 0; i < BEZIER_POINT_COUNT; ++i) {
        int x = (int)lround(curve->points[i].x), y = (int)lround(curve->points[i].y);
        uint32_t color = i == 0 ? gfx_rgb(80, 220, 130) :
                         i == 3 ? gfx_rgb(240, 110, 110) : gfx_rgb(100, 150, 255);
        if (i == selected) { gfx_circle(canvas, x, y, 7, gfx_rgb(255, 220, 80)); }
        gfx_rect(canvas, x - 3, y - 3, 7, 7, true, color);
    }
    return true;
}
